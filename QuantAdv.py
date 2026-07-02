#!/usr/bin/env python
# coding: utf-8

import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import pandas as pd
import copy
import os
import traceback
import sys
import matplotlib.pyplot as plt
import seaborn as sns

import torchattacks
from autoattack import AutoAttack

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
SWEEP_CSV = os.path.join(DATA_DIR, "results_sweep.csv")
PLOT_PNG = os.path.join(DATA_DIR, "accuracy_plot.png")
CIFAR10_ROOT = os.environ.get("CIFAR10_ROOT", "./")

missing = [pkg for pkg in ("torchattacks", "autoattack") if importlib.util.find_spec(pkg) is None]
if missing:
    raise ImportError(f"Missing packages: {missing}.\nInstall via: pip install -r requirements.txt")
print("All required packages are available.")

expected = os.path.join(CIFAR10_ROOT, "cifar-10-batches-py")
if not os.path.isdir(expected):
    raise FileNotFoundError(f"Expected extracted CIFAR-10 at {expected!r}")

def get_dataloaders(batch_size=100, eval_n=500, finetune_n=4000):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    train_full = torchvision.datasets.CIFAR10(root=CIFAR10_ROOT, train=True, download=False, transform=transform_train)
    test_full = torchvision.datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=False, transform=transform_test)

    finetune_subset = torch.utils.data.Subset(train_full, list(range(finetune_n)))
    eval_subset = torch.utils.data.Subset(test_full, list(range(eval_n)))
    
    workers = min(4, os.cpu_count() or 1)

    finetune_loader = torch.utils.data.DataLoader(
        finetune_subset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_subset, batch_size=100, shuffle=False, num_workers=workers, pin_memory=True
    )

    return finetune_loader, eval_loader

PRETRAINED_NAMES = {
    "ResNet20": "cifar10_resnet20",
    "ResNet56": "cifar10_resnet56",
    "MobileNetV2": "cifar10_mobilenetv2_x1_0",
    "VGG16_BN": "cifar10_vgg16_bn",
    "ShuffleNetV2": "cifar10_shufflenetv2_x1_0",
    "RepVGG_A0": "cifar10_repvgg_a0"
}

def load_pretrained(arch_key):
    hub_name = PRETRAINED_NAMES[arch_key]
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", hub_name, pretrained=True)
    return model.to(device).eval()

def sanity_check_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def quantize_tensor(t, bits, use_ste):
    if bits is None:
        return t
    qmax = 2 ** (bits - 1) - 1
    scale = torch.clamp(t.detach().abs().max() / qmax, min=1e-8)
    t_scaled = t / scale
    t_round = FakeQuantSTE.apply(t_scaled) if use_ste else torch.round(t_scaled)
    t_round = torch.clamp(t_round, -qmax - 1, qmax)
    return t_round * scale

class QuantConv2d(nn.Conv2d):
    def forward(self, x):
        bits = getattr(self, 'bits', None)
        use_ste = getattr(self, 'use_ste', False)
        quant_weight = getattr(self, 'quant_weight', True)
        quant_act = getattr(self, 'quant_act', True)
        
        w = quantize_tensor(self.weight, bits, use_ste) if quant_weight else self.weight
        out = self._conv_forward(x, w, self.bias)
        if quant_act:
            out = quantize_tensor(out, bits, use_ste)
        return out

class QuantLinear(nn.Linear):
    def forward(self, x):
        bits = getattr(self, 'bits', None)
        use_ste = getattr(self, 'use_ste', False)
        quant_weight = getattr(self, 'quant_weight', True)
        quant_act = getattr(self, 'quant_act', True)
        
        w = quantize_tensor(self.weight, bits, use_ste) if quant_weight else self.weight
        out = F.linear(x, w, self.bias)
        if quant_act:
            out = quantize_tensor(out, bits, use_ste)
        return out

def _to_quant_module(mod, bits, quant_weight=True, quant_act=True):
    if isinstance(mod, nn.Conv2d):
        new = QuantConv2d(mod.in_channels, mod.out_channels, mod.kernel_size,
                          mod.stride, mod.padding, mod.dilation, mod.groups,
                          mod.bias is not None, mod.padding_mode)
        new.weight = mod.weight
        if mod.bias is not None:
            new.bias = mod.bias
        new.bits, new.use_ste, new.quant_weight, new.quant_act = bits, False, quant_weight, quant_act
        return new
    if isinstance(mod, nn.Linear):
        new = QuantLinear(mod.in_features, mod.out_features, bias=mod.bias is not None)
        new.weight = mod.weight
        if mod.bias is not None:
            new.bias = mod.bias
        new.bits, new.use_ste, new.quant_weight, new.quant_act = bits, False, quant_weight, quant_act
        return new
    return None

def _replace_recursive(module, bits, quant_weight=True, quant_act=True):
    for name, child in list(module.named_children()):
        nc = _to_quant_module(child, bits, quant_weight, quant_act)
        if nc is not None:
            setattr(module, name, nc)
        else:
            _replace_recursive(child, bits, quant_weight, quant_act)

def convert_to_quant(model, bits, quant_weight=True, quant_act=True):
    m = copy.deepcopy(model)
    _replace_recursive(m, bits, quant_weight, quant_act)
    return m

def set_ste_mode(model, flag):
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            mod.use_ste = flag

def count_quant_layers(model):
    return sum(1 for m in model.modules() if isinstance(m, (QuantConv2d, QuantLinear)))

def prepare_qat(fp32_model, bits, finetune_loader, epochs=3, lr=1e-3):
    m = convert_to_quant(fp32_model, bits, quant_weight=True, quant_act=True)
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
        
    set_ste_mode(m, True)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(epochs):
        running = 0.0
        for x, y in finetune_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"  QAT epoch {epoch+1}/{epochs} avg loss {running/len(finetune_loader):.4f}")
    set_ste_mode(m, False)
    return m.eval()

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
CLIP_MIN = ((0.0 - CIFAR_MEAN) / CIFAR_STD)
CLIP_MAX = ((1.0 - CIFAR_MEAN) / CIFAR_STD)

def run_fgsm_pgd(model, loader, eps=8/255):
    model.eval()
    fgsm = torchattacks.FGSM(model, eps=eps)
    pgd = torchattacks.PGD(model, eps=eps, alpha=2/255, steps=20, random_start=True)
    out = {}
    for name, atk in [("FGSM", fgsm), ("PGD", pgd)]:
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_adv = atk(x, y)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        out[name] = correct / total
    return out

def run_autoattack(model, loader, eps=8/255):
    model.eval()
    adversary = AutoAttack(model, norm="Linf", eps=eps, version="standard", device=device, verbose=False)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def transfer_attack(source_model, target_model, loader, eps=8/255):
    pgd = torchattacks.PGD(source_model, eps=eps, alpha=2/255, steps=20, random_start=True)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd(x, y)
        with torch.no_grad():
            pred = target_model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

def bpda_pgd_attack(model, x, y, eps=8/255, alpha=2/255, steps=20):
    set_ste_mode(model, True)
    clip_min = CLIP_MIN.to(device)
    clip_max = CLIP_MAX.to(device)
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    set_ste_mode(model, False)
    return x_adv.detach()

def run_bpda(model, loader, eps=8/255):
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = bpda_pgd_attack(model, x, y, eps=eps)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

def gradient_diagnostics(model, loader, fp32_ref=None):
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    set_ste_mode(model, False)
    x_in = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_in), y)
    g_hard = torch.autograd.grad(loss, x_in)[0].flatten()

    set_ste_mode(model, True)
    x_in2 = x.clone().requires_grad_(True)
    loss2 = F.cross_entropy(model(x_in2), y)
    g_ste = torch.autograd.grad(loss2, x_in2)[0].flatten()
    set_ste_mode(model, False)

    diagnostics = {
        "frac_zero_grad_hard": (g_hard.abs() < 1e-8).float().mean().item(),
        "frac_zero_grad_ste": (g_ste.abs() < 1e-8).float().mean().item(),
        "grad_norm_hard": g_hard.norm().item(),
        "grad_norm_ste": g_ste.norm().item(),
    }

    if fp32_ref is not None:
        fp32_ref.eval()
        x_ref = x.clone().requires_grad_(True)
        loss_ref = F.cross_entropy(fp32_ref(x_ref), y)
        g_ref = torch.autograd.grad(loss_ref, x_ref)[0].flatten()
        
        cos_sim = F.cosine_similarity(g_ste.unsqueeze(0), g_ref.unsqueeze(0)).item()
        diagnostics["grad_cosine_sim_with_FP32"] = cos_sim

    return diagnostics

def run_suite(model, loader, name, fp32_ref=None, eps=8/255):
    model.eval()
    results = {"model": name}

    try:
        results["clean_acc"] = sanity_check_accuracy(model, loader)
    except Exception as e:
        print(f"  [WARN] clean_acc failed for {name}: {e}")
        results["clean_acc"] = None

    try:
        results.update(run_fgsm_pgd(model, loader, eps=eps))
    except Exception as e:
        print(f"  [WARN] FGSM/PGD failed for {name}: {e}")
        results["FGSM"] = results.get("FGSM", None)
        results["PGD"] = results.get("PGD", None)

    try:
        results["AutoAttack"] = run_autoattack(model, loader, eps=eps)
    except Exception as e:
        print(f"  [WARN] AutoAttack failed for {name}: {e}")
        results["AutoAttack"] = None

    if fp32_ref is not None:
        try:
            results["Transfer_from_FP32"] = transfer_attack(fp32_ref, model, loader, eps=eps)
        except Exception as e:
            print(f"  [WARN] transfer_attack failed for {name}: {e}")
            results["Transfer_from_FP32"] = None

    if count_quant_layers(model) > 0:
        try:
            results["BPDA_PGD"] = run_bpda(model, loader, eps=eps)
        except Exception as e:
            print(f"  [WARN] BPDA failed for {name}: {e}")
            results["BPDA_PGD"] = None
        try:
            results.update(gradient_diagnostics(model, loader, fp32_ref=fp32_ref))
        except Exception as e:
            print(f"  [WARN] gradient_diagnostics failed for {name}: {e}")

    return results

def run_epsilon_sweep_for_model(model, loader, name, epsilons):
    rows = []
    is_quant = count_quant_layers(model) > 0
    for eps in epsilons:
        row = {"model": name, "epsilon": eps}
        try:
            pgd = torchattacks.PGD(model, eps=eps, alpha=2/255, steps=20, random_start=True)
            correct, total = 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                x_adv = pgd(x, y)
                with torch.no_grad():
                    pred = model(x_adv).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            row["PGD_acc"] = correct / total
        except Exception as e:
            print(f"  [WARN] PGD sweep failed for {name} eps={eps:.4f}: {e}")
            row["PGD_acc"] = None

        if is_quant:
            try:
                row["BPDA_acc"] = run_bpda(model, loader, eps=eps)
            except Exception as e:
                print(f"  [WARN] BPDA sweep failed for {name} eps={eps:.4f}: {e}")
                row["BPDA_acc"] = None
        rows.append(row)
    return rows

def parallelize(model):
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        return nn.DataParallel(model)
    return model

def main():
    finetune_loader, eval_loader = get_dataloaders()
    model_registry = {}

    for arch_key in PRETRAINED_NAMES:
        print(f"\n>>> {arch_key} <<<")
        try:
            fp32 = load_pretrained(arch_key)
            acc = sanity_check_accuracy(fp32, eval_loader)
            print(f"  loaded pretrained {arch_key}, clean acc: {acc:.3f}")
            model_registry[f"{arch_key}_FP32"] = (fp32, None)
        except Exception as e:
            print(f"  [FAIL] could not load {arch_key}: {e}")
            traceback.print_exc()
            continue

        try:
            int8_ptq = convert_to_quant(fp32, bits=8, quant_weight=True, quant_act=True)
            model_registry[f"{arch_key}_int8_PTQ"] = (int8_ptq, fp32)
        except Exception as e:
            print(f"  [FAIL] int8 PTQ for {arch_key}: {e}")

        try:
            int4_ptq = convert_to_quant(fp32, bits=4, quant_weight=True, quant_act=True)
            model_registry[f"{arch_key}_int4_PTQ"] = (int4_ptq, fp32)
        except Exception as e:
            print(f"  [FAIL] int4 PTQ for {arch_key}: {e}")

        try:
            int8_qat = prepare_qat(fp32, bits=8, finetune_loader=finetune_loader, epochs=3)
            model_registry[f"{arch_key}_int8_QAT"] = (int8_qat, fp32)
        except Exception as e:
            print(f"  [FAIL] int8 QAT for {arch_key}: {e}")
            traceback.print_exc()

    print("\nRegistry built:", list(model_registry.keys()))

    for k in model_registry:
        m, r = model_registry[k]
        model_registry[k] = (parallelize(m), parallelize(r) if r else None)

    if os.path.exists(RESULTS_CSV):
        df_results = pd.read_csv(RESULTS_CSV)
        done = set(df_results["model"].astype(str))
    else:
        df_results = pd.DataFrame(columns=["model"])
        done = set()

    for name, (model, ref) in list(model_registry.items()):
        if name in done:
            print(f"Skipping {name} (already in {RESULTS_CSV})")
            continue

        print(f"\nEvaluating {name} ...")
        try:
            res = run_suite(model, eval_loader, name, fp32_ref=ref)
        except Exception as e:
            print(f"  [FAIL] run_suite failed for {name}: {e}")
            traceback.print_exc()
            res = {"model": name}

        new_row = pd.DataFrame([res])
        df_results = pd.concat([df_results, new_row], ignore_index=True)
        df_results.to_csv(RESULTS_CSV, index=False)

        print("Result:")
        print(new_row.to_string(index=False))
        print("-" * 100)

    print("\nFinal results:")
    print(df_results)

    acc_cols = [c for c in ["clean_acc", "FGSM", "PGD", "AutoAttack", "Transfer_from_FP32", "BPDA_PGD"] if c in df_results.columns]

    if len(acc_cols) > 0:
        df_plot = df_results.melt(id_vars="model", value_vars=acc_cols, var_name="Attack", value_name="Accuracy")

        plt.figure(figsize=(14, 6))
        sns.barplot(data=df_plot, x="model", y="Accuracy", hue="Attack")
        plt.xticks(rotation=45, ha="right")
        plt.title("Model Accuracy under Various Adversarial Attacks")
        plt.ylim(0, 1.0)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
        plt.show()

    SWEEP_EPSILONS = [1/255, 2/255, 4/255, 8/255, 16/255]

    if os.path.exists(SWEEP_CSV):
        df_sweep = pd.read_csv(SWEEP_CSV)
        sweep_done = set(zip(df_sweep["model"].astype(str), df_sweep["epsilon"].round(6)))
    else:
        df_sweep = pd.DataFrame()
        sweep_done = set()

    for name, (model, ref) in model_registry.items():
        print(f"\nSweeping {name} ...")
        pending_eps = [eps for eps in SWEEP_EPSILONS if (name, round(eps, 6)) not in sweep_done]
        if not pending_eps:
            print(f"  Skipping {name} (already done)")
            continue
        try:
            rows = run_epsilon_sweep_for_model(model, eval_loader, name, pending_eps)
            if rows:
                new_sweep = pd.DataFrame(rows)
                df_sweep = pd.concat([df_sweep, new_sweep], ignore_index=True)
                df_sweep.to_csv(SWEEP_CSV, index=False)
        except Exception as e:
            print(f"  [FAIL] epsilon sweep failed for {name}: {e}")
            traceback.print_exc()

    print("\nEpsilon sweep completed. Results saved to", SWEEP_CSV)
    print("All done.")

if __name__ == '__main__':
    main()