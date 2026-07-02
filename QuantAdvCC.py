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
import json
import traceback
import sys

import torchattacks
from autoattack import AutoAttack

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

CIFAR100_ROOT = os.environ.get("CIFAR100_ROOT", os.environ.get("CIFAR10_ROOT", "./"))

SEEDS = [0, 1, 2]

"""
Threaded single-model, single-quantization, single-attack analysis with eplison sweep.
"""

def results_csv_path(model_name):
    return os.path.join(DATA_DIR, f"results_{model_name}.csv")

def sweep_csv_path(model_name):
    return os.path.join(DATA_DIR, f"sweep_{model_name}.csv")

def ablation_csv_path(model_name):
    return os.path.join(DATA_DIR, f"ablation_{model_name}.csv")

def layerwise_csv_path(model_name):
    return os.path.join(DATA_DIR, f"layerwise_{model_name}.csv")

def trajectory_json_path(model_name):
    return os.path.join(DATA_DIR, f"trajectory_{model_name}.json")

# NEW (item 2): weight-only vs activation-only vs both quantization ablation
def component_ablation_csv_path(model_name):
    return os.path.join(DATA_DIR, f"component_ablation_{model_name}.csv")

missing = [pkg for pkg in ("torchattacks", "autoattack") if importlib.util.find_spec(pkg) is None]
if missing:
    raise ImportError(f"Missing packages: {missing}.\nInstall via: pip install -r requirements.txt")
print("All required packages are available.")

expected = os.path.join(CIFAR100_ROOT, "cifar-100-python")
if not os.path.isdir(expected):
    raise FileNotFoundError(f"Expected extracted CIFAR-100 at {expected!r}")

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)

def get_dataloaders(batch_size=128, eval_n=3000, finetune_n=5000, eval_batch_size=250):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
    ])

    train_full = torchvision.datasets.CIFAR100(root=CIFAR100_ROOT, train=True, download=False, transform=transform_train)
    test_full = torchvision.datasets.CIFAR100(root=CIFAR100_ROOT, train=False, download=False, transform=transform_test)

    finetune_subset = torch.utils.data.Subset(train_full, list(range(finetune_n)))
    eval_subset = torch.utils.data.Subset(test_full, list(range(eval_n)))

    workers = min(4, os.cpu_count() or 1)

    finetune_loader = torch.utils.data.DataLoader(
        finetune_subset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_subset, batch_size=eval_batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )

    return finetune_loader, eval_loader

PRETRAINED_NAMES = {
    "ResNet20": "cifar100_resnet20",
    "ResNet56": "cifar100_resnet56",
    "MobileNetV2": "cifar100_mobilenetv2_x1_0",
    "VGG16_BN": "cifar100_vgg16_bn",
    "ShuffleNetV2": "cifar100_shufflenetv2_x1_0",
    "RepVGG_A0": "cifar100_repvgg_a0"
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

# NEW (item 2): flip weight/activation quantization on an already-built
# quantized model without rebuilding it. Lets us reuse one model object for
# all three ablation configs instead of re-running convert_to_quant/QAT.
def set_quant_components(model, quant_weight, quant_act):
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            mod.quant_weight = quant_weight
            mod.quant_act = quant_act

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

CIFAR_MEAN_T = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
CIFAR_STD_T = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
CLIP_MIN = ((0.0 - CIFAR_MEAN_T) / CIFAR_STD_T)
CLIP_MAX = ((1.0 - CIFAR_MEAN_T) / CIFAR_STD_T)

def run_fgsm_pgd(model, loader, eps=8/255, seeds=SEEDS):
    """
    NOTE (item 1): PGD is stochastic (random_start=True), so it's now
    averaged over `seeds` independent runs. FGSM has no randomness, so it's
    still a single deterministic pass. Returns PGD_mean/PGD_std alongside the
    original PGD key (set to the mean, for backward-compat with plotting code
    that expects a scalar "PGD" column).
    """
    model.eval()
    fgsm = torchattacks.FGSM(model, eps=eps)
    out = {}

    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = fgsm(x, y)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    out["FGSM"] = correct / total

    pgd_accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        pgd = torchattacks.PGD(model, eps=eps, alpha=2/255, steps=20, random_start=True)
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_adv = pgd(x, y)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        pgd_accs.append(correct / total)
    out["PGD"] = float(np.mean(pgd_accs))
    out["PGD_mean"] = float(np.mean(pgd_accs))
    out["PGD_std"] = float(np.std(pgd_accs))
    return out

def run_autoattack(model, loader, eps=8/255):
    model.eval()
    adversary = AutoAttack(model, norm="Linf", eps=eps, version="custom", device=device, verbose=False)
    adversary.attacks_to_run = ["apgd-ce", "apgd-t"]
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

def _run_bpda_once(model, loader, eps, n_restarts):
    correct_masks = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        worst_correct = torch.ones(y.size(0), dtype=torch.bool, device=device)
        for _ in range(n_restarts):
            x_adv = bpda_pgd_attack(model, x, y, eps=eps)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
            worst_correct &= (pred == y)
        correct_masks.append(worst_correct)
    all_correct = torch.cat(correct_masks)
    return all_correct.float().mean().item()

def run_bpda(model, loader, eps=8/255, n_restarts=1, seeds=SEEDS):
    """
    NOTE (item 1): now runs the whole worst-case-over-n_restarts procedure
    once per seed and reports mean/std across seeds, in addition to the
    original scalar (mean of seeds) for backward-compat.
    """
    accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        accs.append(_run_bpda_once(model, loader, eps, n_restarts))
    return {
        "BPDA_PGD": float(np.mean(accs)),
        "BPDA_PGD_mean": float(np.mean(accs)),
        "BPDA_PGD_std": float(np.std(accs)),
    }

def gradient_diagnostics(model, loader, fp32_ref=None, max_batches=5):
    set_ste_mode(model, False)
    frac_zero_hard, norm_hard = [], []
    frac_zero_ste, norm_ste = [], []
    cos_sims = []

    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        set_ste_mode(model, False)
        x_in = x.clone().requires_grad_(True)
        loss = F.cross_entropy(model(x_in), y)
        g_hard = torch.autograd.grad(loss, x_in)[0].flatten()
        frac_zero_hard.append((g_hard.abs() < 1e-8).float().mean().item())
        norm_hard.append(g_hard.norm().item())

        set_ste_mode(model, True)
        x_in2 = x.clone().requires_grad_(True)
        loss2 = F.cross_entropy(model(x_in2), y)
        g_ste = torch.autograd.grad(loss2, x_in2)[0].flatten()
        set_ste_mode(model, False)
        frac_zero_ste.append((g_ste.abs() < 1e-8).float().mean().item())
        norm_ste.append(g_ste.norm().item())

        if fp32_ref is not None:
            fp32_ref.eval()
            x_ref = x.clone().requires_grad_(True)
            loss_ref = F.cross_entropy(fp32_ref(x_ref), y)
            g_ref = torch.autograd.grad(loss_ref, x_ref)[0].flatten()
            cos_sims.append(F.cosine_similarity(g_ste.unsqueeze(0), g_ref.unsqueeze(0)).item())

    diagnostics = {
        "frac_zero_grad_hard": float(np.mean(frac_zero_hard)),
        "frac_zero_grad_ste": float(np.mean(frac_zero_ste)),
        "grad_norm_hard": float(np.mean(norm_hard)),
        "grad_norm_ste": float(np.mean(norm_ste)),
    }
    if cos_sims:
        diagnostics["grad_cosine_sim_with_FP32"] = float(np.mean(cos_sims))
    return diagnostics

def random_noise_attack(model, loader, eps=8/255, n_restarts=1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            worst_correct = torch.ones(y.size(0), dtype=torch.bool, device=device)
            for _ in range(n_restarts):
                noise = torch.empty_like(x).uniform_(-eps, eps)
                x_adv = torch.max(torch.min(x + noise, clip_max), clip_min)
                pred = model(x_adv).argmax(dim=1)
                worst_correct &= (pred == y)
            correct += worst_correct.sum().item()
            total += y.size(0)
    return correct / total

def run_random_noise_seeded(model, loader, eps=8/255, seeds=SEEDS):
    """NOTE (item 1): seed-averaged wrapper around random_noise_attack."""
    accs = [random_noise_attack(model, loader, eps=eps, seed=s) for s in seeds]
    return {
        "Random_Noise": float(np.mean(accs)),
        "Random_Noise_mean": float(np.mean(accs)),
        "Random_Noise_std": float(np.std(accs)),
    }


def pgd_steps_ablation(model, loader, eps=8/255, step_list=(0, 1, 2, 5, 10, 20, 50)):
    model.eval()
    out = {}
    for steps in step_list:
        if steps == 0:
            acc = random_noise_attack(model, loader, eps=eps, seed=0)
        else:
            pgd = torchattacks.PGD(model, eps=eps, alpha=2/255, steps=steps, random_start=True)
            correct, total = 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                x_adv = pgd(x, y)
                with torch.no_grad():
                    pred = model(x_adv).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            acc = correct / total
        out[steps] = acc
    return out


def pgd_trajectory_diagnostics(model, loader, eps=8/255, alpha=2/255, steps=20, max_batches=5):
    model.eval()
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)
    step_grad_norms = [0.0] * steps
    step_movement = [0.0] * steps
    n_batches = 0
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        noise = torch.empty_like(x).uniform_(-eps, eps)
        x_start = torch.max(torch.min(x + noise, clip_max), clip_min).detach()
        x_adv = x_start.clone()
        for s in range(steps):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            step_grad_norms[s] += grad.flatten(1).norm(dim=1).mean().item()
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
            x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
            step_movement[s] += (x_adv - x_start).flatten(1).abs().max(dim=1).values.mean().item()
        n_batches += 1
    return {
        "grad_norm_per_step": [g / n_batches for g in step_grad_norms],
        "movement_from_random_start_per_step": [m / n_batches for m in step_movement],
    }


def layerwise_grad_profile(model, loader, use_ste, max_batches=3):
    quant_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, (QuantConv2d, QuantLinear))]
    norms = {n: [] for n, _ in quant_layers}
    handles = []

    def make_hook(name):
        def hook(module, grad_input, grad_output):
            gi = grad_input[0]
            if gi is not None:
                norms[name].append(gi.flatten(1).norm(dim=1).mean().item())
        return hook

    for n, m in quant_layers:
        handles.append(m.register_full_backward_hook(make_hook(n)))

    set_ste_mode(model, use_ste)
    model.eval()
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        x = x.clone().requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        model.zero_grad(set_to_none=True)
        loss.backward()

    for h in handles:
        h.remove()
    set_ste_mode(model, False)

    ordered_names = [n for n, _ in quant_layers]
    return {n: (float(np.mean(norms[n])) if len(norms[n]) else None) for n in ordered_names}


def staircase_diagnostic(model, loader, radius=1/255, n_points=40):
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    direction = torch.randn_like(x)
    flat_norm = direction.flatten(1).norm(dim=1).view(-1, *([1] * (x.dim() - 1)))
    direction = direction / flat_norm
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)

    with torch.no_grad():
        prev_logits = model(x)
        plateau_hits = 0.0
        for i in range(1, n_points + 1):
            step = x + direction * (radius * i / n_points)
            step = torch.max(torch.min(step, clip_max), clip_min)
            logits = model(step)
            plateau_hits += (logits == prev_logits).all(dim=1).float().mean().item()
            prev_logits = logits
    return {"plateau_fraction": plateau_hits / n_points}


# NEW (item 2): weight-only vs activation-only vs both quantization ablation.
# Cheap by design -- reuses the already-built quantized model, just flips
# quant_weight/quant_act flags in place, and only computes clean_acc + a
# single-seed 20-step PGD + frac_zero_grad_hard per config (not the full
# AutoAttack/BPDA/trajectory suite). Restores the model to (True, True)
# (its original state) before returning.
def run_quant_component_ablation(model, loader, name, eps=8/255):
    configs = [
        ("weight_only", True, False),
        ("act_only", False, True),
        ("both", True, True),
    ]
    rows = []
    for label, qw, qa in configs:
        set_quant_components(model, qw, qa)
        clean_acc = sanity_check_accuracy(model, loader)

        torch.manual_seed(0)
        pgd = torchattacks.PGD(model, eps=eps, alpha=2/255, steps=20, random_start=True)
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_adv = pgd(x, y)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        pgd_acc = correct / total

        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        x_in = x.clone().requires_grad_(True)
        loss = F.cross_entropy(model(x_in), y)
        g_hard = torch.autograd.grad(loss, x_in)[0].flatten()
        frac_zero = (g_hard.abs() < 1e-8).float().mean().item()

        rows.append({
            "model": name, "config": label,
            "quant_weight": qw, "quant_act": qa,
            "clean_acc": clean_acc, "PGD_acc": pgd_acc,
            "frac_zero_grad_hard": frac_zero,
        })

    # restore original (both quantized) state
    set_quant_components(model, True, True)
    return rows


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

    try:
        results.update(run_random_noise_seeded(model, loader, eps=eps))
    except Exception as e:
        print(f"  [WARN] random_noise_attack failed for {name}: {e}")
        results["Random_Noise"] = None

    if count_quant_layers(model) > 0:
        try:
            results.update(run_bpda(model, loader, eps=eps, n_restarts=5))
        except Exception as e:
            print(f"  [WARN] BPDA failed for {name}: {e}")
            results["BPDA_PGD"] = None

        try:
            results.update(gradient_diagnostics(model, loader, fp32_ref=fp32_ref, max_batches=5))
        except Exception as e:
            print(f"  [WARN] gradient_diagnostics failed for {name}: {e}")

        try:
            results.update(staircase_diagnostic(model, loader))
        except Exception as e:
            print(f"  [WARN] staircase_diagnostic failed for {name}: {e}")

        try:
            ablation = pgd_steps_ablation(model, loader, eps=eps)
            pd.DataFrame([{"model": name, "steps": k, "acc": v} for k, v in ablation.items()]) \
              .to_csv(ablation_csv_path(name), index=False)
        except Exception as e:
            print(f"  [WARN] pgd_steps_ablation failed for {name}: {e}")

        try:
            traj = pgd_trajectory_diagnostics(model, loader, eps=eps, max_batches=5)
            with open(trajectory_json_path(name), "w") as f:
                json.dump(traj, f, indent=2)
        except Exception as e:
            print(f"  [WARN] pgd_trajectory_diagnostics failed for {name}: {e}")

        try:
            prof_hard = layerwise_grad_profile(model, loader, use_ste=False)
            prof_ste = layerwise_grad_profile(model, loader, use_ste=True)
            rows = [{"model": name, "layer": n, "grad_norm_hard": prof_hard.get(n),
                     "grad_norm_ste": prof_ste.get(n)} for n in prof_hard]
            pd.DataFrame(rows).to_csv(layerwise_csv_path(name), index=False)
        except Exception as e:
            print(f"  [WARN] layerwise_grad_profile failed for {name}: {e}")

        # NEW (item 2): weight-only / activation-only / both ablation
        try:
            rows = run_quant_component_ablation(model, loader, name, eps=eps)
            pd.DataFrame(rows).to_csv(component_ablation_csv_path(name), index=False)
        except Exception as e:
            print(f"  [WARN] run_quant_component_ablation failed for {name}: {e}")

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

        try:
            row["Random_Noise_acc"] = random_noise_attack(model, loader, eps=eps)
        except Exception as e:
            print(f"  [WARN] random_noise sweep failed for {name} eps={eps:.4f}: {e}")
            row["Random_Noise_acc"] = None

        if is_quant:
            try:
                row["BPDA_acc"] = _run_bpda_once(model, loader, eps=eps, n_restarts=3)
            except Exception as e:
                print(f"  [WARN] BPDA sweep failed for {name} eps={eps:.4f}: {e}")
                row["BPDA_acc"] = None
        rows.append(row)
    return rows

def parallelize(model):
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        return nn.DataParallel(model)
    return model

def parse_model_name(model_name):
    suffixes = (
        "_FP32",
        "_int8_PTQ",
        "_int8_QAT",
    )

    for suffix in suffixes:
        if model_name.endswith(suffix):
            arch = model_name[:-len(suffix)]
            mode = suffix[1:]
            return arch, mode

    raise ValueError(f"Unknown model name: {model_name}")

def build_model(arch, mode, finetune_loader):
    fp32 = load_pretrained(arch)

    if mode == "FP32":
        return fp32, None

    if mode == "int8_PTQ":
        return (
            convert_to_quant(fp32, bits=8, quant_weight=True, quant_act=True),
            fp32,
        )

    if mode == "int8_QAT":
        return (
            prepare_qat(fp32, bits=8, finetune_loader=finetune_loader, epochs=3),
            fp32,
        )

    raise ValueError(f"Unknown mode: {mode}")

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python QuantAdvCC.py <model_name>")

    model_name = sys.argv[1]
    arch, mode = parse_model_name(model_name)

    print(f"Building {model_name}...")

    results_csv = results_csv_path(model_name)
    sweep_csv = sweep_csv_path(model_name)

    if os.path.exists(results_csv):
        print(f"{model_name} already has results at {results_csv}; skipping.")
        return

    finetune_loader, eval_loader = get_dataloaders()

    model, ref = build_model(arch, mode, finetune_loader)

    model = parallelize(model)
    if ref is not None:
        ref = parallelize(ref)

    print(f"\nEvaluating {model_name} ...")

    try:
        res = run_suite(model, eval_loader, model_name, fp32_ref=ref)
    except Exception as e:
        print(f"[FAIL] run_suite failed: {e}")
        traceback.print_exc()
        res = {"model": model_name}

    new_row = pd.DataFrame([res])
    new_row.to_csv(results_csv, index=False)

    print("\nResult:")
    print(new_row.to_string(index=False))
    print(f"Wrote {results_csv}")

    SWEEP_EPSILONS = [1 / 255, 2 / 255, 4 / 255, 16 / 255]  # 8/255 already covered by run_suite

    if os.path.exists(sweep_csv):
        df_sweep = pd.read_csv(sweep_csv)
        sweep_done = set(zip(df_sweep["model"].astype(str), df_sweep["epsilon"].round(6)))
    else:
        df_sweep = pd.DataFrame()
        sweep_done = set()

    pending_eps = [eps for eps in SWEEP_EPSILONS if (model_name, round(eps, 6)) not in sweep_done]

    if pending_eps:
        print(f"\nRunning epsilon sweep for {model_name}")
        try:
            rows = run_epsilon_sweep_for_model(model, eval_loader, model_name, pending_eps)
            if rows:
                new_sweep = pd.DataFrame(rows)
                df_sweep = pd.concat([df_sweep, new_sweep], ignore_index=True)
                df_sweep.to_csv(sweep_csv, index=False)
        except Exception as e:
            print(f"[FAIL] epsilon sweep failed: {e}")
            traceback.print_exc()
    else:
        print(f"{model_name} epsilon sweep already complete.")

    print("\nAll done.")

if __name__ == '__main__':
    main()
