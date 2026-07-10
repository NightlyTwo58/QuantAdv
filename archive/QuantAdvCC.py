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
import time
import torchattacks
from autoattack import AutoAttack
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CIFAR100_ROOT = os.environ.get("CIFAR100_ROOT", os.environ.get("CIFAR10_ROOT", "./"))
SEEDS = [0, 1, 2]
"""
Threaded src-model, src-quantization, src-attack analysis with eplison sweep.
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
def get_dataloaders(batch_size=128, eval_n=3000, finetune_n=5000, eval_batch_size=500):
    print(f"[data] Building dataloaders (eval_n={eval_n}, finetune_n={finetune_n}, batch_size={batch_size}, eval_batch_size={eval_batch_size})...")
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
    print(f"[data] Dataloaders ready: {len(finetune_subset)} finetune samples, {len(eval_subset)} eval samples, {workers} workers.")
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
    print(f"[model] Loading pretrained weights for {arch_key} ({hub_name}) from torch.hub...")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", hub_name, pretrained=True)
    print(f"[model] {arch_key} loaded and moved to {device}.")
    return model.to(device).eval()
def sanity_check_accuracy(model, loader):
    print("[eval] Running clean accuracy sanity check...")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            print(f"  [eval] clean_acc batch {bi+1}/{len(loader)} running_acc={correct/total:.4f}")
    acc = correct / total
    print(f"[eval] Clean accuracy: {acc:.4f}")
    return acc
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
    print(f"[quant] Converting model to quantized form (bits={bits}, quant_weight={quant_weight}, quant_act={quant_act})...")
    m = copy.deepcopy(model)
    _replace_recursive(m, bits, quant_weight, quant_act)
    n_layers = count_quant_layers(m)
    print(f"[quant] Conversion complete: {n_layers} quantized layers.")
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
    print(f"[qat] Preparing QAT model (bits={bits}, epochs={epochs}, lr={lr})...")
    m = convert_to_quant(fp32_model, bits, quant_weight=True, quant_act=True)
    if torch.cuda.device_count() > 1:
        print(f"[qat] Wrapping model in DataParallel across {torch.cuda.device_count()} GPUs.")
        m = nn.DataParallel(m)
    set_ste_mode(m, True)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    n_batches = len(finetune_loader)
    print(f"[qat] Starting QAT fine-tuning: {epochs} epochs x {n_batches} batches.")
    for epoch in range(epochs):
        running = 0.0
        epoch_start = time.time()
        for bi, (x, y) in enumerate(finetune_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            if (bi + 1) % 10 == 0 or (bi + 1) == n_batches:
                print(f"  [qat] epoch {epoch+1}/{epochs} batch {bi+1}/{n_batches} loss={loss.item():.4f} avg_loss={running/(bi+1):.4f}")
        elapsed = time.time() - epoch_start
        print(f"  [qat] QAT epoch {epoch+1}/{epochs} avg loss {running/len(finetune_loader):.4f} ({elapsed:.1f}s)")
    set_ste_mode(m, False)
    print("[qat] QAT fine-tuning complete.")
    return m.eval()
def prepare_qat_adv(fp32_model, bits, finetune_loader, epochs=3, lr=1e-3, eps=8/255, alpha=2/255, steps=10):
    print(f"[qat_adv] Preparing Adversarial QAT model (bits={bits}, epochs={epochs}, eps={eps:.4f})...")
    m = convert_to_quant(fp32_model, bits, quant_weight=True, quant_act=True)
    
    raw_model = m
    if torch.cuda.device_count() > 1:
        print(f"[qat_adv] Wrapping weight-update model in DataParallel ({torch.cuda.device_count()} GPUs).")
        m = nn.DataParallel(m)
        raw_model = m.module
        
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)
    n_batches = len(finetune_loader)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for bi, (x, y) in enumerate(finetune_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            set_ste_mode(raw_model, True)
            raw_model.eval() 
            
            x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
            x_adv = torch.clamp(x_adv, clip_min, clip_max).detach()
            
            for _ in range(steps):
                x_adv.requires_grad_(True)
                with torch.set_grad_enabled(True):
                    loss_inner = F.cross_entropy(raw_model(x_adv), y)
                grad = torch.autograd.grad(loss_inner, x_adv, retain_graph=False, create_graph=False)[0]
                x_adv = x_adv.detach() + alpha * grad.sign()
                x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
                x_adv = torch.clamp(x_adv, clip_min, clip_max).detach()
            
            m.train() 
            set_ste_mode(raw_model, True)
            
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(x_adv), y)
            loss.backward()
            opt.step()
            
            running_loss += loss.item()
            if (bi + 1) % 10 == 0 or (bi + 1) == n_batches:
                print(f"  [qat_adv] epoch {epoch+1}/{epochs} batch {bi+1}/{n_batches} loss={loss.item():.4f} avg_loss={running_loss/(bi+1):.4f}")
                
    set_ste_mode(raw_model, False)
    return m.eval()
CIFAR_MEAN_T = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
CIFAR_STD_T = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
CLIP_MIN = ((0.0 - CIFAR_MEAN_T) / CIFAR_STD_T)
CLIP_MAX = ((1.0 - CIFAR_MEAN_T) / CIFAR_STD_T)
def run_fgsm_pgd(model, loader, eps=8/255, seeds=SEEDS):
    print(f"[attack] Running FGSM + PGD (eps={eps:.4f}, seeds={seeds})...")
    model.eval()
    fgsm = torchattacks.FGSM(model, eps=eps)
    out = {}
    correct, total = 0, 0
    n_batches = len(loader)
    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = fgsm(x, y)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        print(f"  [attack][FGSM] batch {bi+1}/{n_batches} running_acc={correct/total:.4f}")
    out["FGSM"] = correct / total
    print(f"[attack] FGSM accuracy: {out['FGSM']:.4f}")
    pgd_accs = []
    for seed in seeds:
        print(f"[attack] Running PGD with seed={seed}...")
        torch.manual_seed(seed)
        pgd = torchattacks.PGD(model, eps=eps, alpha=2/255, steps=20, random_start=True)
        correct, total = 0, 0
        for bi, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            x_adv = pgd(x, y)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            print(f"  [attack][PGD seed={seed}] batch {bi+1}/{n_batches} running_acc={correct/total:.4f}")
        seed_acc = correct / total
        print(f"[attack] PGD seed={seed} accuracy: {seed_acc:.4f}")
        pgd_accs.append(seed_acc)
    out["PGD"] = float(np.mean(pgd_accs))
    out["PGD_mean"] = float(np.mean(pgd_accs))
    out["PGD_std"] = float(np.std(pgd_accs))
    print(f"[attack] PGD mean accuracy: {out['PGD_mean']:.4f} (std={out['PGD_std']:.4f})")
    return out
def run_autoattack(model, loader, eps=8/255):
    print(f"[attack] Running AutoAttack (eps={eps:.4f}, attacks=apgd-ce,apgd-t)...")
    model.eval()
    adversary = AutoAttack(model, norm="Linf", eps=eps, version="custom", device=device, verbose=False)
    adversary.attacks_to_run = ["apgd-ce", "apgd-t"]
    correct, total = 0, 0
    n_batches = len(loader)
    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        print(f"  [attack][AutoAttack] batch {bi+1}/{n_batches} running_acc={correct/total:.4f}")
    acc = correct / total
    print(f"[attack] AutoAttack accuracy: {acc:.4f}")
    return acc
def transfer_attack(source_model, target_model, loader, eps=8/255):
    print(f"[attack] Running transfer attack (PGD crafted on source model, eps={eps:.4f})...")
    pgd = torchattacks.PGD(source_model, eps=eps, alpha=2/255, steps=20, random_start=True)
    correct, total = 0, 0
    n_batches = len(loader)
    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = pgd(x, y)
        with torch.no_grad():
            pred = target_model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        print(f"  [attack][Transfer] batch {bi+1}/{n_batches} running_acc={correct/total:.4f}")
    acc = correct / total
    print(f"[attack] Transfer attack accuracy: {acc:.4f}")
    return acc
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
    n_batches = len(loader)
    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        worst_correct = torch.ones(y.size(0), dtype=torch.bool, device=device)
        for r in range(n_restarts):
            x_adv = bpda_pgd_attack(model, x, y, eps=eps)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
            worst_correct &= (pred == y)
        print(f"  [attack][BPDA] batch {bi+1}/{n_batches} ({n_restarts} restarts) running_correct_frac={worst_correct.float().mean().item():.4f}")
        correct_masks.append(worst_correct)
    all_correct = torch.cat(correct_masks)
    return all_correct.float().mean().item()
def run_bpda(model, loader, eps=8/255, n_restarts=1, seeds=SEEDS):
    """
    NOTE (item 1): now runs the whole worst-case-over-n_restarts procedure
    once per seed and reports mean/std across seeds, in addition to the
    original scalar (mean of seeds) for backward-compat.
    """
    print(f"[attack] Running BPDA-PGD (eps={eps:.4f}, n_restarts={n_restarts}, seeds={seeds})...")
    accs = []
    for seed in seeds:
        print(f"[attack] BPDA seed={seed}...")
        torch.manual_seed(seed)
        acc = _run_bpda_once(model, loader, eps, n_restarts)
        print(f"[attack] BPDA seed={seed} accuracy: {acc:.4f}")
        accs.append(acc)
    result = {
        "BPDA_PGD": float(np.mean(accs)),
        "BPDA_PGD_mean": float(np.mean(accs)),
        "BPDA_PGD_std": float(np.std(accs)),
    }
    print(f"[attack] BPDA-PGD mean accuracy: {result['BPDA_PGD_mean']:.4f} (std={result['BPDA_PGD_std']:.4f})")
    return result
def gradient_diagnostics(model, loader, fp32_ref=None, max_batches=5):
    print(f"[diag] Running gradient diagnostics (max_batches={max_batches}, fp32_ref={'yes' if fp32_ref is not None else 'no'})...")
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
        print(f"  [diag] gradient_diagnostics batch {bi+1}/{max_batches} frac_zero_hard={frac_zero_hard[-1]:.4f} frac_zero_ste={frac_zero_ste[-1]:.4f}")
    diagnostics = {
        "frac_zero_grad_hard": float(np.mean(frac_zero_hard)),
        "frac_zero_grad_ste": float(np.mean(frac_zero_ste)),
        "grad_norm_hard": float(np.mean(norm_hard)),
        "grad_norm_ste": float(np.mean(norm_ste)),
    }
    if cos_sims:
        diagnostics["grad_cosine_sim_with_FP32"] = float(np.mean(cos_sims))
    print(f"[diag] Gradient diagnostics complete: {diagnostics}")
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
    print(f"[attack] Running Random Noise attack (eps={eps:.4f}, seeds={seeds})...")
    accs = []
    for s in seeds:
        acc = random_noise_attack(model, loader, eps=eps, seed=s)
        print(f"[attack] Random Noise seed={s} accuracy: {acc:.4f}")
        accs.append(acc)
    result = {
        "Random_Noise": float(np.mean(accs)),
        "Random_Noise_mean": float(np.mean(accs)),
        "Random_Noise_std": float(np.std(accs)),
    }
    print(f"[attack] Random Noise mean accuracy: {result['Random_Noise_mean']:.4f} (std={result['Random_Noise_std']:.4f})")
    return result
def pgd_steps_ablation(model, loader, eps=8/255, step_list=(0, 1, 2, 5, 10, 20, 50)):
    print(f"[ablation] Running PGD steps ablation (eps={eps:.4f}, steps={list(step_list)})...")
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
        print(f"  [ablation] PGD steps={steps} accuracy={acc:.4f}")
        out[steps] = acc
    print("[ablation] PGD steps ablation complete.")
    return out
def pgd_trajectory_diagnostics(model, loader, eps=8/255, alpha=2/255, steps=20, max_batches=5):
    print(f"[diag] Running PGD trajectory diagnostics (eps={eps:.4f}, steps={steps}, max_batches={max_batches})...")
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
        print(f"  [diag] trajectory batch {bi+1}/{max_batches} processed.")
    print("[diag] PGD trajectory diagnostics complete.")
    return {
        "grad_norm_per_step": [g / n_batches for g in step_grad_norms],
        "movement_from_random_start_per_step": [m / n_batches for m in step_movement],
    }
def layerwise_grad_profile(model, loader, use_ste, max_batches=3):
    print(f"[diag] Running layerwise gradient profile (use_ste={use_ste}, max_batches={max_batches})...")
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
        print(f"  [diag] layerwise profile batch {bi+1}/{max_batches} processed.")
    for h in handles:
        h.remove()
    set_ste_mode(model, False)
    ordered_names = [n for n, _ in quant_layers]
    print(f"[diag] Layerwise gradient profile complete for {len(ordered_names)} layers.")
    return {n: (float(np.mean(norms[n])) if len(norms[n]) else None) for n in ordered_names}
def staircase_diagnostic(model, loader, radius=1/255, n_points=40):
    print(f"[diag] Running staircase diagnostic (radius={radius:.4f}, n_points={n_points})...")
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
    result = {"plateau_fraction": plateau_hits / n_points}
    print(f"[diag] Staircase diagnostic complete: plateau_fraction={result['plateau_fraction']:.4f}")
    return result
# NEW (item 2): weight-only vs activation-only vs both quantization ablation.
# Cheap by design -- reuses the already-built quantized model, just flips
# quant_weight/quant_act flags in place, and only computes clean_acc + a
# src-seed 20-step PGD + frac_zero_grad_hard per config (not the full
# AutoAttack/BPDA/trajectory suite). Restores the model to (True, True)
# (its original state) before returning.
def run_quant_component_ablation(model, loader, name, eps=8/255):
    print(f"[ablation] Running quant component ablation for {name} (eps={eps:.4f})...")
    configs = [
        ("weight_only", True, False),
        ("act_only", False, True),
        ("both", True, True),
    ]
    rows = []
    for label, qw, qa in configs:
        print(f"[ablation] Component config: {label} (quant_weight={qw}, quant_act={qa})")
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
        print(f"[ablation] {label}: clean_acc={clean_acc:.4f} PGD_acc={pgd_acc:.4f} frac_zero_grad_hard={frac_zero:.4f}")
        rows.append({
            "model": name, "config": label,
            "quant_weight": qw, "quant_act": qa,
            "clean_acc": clean_acc, "PGD_acc": pgd_acc,
            "frac_zero_grad_hard": frac_zero,
        })
    # restore original (both quantized) state
    set_quant_components(model, True, True)
    print(f"[ablation] Quant component ablation for {name} complete.")
    return rows
def run_suite(model, loader, name, fp32_ref=None, eps=8/255):
    print(f"\n===== [suite] Starting evaluation suite for {name} (eps={eps:.4f}) =====")
    suite_start = time.time()
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
        print(f"[suite] {name} is quantized ({count_quant_layers(model)} quant layers) -- running quant-specific diagnostics.")
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
            print(f"[suite] Running PGD steps ablation for {name}...")
            ablation = pgd_steps_ablation(model, loader, eps=eps)
            pd.DataFrame([{"model": name, "steps": k, "acc": v} for k, v in ablation.items()]) \
              .to_csv(ablation_csv_path(name), index=False)
            print(f"[suite] Wrote {ablation_csv_path(name)}")
        except Exception as e:
            print(f"  [WARN] pgd_steps_ablation failed for {name}: {e}")
        try:
            print(f"[suite] Running PGD trajectory diagnostics for {name}...")
            traj = pgd_trajectory_diagnostics(model, loader, eps=eps, max_batches=5)
            with open(trajectory_json_path(name), "w") as f:
                json.dump(traj, f, indent=2)
            print(f"[suite] Wrote {trajectory_json_path(name)}")
        except Exception as e:
            print(f"  [WARN] pgd_trajectory_diagnostics failed for {name}: {e}")
        try:
            print(f"[suite] Running layerwise gradient profile for {name}...")
            prof_hard = layerwise_grad_profile(model, loader, use_ste=False)
            prof_ste = layerwise_grad_profile(model, loader, use_ste=True)
            rows = [{"model": name, "layer": n, "grad_norm_hard": prof_hard.get(n),
                     "grad_norm_ste": prof_ste.get(n)} for n in prof_hard]
            pd.DataFrame(rows).to_csv(layerwise_csv_path(name), index=False)
            print(f"[suite] Wrote {layerwise_csv_path(name)}")
        except Exception as e:
            print(f"  [WARN] layerwise_grad_profile failed for {name}: {e}")
        # NEW (item 2): weight-only / activation-only / both ablation
        try:
            rows = run_quant_component_ablation(model, loader, name, eps=eps)
            pd.DataFrame(rows).to_csv(component_ablation_csv_path(name), index=False)
            print(f"[suite] Wrote {component_ablation_csv_path(name)}")
        except Exception as e:
            print(f"  [WARN] run_quant_component_ablation failed for {name}: {e}")
    else:
        print(f"[suite] {name} is FP32 (no quant layers) -- skipping quant-specific diagnostics.")
    elapsed = time.time() - suite_start
    print(f"===== [suite] Finished evaluation suite for {name} in {elapsed:.1f}s =====\n")
    return results
def run_epsilon_sweep_for_model(model, loader, name, epsilons):
    print(f"[sweep] Starting epsilon sweep for {name}: epsilons={[f'{e:.4f}' for e in epsilons]}")
    rows = []
    is_quant = count_quant_layers(model) > 0
    for ei, eps in enumerate(epsilons):
        print(f"[sweep] ({ei+1}/{len(epsilons)}) {name} eps={eps:.4f}")
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
            print(f"  [sweep] eps={eps:.4f} PGD_acc={row['PGD_acc']:.4f}")
        except Exception as e:
            print(f"  [WARN] PGD sweep failed for {name} eps={eps:.4f}: {e}")
            row["PGD_acc"] = None
        try:
            row["Random_Noise_acc"] = random_noise_attack(model, loader, eps=eps)
            print(f"  [sweep] eps={eps:.4f} Random_Noise_acc={row['Random_Noise_acc']:.4f}")
        except Exception as e:
            print(f"  [WARN] random_noise sweep failed for {name} eps={eps:.4f}: {e}")
            row["Random_Noise_acc"] = None
        if is_quant:
            try:
                row["BPDA_acc"] = _run_bpda_once(model, loader, eps=eps, n_restarts=3)
                print(f"  [sweep] eps={eps:.4f} BPDA_acc={row['BPDA_acc']:.4f}")
            except Exception as e:
                print(f"  [WARN] BPDA sweep failed for {name} eps={eps:.4f}: {e}")
                row["BPDA_acc"] = None
        rows.append(row)
    print(f"[sweep] Epsilon sweep for {name} complete.")
    return rows
def parallelize(model):
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        print(f"[model] Parallelizing model across {torch.cuda.device_count()} GPUs.")
        return nn.DataParallel(model)
    return model
def parse_model_name(model_name):
    suffixes = (
        "_FP32",
        "_int8_PTQ",
        "_int8_QAT",
        "_int8_QAT_AT",
    )
    for suffix in suffixes:
        if model_name.endswith(suffix):
            arch = model_name[:-len(suffix)]
            mode = suffix[1:]
            return arch, mode
    raise ValueError(f"Unknown model name: {model_name}")
def build_model(arch, mode, finetune_loader):
    print(f"[build] Building model: arch={arch}, mode={mode}")
    fp32 = load_pretrained(arch)
    if mode == "FP32":
        print(f"[build] Using FP32 model as-is.")
        return fp32, None
    if mode == "int8_PTQ":
        print(f"[build] Applying post-training int8 quantization (PTQ)...")
        m = convert_to_quant(fp32, bits=8, quant_weight=True, quant_act=True)
        print(f"[build] PTQ model built.")
        return m, fp32
    if mode == "int8_QAT":
        print(f"[build] Applying quantization-aware training (QAT)...")
        m = prepare_qat(fp32, bits=8, finetune_loader=finetune_loader, epochs=3)
        print(f"[build] QAT model built.")
        return m, fp32
    if mode == "int8_QAT_AT":
        print(f"[build] Applying adversarial quantization-aware training (QAT_AT)...")
        m = prepare_qat_adv(fp32, bits=8, finetune_loader=finetune_loader, epochs=3)
        print(f"[build] QAT_AT model built.")
        return m, fp32
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
    print(f"[main] Preparing dataloaders...")
    finetune_loader, eval_loader = get_dataloaders()
    print(f"[main] Building model {model_name} (arch={arch}, mode={mode})...")
    model, ref = build_model(arch, mode, finetune_loader)
    model = parallelize(model)
    if ref is not None:
        ref = parallelize(ref)
    print(f"\nEvaluating {model_name} ...")
    run_start = time.time()
    try:
        res = run_suite(model, eval_loader, model_name, fp32_ref=ref)
    except Exception as e:
        print(f"[FAIL] run_suite failed: {e}")
        traceback.print_exc()
        res = {"model": model_name}
    print(f"[main] run_suite finished in {time.time() - run_start:.1f}s")
    new_row = pd.DataFrame([res])
    new_row.to_csv(results_csv, index=False)
    print("\nResult:")
    print(new_row.to_string(index=False))
    print(f"Wrote {results_csv}")
    SWEEP_EPSILONS = [1 / 255, 2 / 255, 4 / 255, 16 / 255]  # 8/255 already covered by run_suite
    if os.path.exists(sweep_csv):
        print(f"[main] Found existing sweep file {sweep_csv}, loading to check completed epsilons...")
        df_sweep = pd.read_csv(sweep_csv)
        sweep_done = set(zip(df_sweep["model"].astype(str), df_sweep["epsilon"].round(6)))
    else:
        df_sweep = pd.DataFrame()
        sweep_done = set()
    pending_eps = [eps for eps in SWEEP_EPSILONS if (model_name, round(eps, 6)) not in sweep_done]
    if pending_eps:
        print(f"\n[main] Running epsilon sweep for {model_name}: pending={[f'{e:.4f}' for e in pending_eps]}")
        sweep_start = time.time()
        try:
            rows = run_epsilon_sweep_for_model(model, eval_loader, model_name, pending_eps)
            if rows:
                new_sweep = pd.DataFrame(rows)
                df_sweep = pd.concat([df_sweep, new_sweep], ignore_index=True)
                df_sweep.to_csv(sweep_csv, index=False)
                print(f"[main] Wrote {sweep_csv}")
        except Exception as e:
            print(f"[FAIL] epsilon sweep failed: {e}")
            traceback.print_exc()
        print(f"[main] Epsilon sweep finished in {time.time() - sweep_start:.1f}s")
    else:
        print(f"{model_name} epsilon sweep already complete.")
    print("\nAll done.")
if __name__ == '__main__':
    main()