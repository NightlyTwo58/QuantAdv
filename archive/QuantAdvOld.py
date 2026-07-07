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
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import autocast, GradScaler

import torchattacks
from autoattack import AutoAttack

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_CSV = os.path.join(DATA_DIR, "accuracyresult.csv")
SWEEP_CSV = os.path.join(DATA_DIR, "sweepresult.csv")
PLOT_PNG = os.path.join(DATA_DIR, "accuracyplot.png")

SEEDS = [0, 1, 2]

"""
Modernized archival non-threaded analysis of metrics of attacks per model and eplison
"""


def ablation_csv_path(model_name):
    return os.path.join(DATA_DIR, f"ablation_{model_name}.csv")


def layerwise_csv_path(model_name):
    return os.path.join(DATA_DIR, f"layerwise_{model_name}.csv")


def trajectory_json_path(model_name):
    return os.path.join(DATA_DIR, f"trajectory_{model_name}.json")


# weight-only vs activation-only vs both quantization ablation
def component_ablation_csv_path(model_name):
    return os.path.join(DATA_DIR, f"component_ablation_{model_name}.csv")


missing = [pkg for pkg in ("torchattacks", "autoattack") if importlib.util.find_spec(pkg) is None]
if missing:
    raise ImportError(f"Missing packages: {missing}.\nInstall via: pip install -r requirements.txt")
print("All required packages are available.")

expected = os.path.join(PROJECT_ROOT, "cifar-10-batches-py")
if not os.path.isdir(expected):
    raise FileNotFoundError(f"Expected extracted CIFAR-10 at {expected!r}")


CIFAR_MEAN_VALUES = (0.4914, 0.4822, 0.4465)
CIFAR_STD_VALUES = (0.2023, 0.1994, 0.2010)


def get_dataloaders(batch_size=1024, eval_n=2000, finetune_n=4000):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN_VALUES, std=CIFAR_STD_VALUES)
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN_VALUES, std=CIFAR_STD_VALUES)
    ])

    train_full = torchvision.datasets.CIFAR10(root=PROJECT_ROOT, train=True, download=False, transform=transform_train)
    test_full = torchvision.datasets.CIFAR10(root=PROJECT_ROOT, train=False, download=False, transform=transform_test)

    finetune_subset = torch.utils.data.Subset(train_full, list(range(finetune_n)))
    eval_subset = torch.utils.data.Subset(test_full, list(range(eval_n)))

    workers = min(16, os.cpu_count() or 1)

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
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with autocast(device_type=device.type):
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


# flip weight/activation quantization on an already-built quantized model
# without rebuilding it, so one model object can be reused for all three
# ablation configs instead of re-running convert_to_quant/QAT.
def set_quant_components(model, quant_weight, quant_act):
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            mod.quant_weight = quant_weight
            mod.quant_act = quant_act


def prepare_qat(fp32_model, bits, finetune_loader, epochs=3, lr=1e-3):
    m = convert_to_quant(fp32_model, bits, quant_weight=True, quant_act=True)
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)

    set_ste_mode(m, True)
    m.train()

    opt = torch.optim.SGD(
        m.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scaler = GradScaler(device=device.type)

    for epoch in range(epochs):
        running = 0.0
        for x, y in finetune_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                loss = F.cross_entropy(m(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        print(f"  QAT epoch {epoch+1}/{epochs} avg loss {running/len(finetune_loader):.4f}")
    set_ste_mode(m, False)
    return m.eval()


CIFAR_MEAN = torch.tensor(CIFAR_MEAN_VALUES).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor(CIFAR_STD_VALUES).view(1, 3, 1, 1)
CLIP_MIN = ((0.0 - CIFAR_MEAN) / CIFAR_STD)
CLIP_MAX = ((1.0 - CIFAR_MEAN) / CIFAR_STD)


class PixelSpaceModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", CIFAR_MEAN.clone())
        self.register_buffer("std", CIFAR_STD.clone())

    def forward(self, x):
        return self.model((x - self.mean.to(x.device)) / self.std.to(x.device))


def normalize_pixels(x):
    return (x - CIFAR_MEAN.to(x.device)) / CIFAR_STD.to(x.device)


def denormalize_inputs(x):
    return x * CIFAR_STD.to(x.device) + CIFAR_MEAN.to(x.device)


def make_torchattack(attack_cls, model, *args, **kwargs):
    attack = attack_cls(model, *args, **kwargs)
    attack.set_normalization_used(mean=CIFAR_MEAN_VALUES, std=CIFAR_STD_VALUES)
    return attack


def accuracy_under_attack(model, loader, attack, target_model=None, max_images=None):
    target = target_model if target_model is not None else model
    correct, total, n_seen = 0, 0, 0
    for x, y in loader:
        if max_images is not None:
            if n_seen >= max_images:
                break
            remaining = max_images - n_seen
            x, y = x[:remaining], y[:remaining]
        x, y = x.to(device), y.to(device)
        x_pixel = denormalize_inputs(x).clamp(0.0, 1.0)
        x_adv_pixel = attack(x_pixel, y)
        x_adv = normalize_pixels(x_adv_pixel)
        with torch.no_grad():
            pred = target(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        n_seen += y.size(0)
    return correct / total if total else None

def run_fgsm_pgd(model, loader, eps=8 / 255, seeds=SEEDS):
    model.eval()
    fgsm = make_torchattack(torchattacks.FGSM, model, eps=eps)
    out = {}

    out["FGSM"] = accuracy_under_attack(model, loader, fgsm)

    pgd_accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        pgd = make_torchattack(torchattacks.PGD, model, eps=eps, alpha=2 / 255, steps=20, random_start=True)
        pgd_accs.append(accuracy_under_attack(model, loader, pgd))
    out["PGD"] = float(np.mean(pgd_accs))
    out["PGD_mean"] = float(np.mean(pgd_accs))
    out["PGD_std"] = float(np.std(pgd_accs))
    return out


def run_autoattack(model, loader, eps=8 / 255):
    model.eval()
    pixel_model = PixelSpaceModel(model).to(device).eval()
    adversary = AutoAttack(pixel_model, norm="Linf", eps=eps, version="standard", device=device, verbose=False)
    adversary.seed = 0
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_pixels = denormalize_inputs(x).clamp(0.0, 1.0)
        x_adv = normalize_pixels(adversary.run_standard_evaluation(x_pixels, y, bs=x.size(0)))
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def run_extra_whitebox_attacks(model, loader, eps=8 / 255, jsma_max_images=200):
    model.eval()
    out = {}

    cw = make_torchattack(torchattacks.CW, model, c=1, kappa=0, steps=50, lr=0.01)
    out["CW"] = accuracy_under_attack(model, loader, cw)

    deepfool = make_torchattack(torchattacks.DeepFool, model, steps=50, overshoot=0.02)
    out["DeepFool"] = accuracy_under_attack(model, loader, deepfool)

    jsma = make_torchattack(torchattacks.JSMA, model, theta=1.0, gamma=0.1)
    out["JSMA"] = accuracy_under_attack(model, loader, jsma, max_images=jsma_max_images)

    return out


def transfer_attack(source_model, target_model, loader, eps=8 / 255):
    pgd = make_torchattack(torchattacks.PGD, source_model, eps=eps, alpha=2 / 255, steps=20, random_start=True)
    return accuracy_under_attack(source_model, loader, pgd, target_model=target_model)


def transfer_attack_mim(source_model, target_model, loader, eps=8 / 255):
    mim = make_torchattack(torchattacks.MIFGSM, source_model, eps=eps, alpha=2 / 255, steps=20, decay=1.0)
    return accuracy_under_attack(source_model, loader, mim, target_model=target_model)


def build_uap(model, loader, eps=8 / 255, delta=0.2, max_iter=10, deepfool_steps=20,
              overshoot=0.02, max_images=1000):
    model.eval()
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)
    sample_x, _ = next(iter(loader))
    v = torch.zeros(1, *sample_x.shape[1:], device=device)
    deepfool = make_torchattack(torchattacks.DeepFool, model, steps=deepfool_steps, overshoot=overshoot)

    fooling_rate, it = 0.0, 0
    while fooling_rate < (1 - delta) and it < max_iter:
        n_seen, n_fooled = 0, 0
        for x, y in loader:
            if n_seen >= max_images:
                break
            x, y = x.to(device), y.to(device)
            x_pert = torch.max(torch.min(x + v, clip_max), clip_min)
            with torch.no_grad():
                pred_orig = model(x).argmax(dim=1)
                pred_pert = model(x_pert).argmax(dim=1)
            still_correct = pred_pert == pred_orig
            if still_correct.any():
                xs, ys = x_pert[still_correct], pred_orig[still_correct]
                x_adv = deepfool(xs, ys)
                v = torch.clamp(v + (x_adv - xs).mean(dim=0, keepdim=True), -eps, eps)
            n_fooled += (pred_pert != pred_orig).sum().item()
            n_seen += y.size(0)
        fooling_rate = n_fooled / max(n_seen, 1)
        it += 1
    return v.detach()


def run_uap_attack(model, loader, eps=8 / 255, max_images=1000):
    v = build_uap(model, loader, eps=eps, max_images=max_images)
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = torch.max(torch.min(x + v, clip_max), clip_min)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def transfer_uap_attack(source_model, target_model, loader, eps=8 / 255, max_images=1000):
    v = build_uap(source_model, loader, eps=eps, max_images=max_images)
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = torch.max(torch.min(x + v, clip_max), clip_min)
        with torch.no_grad():
            pred = target_model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def bpda_pgd_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=20):
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


def run_bpda(model, loader, eps=8 / 255, n_restarts=1, seeds=SEEDS):
    """
    Runs the whole worst-case-over-n_restarts procedure once per seed and
    reports mean/std across seeds, in addition to the original scalar
    (mean of seeds) for backward-compat.
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


def nes_estimate_gradient(model, x, y, n_samples=20, sigma=1e-3, query_chunk=512):
    """
    Estimates d(loss)/dx via antithetic NES sampling: for random directions
    u, the finite-difference loss(x + sigma*u) - loss(x - sigma*u) weights u
    in the gradient average. Uses only forward passes -- no backprop through
    the model. x: (B,C,H,W), y: (B,). Returns a gradient estimate shaped like x.
    """
    if n_samples % 2 != 0:
        n_samples += 1  # antithetic pairs require an even sample count
    n_pairs = n_samples // 2
    B = x.size(0)
    grad_acc = torch.zeros_like(x)

    remaining = n_pairs
    while remaining > 0:
        chunk = max(1, min(remaining, query_chunk // max(B, 1)))
        u = torch.randn(chunk, *x.shape, device=x.device)
        x_plus = (x.unsqueeze(0) + sigma * u).view(chunk * B, *x.shape[1:])
        x_minus = (x.unsqueeze(0) - sigma * u).view(chunk * B, *x.shape[1:])
        y_rep = y.repeat(chunk)

        with torch.no_grad():
            loss_plus = F.cross_entropy(model(x_plus), y_rep, reduction='none').view(chunk, B)
            loss_minus = F.cross_entropy(model(x_minus), y_rep, reduction='none').view(chunk, B)

        weight = (loss_plus - loss_minus).view(chunk, B, 1, 1, 1)
        grad_acc += (weight * u).sum(dim=0)
        remaining -= chunk

    return grad_acc / (2 * n_pairs * sigma)


def nes_pgd_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=10, n_samples=20,
                    sigma=1e-3, query_chunk=512):
    """
    Black-box Linf PGD attack that substitutes the true gradient with the NES
    estimate above. Same random-start / sign-step / projection structure as
    the white-box PGD used elsewhere in this file, so results are directly
    comparable at matched eps.
    """
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()

    for _ in range(steps):
        grad = nes_estimate_gradient(model, x_adv, y, n_samples=n_samples,
                                      sigma=sigma, query_chunk=query_chunk)
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    return x_adv.detach()


def nes_attack(model, loader, eps=8 / 255, n_samples=20, sigma=1e-3, alpha=2 / 255,
               steps=10, seed=None, query_chunk=512):
    """Single-seed NES attack accuracy over the whole loader."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = nes_pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps,
                                n_samples=n_samples, sigma=sigma, query_chunk=query_chunk)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def run_nes_attack(model, loader, eps=8 / 255, seeds=SEEDS, **kwargs):
    """Seed-averaged wrapper around nes_attack (mirrors run_random_noise_seeded)."""
    accs = [nes_attack(model, loader, eps=eps, seed=s, **kwargs) for s in seeds]
    return {
        "NES": float(np.mean(accs)),
        "NES_mean": float(np.mean(accs)),
        "NES_std": float(np.std(accs)),
    }


class SubstituteCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_substitute(target_model, seed_x, rounds=6, epochs_per_round=10, lr=1e-3,
                      lam=0.1, batch_size=128):
    target_model.eval()
    substitute = SubstituteCNN().to(device)
    opt = torch.optim.Adam(substitute.parameters(), lr=lr)
    x = seed_x.clone().to(device)

    for r in range(rounds):
        with torch.no_grad():
            y = target_model(x).argmax(dim=1)

        substitute.train()
        ds = torch.utils.data.TensorDataset(x, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs_per_round):
            for xb, yb in dl:
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(substitute(xb), yb)
                loss.backward()
                opt.step()

        if r < rounds - 1:
            substitute.eval()
            x_in = x.clone().requires_grad_(True)
            loss = F.cross_entropy(substitute(x_in), y)
            grad = torch.autograd.grad(loss, x_in)[0]
            x_aug = (x + lam * grad.sign()).detach()
            x = torch.cat([x, x_aug], dim=0)

    substitute.eval()
    return substitute


def run_surrogate_attack(model, loader, eps=8 / 255, seed_n=500, rounds=6):
    x_seed, n = [], 0
    for x, _ in loader:
        x_seed.append(x)
        n += x.size(0)
        if n >= seed_n:
            break
    x_seed = torch.cat(x_seed, dim=0)[:seed_n]

    substitute = train_substitute(model, x_seed, rounds=rounds)
    return transfer_attack(substitute, model, loader, eps=eps)


def _predict_batch(model, x):
    with torch.no_grad():
        return model(x).argmax(dim=1)


def boundary_attack_single(model, x_orig, y_true, clip_min, clip_max, steps=200,
                            spherical_step=1e-2, source_step=1e-2, step_adapt=1.5,
                            init_tries=200, init_chunk=25):
    """
    Decision-based Boundary Attack (Brendel, Bethge, 2018) for a single
    already-correctly-classified image.
    """
    clip_min = clip_min.to(x_orig.device)
    clip_max = clip_max.to(x_orig.device)

    # 1. find an initial point that the model already misclassifies
    x_adv = None
    tries_left = init_tries
    while tries_left > 0 and x_adv is None:
        chunk = min(init_chunk, tries_left)
        cand = torch.rand(chunk, *x_orig.shape, device=x_orig.device)
        cand = cand * (clip_max - clip_min) + clip_min
        preds = _predict_batch(model, cand)
        mismatch = (preds != y_true).nonzero(as_tuple=True)[0]
        if mismatch.numel() > 0:
            x_adv = cand[mismatch[0]].clone()
        tries_left -= chunk

    if x_adv is None:
        # no adversarial start found within the query budget -> treat as robust
        return x_orig.clone()

    sph_step, src_step = spherical_step, source_step
    sph_hist, src_hist = [], []

    for i in range(steps):
        diff = x_orig - x_adv
        dist = diff.norm()
        if dist.item() < 1e-12:
            break

        # random move orthogonal to the direction toward x_orig, same radius
        perturb = torch.randn_like(x_adv)
        perturb = perturb - (perturb * diff).sum() / (dist ** 2) * diff
        perturb = perturb / (perturb.norm() + 1e-12) * dist * sph_step
        cand = x_adv + perturb
        # re-project onto the sphere of radius `dist` around x_orig
        new_diff = x_orig - cand
        cand = x_orig - new_diff / (new_diff.norm() + 1e-12) * dist
        cand = torch.clamp(cand, clip_min, clip_max)

        sph_ok = (_predict_batch(model, cand.unsqueeze(0))[0] != y_true).item()
        sph_hist.append(sph_ok)

        if sph_ok:
            cand2 = torch.clamp(cand + src_step * (x_orig - cand), clip_min, clip_max)
            src_ok = (_predict_batch(model, cand2.unsqueeze(0))[0] != y_true).item()
            src_hist.append(src_ok)
            if src_ok:
                x_adv = cand2

        # adapt step sizes every 10 iters based on recent local success rate
        if (i + 1) % 10 == 0:
            if sph_hist:
                rate = np.mean(sph_hist[-10:])
                sph_step *= step_adapt if rate > 0.5 else (1 / step_adapt if rate < 0.2 else 1.0)
            if src_hist:
                rate = np.mean(src_hist[-10:])
                src_step *= step_adapt if rate > 0.5 else (1 / step_adapt if rate < 0.2 else 1.0)

    return x_adv.detach()


def run_boundary_attack(model, loader, eps=8 / 255, max_images=50, steps=200, seed=0):
    """
    Runs the Boundary Attack on up to `max_images` correctly-classified
    examples (it is inherently per-sample and query-heavy, so the full eval
    set is not used) and reports:
      - Boundary_acc: fraction of attacked images whose minimal Linf
        perturbation exceeds eps, i.e. would still be correctly classified
        within an eps budget -- directly comparable to the other
        robust-accuracy columns produced elsewhere in this file.
      - Boundary_mean_Linf: mean minimal Linf distance to the boundary found.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    clip_min, clip_max = CLIP_MIN.squeeze(0).to(device), CLIP_MAX.squeeze(0).to(device)

    dists = []
    n_seen = 0
    for x, y in loader:
        if n_seen >= max_images:
            break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        for i in range(x.size(0)):
            if n_seen >= max_images:
                break
            if pred[i] != y[i]:
                continue  # only attack already-correctly-classified examples
            x_adv = boundary_attack_single(model, x[i], y[i], clip_min, clip_max, steps=steps)
            dists.append((x_adv - x[i]).abs().max().item())
            n_seen += 1

    if not dists:
        return {"Boundary_acc": None, "Boundary_mean_Linf": None}

    dists = np.array(dists)
    return {
        "Boundary_acc": float((dists > eps).mean()),
        "Boundary_mean_Linf": float(dists.mean()),
    }


def gradient_diagnostics(model, loader, fp32_ref=None, max_batches=3):
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


def random_noise_attack(model, loader, eps=8 / 255, n_restarts=1, seed=None):
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


def run_random_noise_seeded(model, loader, eps=8 / 255, seeds=SEEDS):
    """Seed-averaged wrapper around random_noise_attack."""
    accs = [random_noise_attack(model, loader, eps=eps, seed=s) for s in seeds]
    return {
        "Random_Noise": float(np.mean(accs)),
        "Random_Noise_mean": float(np.mean(accs)),
        "Random_Noise_std": float(np.std(accs)),
    }


def pgd_steps_ablation(model, loader, eps=8 / 255, step_list=(0, 1, 2, 5, 10, 20, 50)):
    model.eval()
    out = {}
    for steps in step_list:
        if steps == 0:
            acc = random_noise_attack(model, loader, eps=eps, seed=0)
        else:
            pgd = make_torchattack(torchattacks.PGD, model, eps=eps, alpha=2 / 255, steps=steps, random_start=True)
            acc = accuracy_under_attack(model, loader, pgd)
        out[steps] = acc
    return out


def pgd_trajectory_diagnostics(model, loader, eps=8 / 255, alpha=2 / 255, steps=20, max_batches=3):
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


def staircase_diagnostic(model, loader, radius=1 / 255, n_points=40):
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


# weight-only vs activation-only vs both quantization ablation.
# Cheap by design -- reuses the already-built quantized model, just flips
# quant_weight/quant_act flags in place, and only computes clean_acc + a
# single-seed 20-step PGD + frac_zero_grad_hard per config (not the full
# AutoAttack/BPDA/trajectory suite). Restores the model to (True, True)
# (its original state) before returning.
def run_quant_component_ablation(model, loader, name, eps=8 / 255):
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
        pgd = make_torchattack(torchattacks.PGD, model, eps=eps, alpha=2 / 255, steps=20, random_start=True)
        pgd_acc = accuracy_under_attack(model, loader, pgd)

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


def run_suite(model, loader, name, fp32_ref=None, eps=8 / 255):
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

    try:
        results.update(run_extra_whitebox_attacks(model, loader, eps=eps))
    except Exception as e:
        print(f"  [WARN] CW/DeepFool/JSMA failed for {name}: {e}")

    try:
        results["Surrogate_Transfer"] = run_surrogate_attack(model, loader, eps=eps)
    except Exception as e:
        print(f"  [WARN] surrogate attack failed for {name}: {e}")
        results["Surrogate_Transfer"] = None

    if fp32_ref is not None:
        try:
            results["Transfer_from_FP32"] = transfer_attack(fp32_ref, model, loader, eps=eps)
        except Exception as e:
            print(f"  [WARN] transfer_attack failed for {name}: {e}")
            results["Transfer_from_FP32"] = None

        try:
            results["MIM_Transfer"] = transfer_attack_mim(fp32_ref, model, loader, eps=eps)
        except Exception as e:
            print(f"  [WARN] MIM transfer_attack failed for {name}: {e}")
            results["MIM_Transfer"] = None

        try:
            results["UAP_Transfer"] = transfer_uap_attack(fp32_ref, model, loader, eps=eps)
        except Exception as e:
            print(f"  [WARN] UAP transfer_attack failed for {name}: {e}")
            results["UAP_Transfer"] = None

    try:
        results.update(run_random_noise_seeded(model, loader, eps=eps))
    except Exception as e:
        print(f"  [WARN] random_noise_attack failed for {name}: {e}")
        results["Random_Noise"] = None

    if count_quant_layers(model) > 0:
        try:
            results.update(run_bpda(model, loader, eps=eps, n_restarts=2))
        except Exception as e:
            print(f"  [WARN] BPDA failed for {name}: {e}")
            results["BPDA_PGD"] = None

        try:
            results.update(gradient_diagnostics(model, loader, fp32_ref=fp32_ref, max_batches=3))
        except Exception as e:
            print(f"  [WARN] gradient_diagnostics failed for {name}: {e}")

        try:
            results.update(staircase_diagnostic(model, loader))
        except Exception as e:
            print(f"  [WARN] staircase_diagnostic failed for {name}: {e}")

        try:
            results.update(run_boundary_attack(model, loader, eps=eps, max_images=30, steps=500, seed=0))
        except Exception as e:
            print(f"  [WARN] boundary_attack failed for {name}: {e}")
            results["Boundary_acc"] = None
            results["Boundary_mean_Linf"] = None

        try:
            results.update(run_nes_attack(model, loader, eps=eps, seeds=SEEDS, n_samples=100, query_chunk=512
            ))
        except Exception as e:
            print(f"  [WARN] NES attack failed for {name}: {e}")
            results["NES"] = None

        try:
            ablation = pgd_steps_ablation(model, loader, eps=eps)
            pd.DataFrame([{"model": name, "steps": k, "acc": v} for k, v in ablation.items()]) \
                .to_csv(ablation_csv_path(name), index=False)
        except Exception as e:
            print(f"  [WARN] pgd_steps_ablation failed for {name}: {e}")

        try:
            traj = pgd_trajectory_diagnostics(model, loader, eps=eps, max_batches=3)
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

        # weight-only / activation-only / both ablation
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
            pgd = make_torchattack(torchattacks.PGD, model, eps=eps, alpha=2 / 255, steps=20, random_start=True)
            row["PGD_acc"] = accuracy_under_attack(model, loader, pgd)
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
            int8_qat = prepare_qat(fp32, bits=8, finetune_loader=finetune_loader, epochs=5)
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

    acc_cols = [c for c in ["clean_acc", "FGSM", "PGD", "CW", "DeepFool", "JSMA", "AutoAttack",
                             "Transfer_from_FP32", "MIM_Transfer", "UAP_Transfer",
                             "Surrogate_Transfer", "Random_Noise", "BPDA_PGD"]
                if c in df_results.columns]

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

    SWEEP_EPSILONS = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]

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
