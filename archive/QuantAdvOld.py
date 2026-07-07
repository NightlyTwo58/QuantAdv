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

# defenses.py is imported at the very end of this file (see bottom), after
# every name it depends on (device, denormalize_inputs, convert_to_quant,
# make_torchattack, config constants, ...) has been defined.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import *

print("device:", device)

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


def margin_json_path(model_name):
    return os.path.join(DATA_DIR, f"margin_{model_name}.json")


missing = [pkg for pkg in ("torchattacks", "autoattack") if importlib.util.find_spec(pkg) is None]
if missing:
    raise ImportError(f"Missing packages: {missing}.\nInstall via: pip install -r requirements.txt")
print("All required packages are available.")

if not os.path.isdir(CIFAR10_DIR):
    raise FileNotFoundError(f"Expected extracted CIFAR-10 at {CIFAR10_DIR!r}")


def get_dataloaders(batch_size=DEFAULT_BATCH_SIZE, eval_n=DEFAULT_EVAL_N, finetune_n=DEFAULT_FINETUNE_N):
    transform_train = T.Compose([
        T.RandomCrop(CIFAR_IMAGE_SIZE, padding=CIFAR_RANDOM_CROP_PADDING),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN_VALUES, std=CIFAR_STD_VALUES)
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR_MEAN_VALUES, std=CIFAR_STD_VALUES)
    ])

    train_full = torchvision.datasets.CIFAR10(root=PROJECT_ROOT, train=True, download=CIFAR_DOWNLOAD, transform=transform_train)
    test_full = torchvision.datasets.CIFAR10(root=PROJECT_ROOT, train=False, download=CIFAR_DOWNLOAD, transform=transform_test)

    finetune_subset = torch.utils.data.Subset(train_full, list(range(finetune_n)))
    eval_subset = torch.utils.data.Subset(test_full, list(range(eval_n)))

    workers = min(MAX_DATA_WORKERS, os.cpu_count() or 1)

    finetune_loader = torch.utils.data.DataLoader(
        finetune_subset, batch_size=batch_size, shuffle=TRAIN_SHUFFLE, num_workers=workers, pin_memory=PIN_MEMORY
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_subset, batch_size=DEFAULT_EVAL_BATCH_SIZE, shuffle=EVAL_SHUFFLE, num_workers=workers, pin_memory=PIN_MEMORY
    )

    return finetune_loader, eval_loader


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
    scale = torch.clamp(t.detach().abs().max() / qmax, min=QUANT_SCALE_MIN)
    t_scaled = t / scale
    t_round = FakeQuantSTE.apply(t_scaled) if use_ste else torch.round(t_scaled)
    t_round = torch.clamp(t_round, -qmax - 1, qmax)
    return t_round * scale


class QuantConv2d(nn.Conv2d):
    def forward(self, x):
        bits = getattr(self, 'bits', None)
        use_ste = getattr(self, 'use_ste', QUANT_DEFAULT_USE_STE)
        quant_weight = getattr(self, 'quant_weight', QUANT_DEFAULT_WEIGHT)
        quant_act = getattr(self, 'quant_act', QUANT_DEFAULT_ACT)

        w = quantize_tensor(self.weight, bits, use_ste) if quant_weight else self.weight
        out = self._conv_forward(x, w, self.bias)
        if quant_act:
            out = quantize_tensor(out, bits, use_ste)
        return out


class QuantLinear(nn.Linear):
    def forward(self, x):
        bits = getattr(self, 'bits', None)
        use_ste = getattr(self, 'use_ste', QUANT_DEFAULT_USE_STE)
        quant_weight = getattr(self, 'quant_weight', QUANT_DEFAULT_WEIGHT)
        quant_act = getattr(self, 'quant_act', QUANT_DEFAULT_ACT)

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
        new.bits, new.use_ste, new.quant_weight, new.quant_act = bits, QUANT_DEFAULT_USE_STE, quant_weight, quant_act
        return new
    if isinstance(mod, nn.Linear):
        new = QuantLinear(mod.in_features, mod.out_features, bias=mod.bias is not None)
        new.weight = mod.weight
        if mod.bias is not None:
            new.bias = mod.bias
        new.bits, new.use_ste, new.quant_weight, new.quant_act = bits, QUANT_DEFAULT_USE_STE, quant_weight, quant_act
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


def prepare_qat(fp32_model, bits, finetune_loader, epochs=QAT_EPOCHS_DEFAULT, lr=QAT_LR):
    m = convert_to_quant(fp32_model, bits, quant_weight=True, quant_act=True)
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)

    set_ste_mode(m, True)
    m.train()

    opt = torch.optim.SGD(
        m.parameters(),
        lr=lr,
        momentum=QAT_MOMENTUM,
        weight_decay=QAT_WEIGHT_DECAY,
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

def run_fgsm_pgd(model, loader, eps=DEFAULT_EPS, seeds=SEEDS):
    model.eval()
    fgsm = make_torchattack(torchattacks.FGSM, model, eps=eps)
    out = {}

    out["FGSM"] = accuracy_under_attack(model, loader, fgsm)

    pgd_accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        pgd = make_torchattack(torchattacks.PGD, model, eps=eps, alpha=PGD_ALPHA, steps=PGD_STEPS, random_start=PGD_RANDOM_START)
        pgd_accs.append(accuracy_under_attack(model, loader, pgd))
    out["PGD"] = float(np.mean(pgd_accs))
    out["PGD_mean"] = float(np.mean(pgd_accs))
    out["PGD_std"] = float(np.std(pgd_accs))
    return out


def run_autoattack(model, loader, eps=DEFAULT_EPS):
    model.eval()
    pixel_model = PixelSpaceModel(model).to(device).eval()
    adversary = AutoAttack(pixel_model, norm=AUTOATTACK_NORM, eps=eps, version=AUTOATTACK_VERSION, device=device, verbose=AUTOATTACK_VERBOSE)
    adversary.seed = AUTOATTACK_SEED
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


def run_extra_whitebox_attacks(model, loader, eps=DEFAULT_EPS, jsma_max_images=JSMA_MAX_IMAGES):
    model.eval()
    out = {}

    cw = make_torchattack(torchattacks.CW, model, c=CW_C, kappa=CW_KAPPA, steps=CW_STEPS, lr=CW_LR)
    out["CW"] = accuracy_under_attack(model, loader, cw)

    deepfool = make_torchattack(torchattacks.DeepFool, model, steps=DEEPFOOL_STEPS, overshoot=DEEPFOOL_OVERSHOOT)
    out["DeepFool"] = accuracy_under_attack(model, loader, deepfool)

    jsma = make_torchattack(torchattacks.JSMA, model, theta=JSMA_THETA, gamma=JSMA_GAMMA)
    out["JSMA"] = accuracy_under_attack(model, loader, jsma, max_images=jsma_max_images)

    return out


def transfer_attack(source_model, target_model, loader, eps=DEFAULT_EPS):
    pgd = make_torchattack(torchattacks.PGD, source_model, eps=eps, alpha=PGD_ALPHA, steps=PGD_STEPS, random_start=PGD_RANDOM_START)
    return accuracy_under_attack(source_model, loader, pgd, target_model=target_model)


def transfer_attack_mim(source_model, target_model, loader, eps=DEFAULT_EPS):
    mim = make_torchattack(torchattacks.MIFGSM, source_model, eps=eps, alpha=PGD_ALPHA, steps=PGD_STEPS, decay=MIFGSM_DECAY)
    return accuracy_under_attack(source_model, loader, mim, target_model=target_model)


def build_uap(model, loader, eps=DEFAULT_EPS, delta=UAP_DELTA, max_iter=UAP_MAX_ITER, deepfool_steps=UAP_DEEPFOOL_STEPS,
              overshoot=UAP_OVERSHOOT, max_images=UAP_MAX_IMAGES):
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


def run_uap_attack(model, loader, eps=DEFAULT_EPS, max_images=UAP_MAX_IMAGES):
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


def transfer_uap_attack(source_model, target_model, loader, eps=DEFAULT_EPS, max_images=UAP_MAX_IMAGES):
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


def bpda_pgd_attack(model, x, y, eps=DEFAULT_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS):
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


def run_bpda(model, loader, eps=DEFAULT_EPS, n_restarts=BPDA_RESTARTS_DEFAULT, seeds=SEEDS):
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


def nes_estimate_gradient(model, x, y, n_samples=NES_SAMPLES_DEFAULT, sigma=NES_SIGMA, query_chunk=NES_QUERY_CHUNK):
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


def nes_pgd_attack(model, x, y, eps=DEFAULT_EPS, alpha=PGD_ALPHA, steps=NES_STEPS, n_samples=NES_SAMPLES_DEFAULT,
                    sigma=NES_SIGMA, query_chunk=NES_QUERY_CHUNK):
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


def nes_attack(model, loader, eps=DEFAULT_EPS, n_samples=NES_SAMPLES_DEFAULT, sigma=NES_SIGMA, alpha=PGD_ALPHA,
               steps=NES_STEPS, seed=None, query_chunk=NES_QUERY_CHUNK):
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


def run_nes_attack(model, loader, eps=DEFAULT_EPS, seeds=SEEDS, **kwargs):
    """Seed-averaged wrapper around nes_attack (mirrors run_random_noise_seeded)."""
    accs = [nes_attack(model, loader, eps=eps, seed=s, **kwargs) for s in seeds]
    return {
        "NES": float(np.mean(accs)),
        "NES_mean": float(np.mean(accs)),
        "NES_std": float(np.std(accs)),
    }


class SubstituteCNN(nn.Module):
    def __init__(self, num_classes=SUBSTITUTE_NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, SUBSTITUTE_CONV1_CHANNELS, SUBSTITUTE_KERNEL_SIZE, padding=SUBSTITUTE_CONV_PADDING), nn.ReLU(inplace=SUBSTITUTE_RELU_INPLACE),
            nn.Conv2d(SUBSTITUTE_CONV1_CHANNELS, SUBSTITUTE_CONV1_CHANNELS, SUBSTITUTE_KERNEL_SIZE, padding=SUBSTITUTE_CONV_PADDING), nn.ReLU(inplace=SUBSTITUTE_RELU_INPLACE),
            nn.MaxPool2d(SUBSTITUTE_POOL_KERNEL),
            nn.Conv2d(SUBSTITUTE_CONV1_CHANNELS, SUBSTITUTE_CONV2_CHANNELS, SUBSTITUTE_KERNEL_SIZE, padding=SUBSTITUTE_CONV_PADDING), nn.ReLU(inplace=SUBSTITUTE_RELU_INPLACE),
            nn.Conv2d(SUBSTITUTE_CONV2_CHANNELS, SUBSTITUTE_CONV2_CHANNELS, SUBSTITUTE_KERNEL_SIZE, padding=SUBSTITUTE_CONV_PADDING), nn.ReLU(inplace=SUBSTITUTE_RELU_INPLACE),
            nn.MaxPool2d(SUBSTITUTE_POOL_KERNEL),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(SUBSTITUTE_CONV2_CHANNELS * SUBSTITUTE_LINEAR_FEATURE_MAP * SUBSTITUTE_LINEAR_FEATURE_MAP, SUBSTITUTE_HIDDEN_DIM), nn.ReLU(inplace=SUBSTITUTE_RELU_INPLACE),
            nn.Linear(SUBSTITUTE_HIDDEN_DIM, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_substitute(target_model, seed_x, rounds=SUBSTITUTE_ROUNDS, epochs_per_round=SUBSTITUTE_EPOCHS_PER_ROUND, lr=SUBSTITUTE_LR,
                      lam=SUBSTITUTE_LAMBDA, batch_size=SUBSTITUTE_BATCH_SIZE):
    target_model.eval()
    substitute = SubstituteCNN().to(device)
    opt = torch.optim.Adam(substitute.parameters(), lr=lr)
    x = seed_x.clone().to(device)

    for r in range(rounds):
        with torch.no_grad():
            y = target_model(x).argmax(dim=1)

        substitute.train()
        ds = torch.utils.data.TensorDataset(x, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=TRAIN_SHUFFLE)
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


def run_surrogate_attack(model, loader, eps=DEFAULT_EPS, seed_n=SURROGATE_SEED_N, rounds=SUBSTITUTE_ROUNDS):
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


def boundary_attack_single(model, x_orig, y_true, clip_min, clip_max, steps=BOUNDARY_STEPS_DEFAULT,
                            spherical_step=BOUNDARY_SPHERICAL_STEP, source_step=BOUNDARY_SOURCE_STEP, step_adapt=BOUNDARY_STEP_ADAPT,
                            init_tries=BOUNDARY_INIT_TRIES, init_chunk=BOUNDARY_INIT_CHUNK):
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
        if dist.item() < BOUNDARY_MIN_DIST:
            break

        # random move orthogonal to the direction toward x_orig, same radius
        perturb = torch.randn_like(x_adv)
        perturb = perturb - (perturb * diff).sum() / (dist ** 2) * diff
        perturb = perturb / (perturb.norm() + BOUNDARY_MIN_DIST) * dist * sph_step
        cand = x_adv + perturb
        # re-project onto the sphere of radius `dist` around x_orig
        new_diff = x_orig - cand
        cand = x_orig - new_diff / (new_diff.norm() + BOUNDARY_MIN_DIST) * dist
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
        if (i + 1) % BOUNDARY_ADAPT_INTERVAL == 0:
            if sph_hist:
                rate = np.mean(sph_hist[-10:])
                sph_step *= step_adapt if rate > BOUNDARY_SPH_SUCCESS_HIGH else (1 / step_adapt if rate < BOUNDARY_SPH_SUCCESS_LOW else 1.0)
            if src_hist:
                rate = np.mean(src_hist[-10:])
                src_step *= step_adapt if rate > BOUNDARY_SPH_SUCCESS_HIGH else (1 / step_adapt if rate < BOUNDARY_SPH_SUCCESS_LOW else 1.0)

    return x_adv.detach()


def run_boundary_attack(model, loader, eps=DEFAULT_EPS, max_images=BOUNDARY_MAX_IMAGES_DEFAULT, steps=BOUNDARY_STEPS_DEFAULT, seed=BOUNDARY_SEED):
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


def gradient_diagnostics(model, loader, fp32_ref=None, max_batches=GRAD_DIAG_MAX_BATCHES):
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
        frac_zero_hard.append((g_hard.abs() < GRAD_ZERO_THRESHOLD).float().mean().item())
        norm_hard.append(g_hard.norm().item())

        set_ste_mode(model, True)
        x_in2 = x.clone().requires_grad_(True)
        loss2 = F.cross_entropy(model(x_in2), y)
        g_ste = torch.autograd.grad(loss2, x_in2)[0].flatten()
        set_ste_mode(model, False)
        frac_zero_ste.append((g_ste.abs() < GRAD_ZERO_THRESHOLD).float().mean().item())
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


def random_noise_attack(model, loader, eps=DEFAULT_EPS, n_restarts=1, seed=None):
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


def run_random_noise_seeded(model, loader, eps=DEFAULT_EPS, seeds=SEEDS):
    """Seed-averaged wrapper around random_noise_attack."""
    accs = [random_noise_attack(model, loader, eps=eps, seed=s) for s in seeds]
    return {
        "Random_Noise": float(np.mean(accs)),
        "Random_Noise_mean": float(np.mean(accs)),
        "Random_Noise_std": float(np.std(accs)),
    }


def pgd_steps_ablation(model, loader, eps=DEFAULT_EPS, step_list=PGD_ABLATION_STEPS):
    model.eval()
    out = {}
    for steps in step_list:
        if steps == 0:
            acc = random_noise_attack(model, loader, eps=eps, seed=0)
        else:
            pgd = make_torchattack(torchattacks.PGD, model, eps=eps, alpha=PGD_ALPHA, steps=steps, random_start=PGD_RANDOM_START)
            acc = accuracy_under_attack(model, loader, pgd)
        out[steps] = acc
    return out


def pgd_trajectory_diagnostics(model, loader, eps=DEFAULT_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS, max_batches=TRAJECTORY_MAX_BATCHES):
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


def layerwise_grad_profile(model, loader, use_ste, max_batches=LAYERWISE_MAX_BATCHES):
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


def staircase_diagnostic(model, loader, radius=STAIRCASE_RADIUS, n_points=STAIRCASE_N_POINTS):
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
def run_quant_component_ablation(model, loader, name, eps=DEFAULT_EPS):
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
        pgd = make_torchattack(torchattacks.PGD, model, eps=eps, alpha=PGD_ALPHA, steps=PGD_STEPS, random_start=PGD_RANDOM_START)
        pgd_acc = accuracy_under_attack(model, loader, pgd)

        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        x_in = x.clone().requires_grad_(True)
        loss = F.cross_entropy(model(x_in), y)
        g_hard = torch.autograd.grad(loss, x_in)[0].flatten()
        frac_zero = (g_hard.abs() < GRAD_ZERO_THRESHOLD).float().mean().item()

        rows.append({
            "model": name, "config": label,
            "quant_weight": qw, "quant_act": qa,
            "clean_acc": clean_acc, "PGD_acc": pgd_acc,
            "frac_zero_grad_hard": frac_zero,
        })

    # restore original (both quantized) state
    set_quant_components(model, True, True)
    return rows


def confidence_margin_diagnostic(model, loader, eps=DEFAULT_EPS, steps=MARGIN_STEPS, max_batches=MARGIN_MAX_BATCHES):
    model.eval()
    pgd = make_torchattack(torchattacks.PGD, model, eps=eps, alpha=PGD_ALPHA, steps=steps, random_start=PGD_RANDOM_START)
    clean_margins, adv_margins = [], []
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            top2 = F.softmax(model(x), dim=1).topk(2, dim=1).values
        clean_margins.extend((top2[:, 0] - top2[:, 1]).cpu().tolist())

        x_pixel = denormalize_inputs(x).clamp(0.0, 1.0)
        x_adv = normalize_pixels(pgd(x_pixel, y))
        with torch.no_grad():
            top2_adv = F.softmax(model(x_adv), dim=1).topk(2, dim=1).values
        adv_margins.extend((top2_adv[:, 0] - top2_adv[:, 1]).cpu().tolist())

    return {"clean_margins": clean_margins, "adv_margins": adv_margins}


def run_suite(model, loader, name, fp32_ref=None, eps=DEFAULT_EPS):
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
            results.update(run_bpda(model, loader, eps=eps, n_restarts=BPDA_RESTARTS_SUITE))
        except Exception as e:
            print(f"  [WARN] BPDA failed for {name}: {e}")
            results["BPDA_PGD"] = None

        try:
            results.update(gradient_diagnostics(model, loader, fp32_ref=fp32_ref, max_batches=GRAD_DIAG_MAX_BATCHES))
        except Exception as e:
            print(f"  [WARN] gradient_diagnostics failed for {name}: {e}")

        try:
            results.update(staircase_diagnostic(model, loader))
        except Exception as e:
            print(f"  [WARN] staircase_diagnostic failed for {name}: {e}")

        try:
            results.update(run_boundary_attack(model, loader, eps=eps, max_images=BOUNDARY_MAX_IMAGES_SUITE, steps=BOUNDARY_STEPS_SUITE, seed=BOUNDARY_SEED))
        except Exception as e:
            print(f"  [WARN] boundary_attack failed for {name}: {e}")
            results["Boundary_acc"] = None
            results["Boundary_mean_Linf"] = None

        try:
            results.update(run_nes_attack(model, loader, eps=eps, seeds=SEEDS, n_samples=NES_SAMPLES_SUITE, query_chunk=NES_QUERY_CHUNK
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
            traj = pgd_trajectory_diagnostics(model, loader, eps=eps, max_batches=TRAJECTORY_MAX_BATCHES)
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

        try:
            margins = confidence_margin_diagnostic(model, loader, eps=eps, max_batches=MARGIN_MAX_BATCHES)
            with open(margin_json_path(name), "w") as f:
                json.dump(margins, f)
        except Exception as e:
            print(f"  [WARN] confidence_margin_diagnostic failed for {name}: {e}")

    return results


def run_epsilon_sweep_for_model(model, loader, name, epsilons):
    rows = []
    is_quant = count_quant_layers(model) > 0
    for eps in epsilons:
        row = {"model": name, "epsilon": eps}
        try:
            pgd = make_torchattack(torchattacks.PGD, model, eps=eps, alpha=PGD_ALPHA, steps=PGD_STEPS, random_start=PGD_RANDOM_START)
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
                row["BPDA_acc"] = _run_bpda_once(model, loader, eps=eps, n_restarts=BPDA_RESTARTS_SWEEP)
            except Exception as e:
                print(f"  [WARN] BPDA sweep failed for {name} eps={eps:.4f}: {e}")
                row["BPDA_acc"] = None
        rows.append(row)
    return rows


def plot_epsilon_sweep_curves(df_sweep):
    if df_sweep is None or df_sweep.empty:
        return
    value_cols = [c for c in ["PGD_acc", "Random_Noise_acc", "BPDA_acc"] if c in df_sweep.columns]
    if not value_cols:
        return
    df_long = df_sweep.melt(id_vars=["model", "epsilon"], value_vars=value_cols, var_name="Attack", value_name="Accuracy")
    df_long = df_long.dropna(subset=["Accuracy"])
    if df_long.empty:
        return

    models = df_long["model"].unique()
    cols = min(SWEEP_PLOT_COLS_MAX, len(models))
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(SWEEP_PLOT_WIDTH * cols, SWEEP_PLOT_HEIGHT * rows), squeeze=False)
    for i, m in enumerate(models):
        ax = axes[i // cols][i % cols]
        sns.lineplot(data=df_long[df_long["model"] == m], x="epsilon", y="Accuracy", hue="Attack", marker="o", ax=ax)
        ax.set_title(m)
        ax.set_ylim(0, PLOT_MAX_ACCURACY)
        ax.grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    for j in range(len(models), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Accuracy vs Perturbation Budget (Epsilon Sweep)")
    fig.tight_layout()
    fig.savefig(SWEEP_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_pgd_steps_ablation(model_names):
    frames = [pd.read_csv(ablation_csv_path(n)) for n in model_names if os.path.exists(ablation_csv_path(n))]
    if not frames:
        return
    df_all = pd.concat(frames, ignore_index=True)

    plt.figure(figsize=ABLATION_FIGSIZE)
    sns.lineplot(data=df_all, x="steps", y="acc", hue="model", marker="o")
    plt.title("PGD Accuracy vs Number of Steps (Gradient Masking Check)")
    plt.xlabel("PGD steps")
    plt.ylabel("Accuracy")
    plt.ylim(0, PLOT_MAX_ACCURACY)
    plt.grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(ABLATION_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_pgd_trajectory(model_names):
    trajs = {}
    for name in model_names:
        p = trajectory_json_path(name)
        if os.path.exists(p):
            with open(p) as f:
                trajs[name] = json.load(f)
    if not trajs:
        return

    fig, axes = plt.subplots(1, 2, figsize=TRAJECTORY_FIGSIZE)
    for name, traj in trajs.items():
        steps = range(1, len(traj["grad_norm_per_step"]) + 1)
        axes[0].plot(steps, traj["grad_norm_per_step"], marker="o", label=name)
        axes[1].plot(steps, traj["movement_from_random_start_per_step"], marker="o", label=name)

    axes[0].set_title("Gradient Norm per PGD Step")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Grad Norm")
    axes[0].set_yscale("log")
    axes[0].grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    axes[0].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    axes[1].set_title("Perturbation Movement per PGD Step")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Linf Movement from Random Start")
    axes[1].grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    axes[1].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    fig.tight_layout()
    fig.savefig(TRAJECTORY_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_layerwise_grad_profile(model_names):
    quant_names = [n for n in model_names if os.path.exists(layerwise_csv_path(n))]
    if not quant_names:
        return

    cols = min(LAYERWISE_PLOT_COLS_MAX, len(quant_names))
    rows = int(np.ceil(len(quant_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(LAYERWISE_PLOT_WIDTH * cols, LAYERWISE_PLOT_HEIGHT * rows), squeeze=False)
    for i, name in enumerate(quant_names):
        df = pd.read_csv(layerwise_csv_path(name))
        ax = axes[i // cols][i % cols]
        x = np.arange(len(df))
        ax.plot(x, df["grad_norm_hard"], marker="o", label="hard-round")
        ax.plot(x, df["grad_norm_ste"], marker="o", label="STE")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(df["layer"], rotation=LAYERWISE_XTICK_ROTATION, fontsize=LAYERWISE_XTICK_FONT_SIZE)
        ax.set_title(name)
        ax.set_ylabel("Grad Norm (log)")
        ax.legend(fontsize=PLOT_LEGEND_FONT_SIZE)
        ax.grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    for j in range(len(quant_names), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Layerwise Gradient Norms: Hard-Round vs STE")
    fig.tight_layout()
    fig.savefig(LAYERWISE_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_component_ablation(model_names):
    frames = [pd.read_csv(component_ablation_csv_path(n)) for n in model_names if os.path.exists(component_ablation_csv_path(n))]
    if not frames:
        return
    df_all = pd.concat(frames, ignore_index=True)
    df_long = df_all.melt(id_vars=["model", "config"], value_vars=["clean_acc", "PGD_acc"], var_name="Metric", value_name="Accuracy")

    g = sns.catplot(data=df_long, x="config", y="Accuracy", hue="Metric", col="model", kind="bar", col_wrap=COMPONENT_ABLATION_COL_WRAP, height=COMPONENT_ABLATION_HEIGHT, sharey=True)
    g.set_titles("{col_name}")
    g.set(ylim=(0, PLOT_MAX_ACCURACY))
    g.savefig(COMPONENT_ABLATION_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_gradient_masking_summary(df_results):
    if df_results is None or df_results.empty or not {"model", "PGD", "AutoAttack"}.issubset(df_results.columns):
        return
    df = df_results.dropna(subset=["PGD", "AutoAttack"]).copy()
    if df.empty:
        return
    df["PGD_minus_AutoAttack"] = df["PGD"] - df["AutoAttack"]

    fig, axes = plt.subplots(1, 2, figsize=MASKING_SUMMARY_FIGSIZE)
    sns.barplot(data=df, x="model", y="PGD_minus_AutoAttack", ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=MASKING_BASELINE_LINEWIDTH)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=SUMMARY_XTICK_ROTATION, ha="right")
    axes[0].set_title("PGD - AutoAttack Accuracy Gap")
    axes[0].grid(axis="y", linestyle="--", alpha=PLOT_GRID_ALPHA)

    if "frac_zero_grad_hard" in df.columns:
        df2 = df.dropna(subset=["frac_zero_grad_hard"])
        sns.scatterplot(data=df2, x="frac_zero_grad_hard", y="PGD_minus_AutoAttack", hue="model", s=MASKING_SCATTER_SIZE, ax=axes[1])
        axes[1].set_title("Masking Gap vs Fraction of Zero Gradients")
        axes[1].grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(MASKING_SUMMARY_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_confidence_margin_diagnostic(model_names):
    data = {}
    for name in model_names:
        p = margin_json_path(name)
        if os.path.exists(p):
            with open(p) as f:
                data[name] = json.load(f)
    if not data:
        return

    cols = min(MARGIN_PLOT_COLS_MAX, len(data))
    rows = int(np.ceil(len(data) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(MARGIN_PLOT_WIDTH * cols, MARGIN_PLOT_HEIGHT * rows), squeeze=False)
    for i, (name, margins) in enumerate(data.items()):
        ax = axes[i // cols][i % cols]
        ax.hist(margins["clean_margins"], bins=MARGIN_HIST_BINS, alpha=MARGIN_HIST_ALPHA, label="clean", density=True)
        ax.hist(margins["adv_margins"], bins=MARGIN_HIST_BINS, alpha=MARGIN_HIST_ALPHA, label="PGD-adv", density=True)
        ax.set_title(name)
        ax.set_xlabel("Top1 - Top2 Softmax Margin")
        ax.legend(fontsize=PLOT_LEGEND_FONT_SIZE)
        ax.grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    for j in range(len(data), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Confidence Margin: Clean vs PGD-Adversarial")
    fig.tight_layout()
    fig.savefig(MARGIN_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_results_heatmap(df_results):
    if df_results is None or df_results.empty:
        return
    candidate_cols = ["clean_acc", "FGSM", "PGD", "AutoAttack", "CW", "DeepFool", "JSMA",
                      "Surrogate_Transfer", "Transfer_from_FP32", "MIM_Transfer", "UAP_Transfer",
                      "Random_Noise", "BPDA_PGD", "NES", "Boundary_acc"]
    cols = [c for c in candidate_cols if c in df_results.columns and df_results[c].notna().any()]
    if not cols:
        return
    df_heat = df_results.set_index("model")[cols].astype(float)

    plt.figure(figsize=(max(HEATMAP_MIN_WIDTH, len(cols)), max(HEATMAP_MIN_HEIGHT, len(df_heat) * HEATMAP_ROW_HEIGHT)))
    sns.heatmap(df_heat, annot=True, fmt=".2f", cmap="RdYlGn", vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, linewidths=HEATMAP_LINEWIDTHS)
    plt.title("Full Results Heatmap: Models vs Attacks")
    plt.tight_layout()
    plt.savefig(HEATMAP_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


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
            int8_ptq = convert_to_quant(fp32, bits=QAT_BITS, quant_weight=True, quant_act=True)
            model_registry[f"{arch_key}_int8_PTQ"] = (int8_ptq, fp32)
        except Exception as e:
            print(f"  [FAIL] int8 PTQ for {arch_key}: {e}")

        try:
            int4_ptq = convert_to_quant(fp32, bits=4, quant_weight=True, quant_act=True)
            model_registry[f"{arch_key}_int4_PTQ"] = (int4_ptq, fp32)
        except Exception as e:
            print(f"  [FAIL] int4 PTQ for {arch_key}: {e}")

        try:
            int8_qat = prepare_qat(fp32, bits=QAT_BITS, finetune_loader=finetune_loader, epochs=QAT_MAIN_EPOCHS)
            model_registry[f"{arch_key}_int8_QAT"] = (int8_qat, fp32)
        except Exception as e:
            print(f"  [FAIL] int8 QAT for {arch_key}: {e}")
            traceback.print_exc()

    try:
        model_registry, df_defense_summary = run_defense_suite(model_registry, finetune_loader, eval_loader)
        if not df_defense_summary.empty:
            print("\nDefense summary (guardrail/detector flag rates, certified accuracy):")
            print(df_defense_summary.to_string(index=False))
    except Exception as e:
        print(f"  [FAIL] run_defense_suite failed: {e}")
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

        plt.figure(figsize=SUMMARY_PLOT_FIGSIZE)
        sns.barplot(data=df_plot, x="model", y="Accuracy", hue="Attack")
        plt.xticks(rotation=SUMMARY_XTICK_ROTATION, ha="right")
        plt.title("Model Accuracy under Various Adversarial Attacks")
        plt.ylim(0, PLOT_MAX_ACCURACY)
        plt.grid(axis="y", linestyle="--", alpha=SUMMARY_GRID_ALPHA)
        plt.tight_layout()
        plt.savefig(PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
        plt.show()

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

    model_names = list(model_registry.keys())

    try:
        plot_epsilon_sweep_curves(df_sweep)
    except Exception as e:
        print(f"  [WARN] plot_epsilon_sweep_curves failed: {e}")

    try:
        plot_pgd_steps_ablation(model_names)
    except Exception as e:
        print(f"  [WARN] plot_pgd_steps_ablation failed: {e}")

    try:
        plot_pgd_trajectory(model_names)
    except Exception as e:
        print(f"  [WARN] plot_pgd_trajectory failed: {e}")

    try:
        plot_layerwise_grad_profile(model_names)
    except Exception as e:
        print(f"  [WARN] plot_layerwise_grad_profile failed: {e}")

    try:
        plot_component_ablation(model_names)
    except Exception as e:
        print(f"  [WARN] plot_component_ablation failed: {e}")

    try:
        plot_gradient_masking_summary(df_results)
    except Exception as e:
        print(f"  [WARN] plot_gradient_masking_summary failed: {e}")

    try:
        plot_confidence_margin_diagnostic(model_names)
    except Exception as e:
        print(f"  [WARN] plot_confidence_margin_diagnostic failed: {e}")

    try:
        plot_results_heatmap(df_results)
    except Exception as e:
        print(f"  [WARN] plot_results_heatmap failed: {e}")

    try:
        plot_defense_comparison(df_results)
    except Exception as e:
        print(f"  [WARN] plot_defense_comparison failed: {e}")

    print("All done.")



def defense_summary_csv_path():
    return os.path.join(DATA_DIR, "defense_summary.csv")


def run_defense_suite(model_registry, finetune_loader, eval_loader):
    """
    Additive defense pass:
      1. Adversarial-trained variants (FP32 + int8) straight into
         model_registry so they flow through the *existing* run_suite /
         epsilon-sweep / plotting code unchanged.
      2. Input-sanitization, certified-smoothing, and guardrail wrappers
         around the plain FP32 and int8_QAT models (also added to
         model_registry for the same reason).
      3. A detection-based active defense (ConvNeXt guard model) wrapped
         around FP32 and int8_QAT.
    Returns the (mutated) model_registry plus a defense-specific summary
    dataframe (guardrail/detector flag rates, certified accuracy) that
    doesn't fit the existing per-attack CSV schema.
    """
    import defenses as dfn

    summary_rows = []

    arch_keys = sorted({name.split("_FP32")[0] for name in model_registry if name.endswith("_FP32")})

    for arch_key in arch_keys:
        fp32_entry = model_registry.get(f"{arch_key}_FP32")
        qat_entry = model_registry.get(f"{arch_key}_int8_QAT")
        if fp32_entry is None:
            continue
        fp32_model = fp32_entry[0]

        try:
            fp32_at = dfn.prepare_adversarial_training(fp32_model, finetune_loader, bits=None)
            model_registry[f"{arch_key}_FP32_AT"] = (fp32_at, fp32_model)
        except Exception as e:
            print(f"  [FAIL] adversarial training (FP32) for {arch_key}: {e}")
            traceback.print_exc()

        try:
            int8_at = dfn.prepare_adversarial_training(fp32_model, finetune_loader, bits=QAT_BITS)
            model_registry[f"{arch_key}_int8_QAT_AT"] = (int8_at, fp32_model)
        except Exception as e:
            print(f"  [FAIL] adversarial training (int8) for {arch_key}: {e}")
            traceback.print_exc()

        wrap_targets = [("FP32", fp32_model)]
        if qat_entry is not None:
            wrap_targets.append(("int8_QAT", qat_entry[0]))

        detector = None
        try:
            detector = dfn.train_adversarial_detector(fp32_model, finetune_loader)
        except Exception as e:
            print(f"  [FAIL] adversarial detector training for {arch_key}: {e}")
            traceback.print_exc()

        for tag, base_model in wrap_targets:
            entry_name = f"{arch_key}_{tag}"

            try:
                sanitized = dfn.SanitizedModel(base_model).to(device).eval()
                model_registry[f"{entry_name}_Sanitized"] = (sanitized, fp32_model)
            except Exception as e:
                print(f"  [FAIL] SanitizedModel for {entry_name}: {e}")

            try:
                smoothed = dfn.SmoothedModel(base_model).to(device).eval()
                model_registry[f"{entry_name}_Smoothed"] = (smoothed, fp32_model)
                cert_stats = dfn.run_certified_accuracy(smoothed, eval_loader)
                summary_rows.append({"model": entry_name, "defense": "randomized_smoothing", **cert_stats})
            except Exception as e:
                print(f"  [FAIL] SmoothedModel/certification for {entry_name}: {e}")
                traceback.print_exc()

            try:
                guardrail = dfn.GuardrailModel(base_model).to(device).eval()
                model_registry[f"{entry_name}_Guardrail"] = (guardrail, fp32_model)
                pgd_for_flagging = make_torchattack(torchattacks.PGD, guardrail, eps=DEFAULT_EPS,
                                                     alpha=PGD_ALPHA, steps=PGD_STEPS, random_start=PGD_RANDOM_START)
                flag_stats = dfn.run_guardrail_flagging_rate(guardrail, eval_loader, attack=pgd_for_flagging)
                summary_rows.append({"model": entry_name, "defense": "guardrail", **flag_stats})
            except Exception as e:
                print(f"  [FAIL] GuardrailModel for {entry_name}: {e}")
                traceback.print_exc()

            if detector is not None:
                try:
                    detect_guard = dfn.DetectGuardModel(base_model, detector).to(device).eval()
                    model_registry[f"{entry_name}_DetectGuard"] = (detect_guard, fp32_model)
                    pgd_for_detect = make_torchattack(torchattacks.PGD, detect_guard, eps=DEFAULT_EPS,
                                                       alpha=PGD_ALPHA, steps=PGD_STEPS, random_start=PGD_RANDOM_START)
                    catch_stats = dfn.run_detector_catch_rate(detect_guard, eval_loader, attack=pgd_for_detect)
                    summary_rows.append({"model": entry_name, "defense": "detector", **catch_stats})
                except Exception as e:
                    print(f"  [FAIL] DetectGuardModel for {entry_name}: {e}")
                    traceback.print_exc()

    df_defense = pd.DataFrame(summary_rows)
    if not df_defense.empty:
        df_defense.to_csv(defense_summary_csv_path(), index=False)
    return model_registry, df_defense


def plot_defense_comparison(df_results):
    if df_results is None or df_results.empty:
        return
    defense_tags = ("_AT", "_Sanitized", "_Smoothed", "_Guardrail", "_DetectGuard")
    df_def = df_results[df_results["model"].astype(str).str.contains("|".join(defense_tags))]
    if df_def.empty:
        return
    cols = [c for c in ["clean_acc", "PGD", "AutoAttack"] if c in df_def.columns and df_def[c].notna().any()]
    if not cols:
        return
    df_long = df_def.melt(id_vars="model", value_vars=cols, var_name="Attack", value_name="Accuracy")

    plt.figure(figsize=SUMMARY_PLOT_FIGSIZE)
    sns.barplot(data=df_long, x="model", y="Accuracy", hue="Attack")
    plt.xticks(rotation=SUMMARY_XTICK_ROTATION, ha="right")
    plt.title("Defense Variants: Accuracy under Attack")
    plt.ylim(0, PLOT_MAX_ACCURACY)
    plt.grid(axis="y", linestyle="--", alpha=SUMMARY_GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "defense_comparison.png"), dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


if __name__ == '__main__':
    main()