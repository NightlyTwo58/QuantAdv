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
import argparse
import subprocess
from contextlib import nullcontext
import matplotlib.pyplot as plt
import seaborn as sns

import torchattacks
from autoattack import AutoAttack

from quantize import Model as QuantModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

USE_AMP = torch.cuda.is_available()


def amp_ctx():
    if USE_AMP:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arch-key", default=None,
                        help="Internal: restrict this run to a single architecture.")
    args, _ = parser.parse_known_args()
    return args


_ARGS = _parse_args()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
SWEEP_CSV = os.path.join(DATA_DIR, "results_sweep.csv")
PLOT_PNG = os.path.join(DATA_DIR, "accuracy_plot.png")
CIFAR10_ROOT = os.environ.get("CIFAR10_ROOT", "./")

SEEDS = [0, 1, 2]

"""
Archived non-threaded single Python invocation analysis of metrics of attacks per model and eplison

Migrated to use quantize.Model class interface (torchao-based quantization).
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

if _ARGS.arch_key is not None:
    if _ARGS.arch_key not in PRETRAINED_NAMES:
        raise ValueError(f"Unknown --arch-key {_ARGS.arch_key!r}, expected one of {list(PRETRAINED_NAMES)}")
    PRETRAINED_NAMES = {_ARGS.arch_key: PRETRAINED_NAMES[_ARGS.arch_key]}
    RESULTS_CSV = os.path.join(DATA_DIR, f"results_{_ARGS.arch_key}.csv")
    SWEEP_CSV = os.path.join(DATA_DIR, f"results_sweep_{_ARGS.arch_key}.csv")


def load_pretrained(arch_key):
    hub_name = PRETRAINED_NAMES[arch_key]
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", hub_name, pretrained=True)
    return model.to(device).eval()


# Evaluation helpers (delegate to QuantModel interface)

def sanity_check_accuracy(model, loader):
    """Delegate to QuantModel.clean_accuracy."""
    return QuantModel.clean_accuracy(model, loader)


def count_quant_layers(model):
    """Count quantized layers using the QuantModel helper."""
    return QuantModel._count_quant_layers(model)


# Attack functions — unchanged from old version (only depend on forward pass)

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
CLIP_MIN = ((0.0 - CIFAR_MEAN) / CIFAR_STD)
CLIP_MAX = ((1.0 - CIFAR_MEAN) / CIFAR_STD)


def run_fgsm_pgd(model, loader, eps=8 / 255, seeds=SEEDS):
    model.eval()
    fgsm = torchattacks.FGSM(model, eps=eps)
    out = {}

    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with amp_ctx():
            x_adv = fgsm(x, y)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    out["FGSM"] = correct / total

    pgd_accs = []
    # PGD's random start is drawn at call-time (from the global RNG), not at
    # construction time, so the attack object can be built once and reused
    # across seeds -- we just need to reseed before each call.
    pgd = torchattacks.PGD(model, eps=eps, alpha=2 / 255, steps=20, random_start=True)
    for seed in seeds:
        torch.manual_seed(seed)
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with amp_ctx():
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


def run_autoattack(model, loader, eps=8 / 255, aa_batch_size=256):
    model.eval()
    adversary = AutoAttack(model, norm="Linf", eps=eps, version="custom", device=device, verbose=False)
    adversary.attacks_to_run = ["apgd-ce", "apgd-t"]

    x_all = torch.cat([x for x, _ in loader], dim=0)
    y_all = torch.cat([y for _, y in loader], dim=0)
    bs = min(aa_batch_size, x_all.size(0))

    with amp_ctx():
        x_adv = adversary.run_standard_evaluation(x_all, y_all, bs=bs)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
    correct = (pred == y_all).sum().item()
    total = y_all.size(0)
    return correct / total


def transfer_attack(source_model, target_model, loader, eps=8 / 255):
    pgd = torchattacks.PGD(source_model, eps=eps, alpha=2 / 255, steps=20, random_start=True)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with amp_ctx():
            x_adv = pgd(x, y)
            with torch.no_grad():
                pred = target_model(x_adv).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def bpda_pgd_attack(model, x, y, eps=8 / 255, alpha=2 / 255, steps=20):
    clip_min = CLIP_MIN.to(device)
    clip_max = CLIP_MAX.to(device)
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with amp_ctx():
            loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    return x_adv.detach()


def _run_bpda_once(model, loader, eps, n_restarts):
    correct_masks = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        worst_correct = torch.ones(y.size(0), dtype=torch.bool, device=device)
        for _ in range(n_restarts):
            x_adv = bpda_pgd_attack(model, x, y, eps=eps)
            with torch.no_grad(), amp_ctx():
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


def gradient_diagnostics_and_layerwise_profile(model, loader, fp32_ref=None, max_batches=5):
    """
    Combined replacement for the old gradient_diagnostics() +
    layerwise_grad_profile() pair (items #5/#6). Both functions looped over
    the same eval batches and independently ran a forward + backward pass
    to get an input gradient -- the only difference was gradient_diagnostics
    read x_in.grad-style stats via autograd.grad, while layerwise_grad_profile
    used backward hooks on the quantized layers. Backward hooks fire during
    ANY backward pass through the module (autograd.grad or .backward()), so
    a single backward pass per batch now feeds both.

    Returns (diagnostics_dict, layerwise_profile_dict).
    """
    quant_layers = [(n, m) for n, m in model.named_modules()
                    if hasattr(m, '_quantized_op') or hasattr(m, 'quantizer')]
    layer_norms = {n: [] for n, _ in quant_layers}
    handles = []

    def make_hook(name):
        def hook(module, grad_input, grad_output):
            gi = grad_input[0]
            if gi is not None:
                layer_norms[name].append(gi.flatten(1).norm(dim=1).mean().item())

        return hook

    for n, m in quant_layers:
        handles.append(m.register_full_backward_hook(make_hook(n)))

    frac_zero_hard, norm_hard = [], []
    cos_sims = []

    model.eval()
    try:
        for bi, (x, y) in enumerate(loader):
            if bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)

            x_in = x.clone().requires_grad_(True)
            with amp_ctx():
                loss = F.cross_entropy(model(x_in), y)
            g_hard = torch.autograd.grad(loss, x_in)[0].flatten()
            frac_zero_hard.append((g_hard.abs() < 1e-8).float().mean().item())
            norm_hard.append(g_hard.norm().item())

            if fp32_ref is not None:
                fp32_ref.eval()
                x_ref = x.clone().requires_grad_(True)
                with amp_ctx():
                    loss_ref = F.cross_entropy(fp32_ref(x_ref), y)
                g_ref = torch.autograd.grad(loss_ref, x_ref)[0].flatten()
                cos_sims.append(F.cosine_similarity(g_hard.unsqueeze(0), g_ref.unsqueeze(0)).item())
    finally:
        for h in handles:
            h.remove()

    diagnostics = {
        "frac_zero_grad_hard": float(np.mean(frac_zero_hard)),
        "frac_zero_grad_ste": float(np.mean(frac_zero_hard)),
        "grad_norm_hard": float(np.mean(norm_hard)),
        "grad_norm_ste": float(np.mean(norm_hard)),
    }
    if cos_sims:
        diagnostics["grad_cosine_sim_with_FP32"] = float(np.mean(cos_sims))

    ordered_names = [n for n, _ in quant_layers]
    layerwise_profile = {
        n: (float(np.mean(layer_norms[n])) if len(layer_norms[n]) else None)
        for n in ordered_names
    }
    return diagnostics, layerwise_profile


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
                with amp_ctx():
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
            pgd = torchattacks.PGD(model, eps=eps, alpha=2 / 255, steps=steps, random_start=True)
            correct, total = 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with amp_ctx():
                    x_adv = pgd(x, y)
                    with torch.no_grad():
                        pred = model(x_adv).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            acc = correct / total
        out[steps] = acc
    return out


def pgd_trajectory_diagnostics(model, loader, eps=8 / 255, alpha=2 / 255, steps=20, max_batches=5):
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
            with amp_ctx():
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


def staircase_diagnostic(model, loader, radius=1 / 255, n_points=40):
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    direction = torch.randn_like(x)
    flat_norm = direction.flatten(1).norm(dim=1).view(-1, *([1] * (x.dim() - 1)))
    direction = direction / flat_norm
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)

    with torch.no_grad(), amp_ctx():
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
# For torchao models: PTQ with weight-only config vs dynamic-activation config.
def run_quant_component_ablation(model_qat_instance, loader, name, eps=8 / 255):
    """
    Run component ablation for a torchao-based model.

    Uses the QuantModel to get different PTQ configs:
    - weight_only: Int4WeightOnlyConfig (weight-only int4)
    - act_only: approximate by evaluating with dynamic activation int8 PTQ
    - both: full dynamic activation int8 PTQ

    Args:
        model_qat_instance: The QuantModel instance (holds all variants).
        loader: DataLoader for evaluation.
        name: Model display name.
        eps: PGD epsilon.
    """
    fp32 = model_qat_instance.model

    configs = [
        ("weight_only", model_qat_instance.int4_PTQ),
        ("act_only", model_qat_instance.int8_PTQ),
        ("both", model_qat_instance.int8_PTQ),
    ]

    rows = []
    for label, qat_model in configs:
        clean_acc = sanity_check_accuracy(qat_model, loader)

        torch.manual_seed(0)
        pgd = torchattacks.PGD(qat_model, eps=eps, alpha=2 / 255, steps=20, random_start=True)
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with amp_ctx():
                x_adv = pgd(x, y)
                with torch.no_grad():
                    pred = qat_model(x_adv).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        pgd_acc = correct / total

        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        x_in = x.clone().requires_grad_(True)
        with amp_ctx():
            loss = F.cross_entropy(qat_model(x_in), y)
        g_hard = torch.autograd.grad(loss, x_in)[0].flatten()
        frac_zero = (g_hard.abs() < 1e-8).float().mean().item()

        rows.append({
            "model": name, "config": label,
            "clean_acc": clean_acc, "PGD_acc": pgd_acc,
            "frac_zero_grad_hard": frac_zero,
        })
    return rows


def parallelize(model):
    """Wrap model with DataParallel if multiple GPUs available."""
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        return nn.DataParallel(model)
    return model


def maybe_compile(model, name=""):
    """
    Wrap with torch.compile for faster CNN inference (item #13, typically
    10-40% faster once warmed up). This is best-effort: torch.compile can
    be fragile with custom/quantized ops (torchao) and with the backward
    hooks used by gradient_diagnostics_and_layerwise_profile, so a failed
    or unstable compile falls back to the eager model instead of crashing
    the whole run. Skipped on CPU, where it rarely helps and adds startup
    overhead.
    """
    if device != "cuda" or not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model)
    except Exception as e:
        print(f"  [WARN] torch.compile failed for {name}, using eager model: {e}")
        return model


# Run the full evaluation suite for a single (model, fp32_ref) pair.

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
            # Single pass now produces both the whole-input gradient stats
            # and the per-layer profile (previously two independent
            # forward+backward passes over the same batches).
            diag, layer_profile = gradient_diagnostics_and_layerwise_profile(
                model, loader, fp32_ref=fp32_ref, max_batches=5)
            results.update(diag)
            rows = [{"model": name, "layer": n, "grad_norm_hard": v, "grad_norm_ste": v}
                    for n, v in layer_profile.items()]
            pd.DataFrame(rows).to_csv(layerwise_csv_path(name), index=False)
        except Exception as e:
            print(f"  [WARN] gradient_diagnostics_and_layerwise_profile failed for {name}: {e}")

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

        # weight-only / activation-only / both ablation
        try:
            # model_qat_instance is the QuantModel that owns this model
            qat_instance = _get_qat_instance_for_model(model)
            if qat_instance is not None:
                rows = run_quant_component_ablation(qat_instance, loader, name, eps=eps)
                pd.DataFrame(rows).to_csv(component_ablation_csv_path(name), index=False)
        except Exception as e:
            print(f"  [WARN] run_quant_component_ablation failed for {name}: {e}")

    return results


# Resolve the QuantModel instance that owns a given sub-model.

def _get_qat_instance_for_model(target_model):
    """
    Find the QuantModel instance that owns *target_model*.
    We store the mapping in a global registry built during main().
    """
    return _model_to_qat_instance.get(id(target_model))


# Populated by main()
_model_to_qat_instance = {}


# Epsilon sweep — unchanged interface, only the model object differs

def run_pgd_epsilon_sweep_shared(model, loader, epsilons, alpha=2 / 255, steps=20):
    """
    Epsilon-projection PGD sweep (item #4). Instead of a full 20-step PGD
    run for every epsilon (5 epsilons x 20 steps = 100 PGD iterations per
    model), this runs ONE 20-step PGD attack at eps_max and, for every
    smaller epsilon in `epsilons`, projects the final perturbation onto
    that epsilon's L_inf ball. This is the amortized epsilon-sweep
    approximation used by several robustness libraries.

    CAVEAT: this is an approximation, not equivalent to independently
    running PGD from scratch at each epsilon. The trajectory's step
    directions and intermediate clipping are computed w.r.t. eps_max, so
    the projected point for a smaller epsilon is not necessarily where PGD
    would have ended up if optimized within that smaller ball from the
    start. In practice this tends to make the attack for eps < eps_max
    slightly weaker (reported accuracy slightly optimistic) than a
    from-scratch run. If you need exact per-epsilon PGD numbers (e.g. for
    a paper's headline result), fall back to running PGD independently per
    epsilon for that specific epsilon.
    """
    eps_max = max(epsilons)
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)

    correct_per_eps = {eps: 0 for eps in epsilons}
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps_max, eps_max)
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()

        for _ in range(steps):
            x_adv.requires_grad_(True)
            with amp_ctx():
                loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps_max), x + eps_max)
            x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()

        with torch.no_grad(), amp_ctx():
            for eps in epsilons:
                delta = torch.clamp(x_adv - x, -eps, eps)
                x_eps = torch.max(torch.min(x + delta, clip_max), clip_min)
                pred = model(x_eps).argmax(dim=1)
                correct_per_eps[eps] += (pred == y).sum().item()
        total += y.size(0)

    return {eps: correct_per_eps[eps] / total for eps in epsilons}


def run_epsilon_sweep_for_model(model, loader, name, epsilons):
    rows = []
    is_quant = count_quant_layers(model) > 0

    try:
        pgd_acc_by_eps = run_pgd_epsilon_sweep_shared(model, loader, epsilons)
    except Exception as e:
        print(f"  [WARN] shared-trajectory PGD sweep failed for {name}: {e}")
        pgd_acc_by_eps = {}

    for eps in epsilons:
        row = {"model": name, "epsilon": eps}
        row["PGD_acc"] = pgd_acc_by_eps.get(eps)

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


def _merge_worker_csvs(arch_keys, pattern, merged_path):
    """Merge per-arch-key CSVs written by dispatch_multi_gpu's workers into
    the shared results file, de-duplicating on (model[, epsilon])."""
    frames = []
    if os.path.exists(merged_path):
        frames.append(pd.read_csv(merged_path))
    for arch_key in arch_keys:
        p = os.path.join(DATA_DIR, pattern.format(arch_key))
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)
    dedup_cols = ["model", "epsilon"] if "epsilon" in merged.columns else ["model"]
    merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
    merged.to_csv(merged_path, index=False)


def dispatch_multi_gpu():
    """
    Multi-GPU parallel evaluation of independent models (item #15). Every
    model architecture is evaluated completely independently of the
    others, so rather than relying only on DataParallel (which splits ONE
    model's batch across GPUs, leaving GPUs idle while ResNet20 finishes
    before ResNet56 starts), this spins up one subprocess per architecture
    and pins each to its own GPU via CUDA_VISIBLE_DEVICES -- set in the
    child's environment *before* that child process even starts, so
    nothing in the rest of this file needs to change: each worker only
    ever sees one GPU, which it addresses as "cuda" exactly like the
    single-GPU code path already does.

    Workers write to per-arch-key CSVs (see the RESULTS_CSV/SWEEP_CSV
    override above); once all workers finish, results are merged back
    into the shared CSVs. Only used when >1 GPU is visible; a single-GPU
    or CPU machine just runs main() directly, unchanged.
    """
    import time

    n_gpus = torch.cuda.device_count()
    arch_keys = list(PRETRAINED_NAMES.keys())
    print(f"\n[dispatch] {n_gpus} GPU(s) visible, {len(arch_keys)} architectures -- "
          f"evaluating architectures in parallel, one process per GPU.")

    def launch(arch_key, gpu_id):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[dispatch] launching {arch_key} on GPU {gpu_id}")
        return subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--arch-key", arch_key],
            env=env,
        )

    pending = list(arch_keys)
    running = {}  # arch_key -> Popen
    next_gpu = 0
    failed = []

    while pending or running:
        while pending and len(running) < n_gpus:
            arch_key = pending.pop(0)
            running[arch_key] = launch(arch_key, next_gpu % n_gpus)
            next_gpu += 1
        for arch_key in list(running):
            ret = running[arch_key].poll()
            if ret is not None:
                if ret != 0:
                    print(f"[dispatch] [WARN] worker for {arch_key} exited with code {ret}")
                    failed.append(arch_key)
                del running[arch_key]
        if running:
            time.sleep(2)

    _merge_worker_csvs(arch_keys, "results_{}.csv", RESULTS_CSV)
    _merge_worker_csvs(arch_keys, "results_sweep_{}.csv", SWEEP_CSV)
    if failed:
        print(f"[dispatch] [WARN] the following architectures failed: {failed}. "
              f"Their per-arch CSVs (if any) were still merged; re-run with "
              f"--arch-key <name> to retry just that one.")
    print("[dispatch] all architectures complete, results merged into", RESULTS_CSV, "and", SWEEP_CSV)


# main() — rewritten to use quantize.Model interface

def main():
    finetune_loader, eval_loader = get_dataloaders()

    eval_batches = [(x.to(device), y.to(device)) for x, y in eval_loader]

    model_registry = {}  # {name: (model, fp32_ref)}
    all_qat_instances = {}  # arch_key -> QuantModel instance

    for arch_key in PRETRAINED_NAMES:
        print(f"\n>>> {arch_key} <<<")
        try:
            fp32 = load_pretrained(arch_key)
            acc = sanity_check_accuracy(fp32, eval_batches)
            print(f"  loaded pretrained {arch_key}, clean acc: {acc:.3f}")
            model_registry[f"{arch_key}_FP32"] = (fp32, None)
        except Exception as e:
            print(f"  [FAIL] could not load {arch_key}: {e}")
            traceback.print_exc()
            continue

        # Build the QuantModel wrapper this auto-constructs PTQ models
        try:
            qat_model = QuantModel(fp32)
            all_qat_instances[arch_key] = qat_model
        except Exception as e:
            print(f"  [FAIL] QuantModel wrapper for {arch_key}: {e}")
            traceback.print_exc()
            continue

        # PTQ int8 (auto-built in QuantModel.__init__)
        model_registry[f"{arch_key}_int8_PTQ"] = (qat_model.int8_PTQ, fp32)

        # PTQ int4 (auto-built in QuantModel.__init__)
        model_registry[f"{arch_key}_int4_PTQ"] = (qat_model.int4_PTQ, fp32)

        # QAT int8
        try:
            qat_model.train_qat(finetune_loader, epochs=3, bits=8)
            model_registry[f"{arch_key}_int8_QAT"] = (qat_model.int8_QAT, fp32)
        except Exception as e:
            print(f"  [FAIL] int8 QAT for {arch_key}: {e}")
            traceback.print_exc()

        # QAT int4
        try:
            qat_model.train_qat(finetune_loader, epochs=3, bits=4)
            model_registry[f"{arch_key}_int4_QAT"] = (qat_model.int4_QAT, fp32)
        except Exception as e:
            print(f"  [FAIL] int4 QAT for {arch_key}: {e}")
            traceback.print_exc()

    print("\nRegistry built:", list(model_registry.keys()))

    for k in model_registry:
        m, r = model_registry[k]
        m = maybe_compile(m, name=k)
        r = maybe_compile(r, name=f"{k}_ref") if r is not None else None
        model_registry[k] = (parallelize(m), parallelize(r) if r else None)

    suffixes = ["int8_PTQ", "int4_PTQ", "int8_QAT", "int4_QAT"]
    for name, (model, ref) in model_registry.items():
        for arch_key in PRETRAINED_NAMES:
            if any(name == f"{arch_key}_{suf}" for suf in suffixes) and arch_key in all_qat_instances:
                _model_to_qat_instance[id(model)] = all_qat_instances[arch_key]
                break

    if os.path.exists(RESULTS_CSV):
        df_results = pd.read_csv(RESULTS_CSV)
        done = set(df_results["model"].astype(str))
        results_rows = df_results.to_dict("records")
    else:
        df_results = pd.DataFrame(columns=["model"])
        done = set()
        results_rows = []

    for name, (model, ref) in list(model_registry.items()):
        if name in done:
            print(f"Skipping {name} (already in {RESULTS_CSV})")
            continue

        print(f"\nEvaluating {name} ...")
        try:
            res = run_suite(model, eval_batches, name, fp32_ref=ref)
        except Exception as e:
            print(f"  [FAIL] run_suite failed for {name}: {e}")
            traceback.print_exc()
            res = {"model": name}

        # Accumulate in a plain list and rebuild the DataFrame, rather than
        # pd.concat-ing a new one-row DataFrame onto df_results every
        # iteration (item #17). Still writes to CSV after every model so
        # progress/resumability is unaffected.
        results_rows.append(res)
        df_results = pd.DataFrame(results_rows)
        df_results.to_csv(RESULTS_CSV, index=False)

        print("Result:")
        print(pd.DataFrame([res]).to_string(index=False))
        print("-" * 100)

    print("\nFinal results:")
    print(df_results)

    acc_cols = [c for c in ["clean_acc", "FGSM", "PGD", "AutoAttack", "Transfer_from_FP32", "Random_Noise", "BPDA_PGD"]
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
        sweep_rows = df_sweep.to_dict("records")
    else:
        df_sweep = pd.DataFrame()
        sweep_done = set()
        sweep_rows = []

    for name, (model, ref) in model_registry.items():
        print(f"\nSweeping {name} ...")
        pending_eps = [eps for eps in SWEEP_EPSILONS if (name, round(eps, 6)) not in sweep_done]
        if not pending_eps:
            print(f"  Skipping {name} (already done)")
            continue
        try:
            rows = run_epsilon_sweep_for_model(model, eval_batches, name, pending_eps)
            if rows:
                sweep_rows.extend(rows)
                df_sweep = pd.DataFrame(sweep_rows)
                df_sweep.to_csv(SWEEP_CSV, index=False)
        except Exception as e:
            print(f"  [FAIL] epsilon sweep failed for {name}: {e}")
            traceback.print_exc()

    print("\nEpsilon sweep completed. Results saved to", SWEEP_CSV)
    print("All done.")


if __name__ == '__main__':
    if _ARGS.arch_key is None and torch.cuda.device_count() > 1:
        dispatch_multi_gpu()
    else:
        main()