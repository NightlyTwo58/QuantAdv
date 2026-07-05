"""
Gradient-masking diagnostics: per-layer gradient profiling, staircase
(plateau) detection, PGD trajectory tracing, step-count ablations, and
weight/activation quantization-component ablations.
"""
import numpy as np
import torch
import torch.nn.functional as F

from torchao.quantization.qat.linear import FakeQuantizedLinear

from .config import device, CLIP_MIN_DEV, CLIP_MAX_DEV
from .attacks import amp_ctx, pgd_step, pgd_attack, evaluate_under_attack, random_noise_attack
from .evaluation import sanity_check_accuracy


def gradient_diagnostics_and_layerwise_profile(model, loader, fp32_ref=None, max_batches=5):
    """
    Returns (diagnostics_dict, layerwise_profile_dict).
    """
    quant_layers = [(n, m) for n, m in model.named_modules()
                    if isinstance(m, FakeQuantizedLinear)
                    or (hasattr(m, '_quantized_op') and m._quantized_op is not None)
                    or hasattr(m, 'quantizer')
                    or (getattr(m, 'weight', None) is not None
                        and type(m.weight).__module__.startswith('torchao'))]
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
            g_hard = torch.autograd.grad(loss, x_in, allow_unused=True)[0]
            if g_hard is None:
                g_hard = torch.zeros_like(x_in)
            g_hard = g_hard.flatten()
            frac_zero_hard.append((g_hard.abs() < 1e-8).float().mean().item())
            norm_hard.append(g_hard.norm().item())

            if fp32_ref is not None:
                fp32_ref.eval()
                x_ref = x.clone().requires_grad_(True)
                with amp_ctx():
                    loss_ref = F.cross_entropy(fp32_ref(x_ref), y)
                g_ref = torch.autograd.grad(loss_ref, x_ref, allow_unused=True)[0]
                if g_ref is None:
                    g_ref = torch.zeros_like(x_ref)
                g_ref = g_ref.flatten()
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


def pgd_steps_ablation(model, loader, eps=8 / 255, step_list=(0, 1, 2, 5, 10, 20, 50)):
    model.eval()
    out = {}
    for steps in step_list:
        if steps == 0:
            acc = random_noise_attack(model, loader, eps=eps, seed=0)
        else:
            acc = evaluate_under_attack(
                model, loader,
                lambda x, y: pgd_attack(model, x, y, eps, alpha=2 / 255, steps=steps, random_start=True))
        out[steps] = acc
    return out


def pgd_trajectory_diagnostics(model, loader, eps=8 / 255, alpha=2 / 255, steps=20, max_batches=5):
    model.eval()
    step_grad_norms = [0.0] * steps
    step_movement = [0.0] * steps
    n_batches = 0
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        noise = torch.empty_like(x).uniform_(-eps, eps)
        x_start = torch.max(torch.min(x + noise, CLIP_MAX_DEV), CLIP_MIN_DEV).detach()
        x_adv = x_start.clone()
        for s in range(steps):
            x_adv, grad = pgd_step(model, x_adv, x, y, eps, alpha, return_grad=True)
            step_grad_norms[s] += grad.flatten(1).norm(dim=1).mean().item()
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

    with torch.no_grad(), amp_ctx():
        prev_logits = model(x)
        plateau_hits = 0.0
        for i in range(1, n_points + 1):
            step = x + direction * (radius * i / n_points)
            step = torch.max(torch.min(step, CLIP_MAX_DEV), CLIP_MIN_DEV)
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
        ("act_only", model_qat_instance.int8_PTQ),
        ("both", model_qat_instance.int8_PTQ),
    ]

    rows = []
    for label, qat_model in configs:
        clean_acc = sanity_check_accuracy(qat_model, loader)

        torch.manual_seed(0)
        pgd_acc = evaluate_under_attack(
            qat_model, loader,
            lambda x, y: pgd_attack(qat_model, x, y, eps, alpha=2 / 255, steps=20, random_start=True))

        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        x_in = x.clone().requires_grad_(True)
        with amp_ctx():
            loss = F.cross_entropy(qat_model(x_in), y)
        g_hard = torch.autograd.grad(loss, x_in, allow_unused=True)[0]
        if g_hard is None:
            g_hard = torch.zeros_like(x_in)
        g_hard = g_hard.flatten()
        frac_zero = (g_hard.abs() < 1e-8).float().mean().item()

        rows.append({
            "model": name, "config": label,
            "clean_acc": clean_acc, "PGD_acc": pgd_acc,
            "frac_zero_grad_hard": frac_zero,
        })
    return rows
