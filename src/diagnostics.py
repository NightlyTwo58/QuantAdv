"""Gradient-masking diagnostics and quantization ablations."""
import json

import numpy as np
import torch
import torch.nn.functional as F
import torchattacks

from .config import *
from .attacks import (
    make_torchattack,
    accuracy_under_attack,
    denormalize_inputs,
    normalize_pixels,
    random_noise_attack,
)
from .evaluation import sanity_check_accuracy
from .quantization import (
    QuantConv2d,
    QuantLinear,
    convert_layer_chunk_to_quant,
    quant_layer_chunks,
    quantizable_layer_names,
    set_quant_components,
    set_ste_mode,
)


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

    try:
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
    finally:
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


def run_chunk_quantization_attacks(fp32_model, loader, name, bits=QAT_BITS, n_chunks=CHUNK_QUANT_NUM_CHUNKS, eps=DEFAULT_EPS):
    layer_names = quantizable_layer_names(fp32_model)
    chunks = quant_layer_chunks(layer_names, n_chunks)
    rows = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_model = convert_layer_chunk_to_quant(fp32_model, chunk, bits=bits, quant_weight=True, quant_act=True)
        row = {
            "model": name,
            "bits": bits,
            "chunk_id": i,
            "chunk_count": len(chunks),
            "chunk_label": f"{i}/{len(chunks)}",
            "chunk_size": len(chunk),
            "first_layer": chunk[0],
            "last_layer": chunk[-1],
            "layers": json.dumps(chunk),
        }
        try:
            row["clean_acc"] = sanity_check_accuracy(chunk_model, loader)
        except Exception as e:
            print(f"  [WARN] chunk clean_acc failed for {name} {row['chunk_label']}: {e}")
            row["clean_acc"] = None
        try:
            pgd = make_torchattack(torchattacks.PGD, chunk_model, eps=eps, alpha=PGD_ALPHA, steps=PGD_STEPS, random_start=PGD_RANDOM_START)
            row["PGD_acc"] = accuracy_under_attack(chunk_model, loader, pgd)
        except Exception as e:
            print(f"  [WARN] chunk PGD failed for {name} {row['chunk_label']}: {e}")
            row["PGD_acc"] = None
        rows.append(row)
        del chunk_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
