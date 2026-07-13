#!/usr/bin/env python
# coding: utf-8
"""
All adversarial-attack logic
- generic PGD/attack helpers (make_torchattack, accuracy_from_adv_fn family)
- white-box attacks (FGSM/PGD, AutoAttack, CW/DeepFool/JSMA)
- transfer attacks (PGD/MIM transfer, UAP, surrogate/substitute training)
- black-box attacks (NES, Boundary attack)
- defense-adaptive attacks (BPDA, EOT-PGD, adaptive guardrail/detector attacks)
- diagnostics that are attack-driven (trajectory, margins, staircase, PGD-steps ablation)
"""

import random
import warnings
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
from autoattack import AutoAttack

import defense as dfn
from torch.amp import autocast
from config import *


def unwrap_model(model):
    """Return the underlying module when ``model`` is wrapped in DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


def set_ste_mode(model, flag):
    """Toggle straight-through gradients on fake-quantized modules."""
    toggled = 0
    for mod in model.modules():
        if hasattr(mod, "use_ste") and hasattr(mod, "bits"):
            mod.use_ste = flag
            toggled += 1
    return toggled


def get_ste_mode(model):
    """Return the current ``use_ste`` state of a model's quantized modules."""
    values = {
        mod.use_ste
        for mod in model.modules()
        if hasattr(mod, "use_ste") and hasattr(mod, "bits")
    }
    if not values:
        return None
    if len(values) > 1:
        return "mixed"
    return next(iter(values))


@contextmanager
def ste_mode(model, flag):
    """Run a block with an explicit, restored ``use_ste`` state.

    Every attack or diagnostic that cares whether gradients are computed
    through the true (masking) rounding op or through the straight-through
    bypass should wrap its gradient computation in this, e.g.::
        with ste_mode(model, False):   # real hard-round gradient
        with ste_mode(model, True):    # STE / BPDA-style bypass
    """
    previous = get_ste_mode(model)
    set_ste_mode(model, flag)
    try:
        yield
    finally:
        # Restore exactly what was there before (True/False), or leave
        # everything at False if the model had no quantized layers / no
        # prior state to speak of.
        set_ste_mode(model, previous if isinstance(previous, bool) else False)


def normalize_pixels(x):
    """Normalize pixel-space tensors with the configured dataset constants."""
    return (x - DATASET_MEAN.to(x.device)) / DATASET_STD.to(x.device)


def denormalize_inputs(x):
    """Map normalized dataset tensors back to pixel space."""
    return x * DATASET_STD.to(x.device) + DATASET_MEAN.to(x.device)


class PixelSpaceModel(nn.Module):
    """Adapter exposing a normalized-input model as a pixel-space model."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", DATASET_MEAN.clone())
        self.register_buffer("std", DATASET_STD.clone())

    def forward(self, x):
        return self.model((x - self.mean.to(x.device)) / self.std.to(x.device))


def make_torchattack(attack_cls, model, *args, **kwargs):
    """Create a torchattacks instance for normalized configured-dataset inputs."""
    attack = attack_cls(model, *args, **kwargs)
    attack.set_normalization_used(mean=DATASET_MEAN_VALUES, std=DATASET_STD_VALUES)
    return attack


def accuracy_from_adv_fn(
    model,
    loader,
    adv_fn=None,
    target_model=None,
    max_images=None,
    use_autocast=False,
    return_vector=False,
):
    """Measure accuracy after an optional adversarial-example function."""

    target = target_model if target_model is not None else model
    correct, total, n_seen = 0, 0, 0
    correct_vectors = []
    for x, y in loader:
        if max_images is not None:
            if n_seen >= max_images:
                break
            remaining = max_images - n_seen
            x, y = x[:remaining], y[:remaining]
        x, y = x.to(device, non_blocking=NON_BLOCKING_TRANSFER), y.to(
            device, non_blocking=NON_BLOCKING_TRANSFER
        )
        x_adv = adv_fn(x, y) if adv_fn is not None else x
        with torch.no_grad():
            if use_autocast:
                with autocast(device_type=device.type):
                    pred = target(x_adv).argmax(dim=1)
            else:
                pred = target(x_adv).argmax(dim=1)
        batch_correct = pred == y
        correct += batch_correct.sum().item()
        if return_vector:
            correct_vectors.append(batch_correct.detach().cpu())
        total += y.size(0)
        n_seen += y.size(0)
    accuracy = correct / total if total else None
    if return_vector:
        vector = (
            torch.cat(correct_vectors).numpy().astype(bool)
            if correct_vectors
            else np.empty(0, dtype=bool)
        )
        return accuracy, vector
    return accuracy


def accuracy_under_attack(
    model, loader, attack, target_model=None, max_images=None, return_vector=False
):
    """Measure accuracy under a torchattacks-compatible attack object."""

    def adv_fn(x, y):
        return attack(x, y)

    return accuracy_from_adv_fn(
        model,
        loader,
        adv_fn,
        target_model=target_model,
        max_images=max_images,
        return_vector=return_vector,
    )


def evaluate_normalized_attack(model, loader, attack_fn):
    """Evaluate an attack function that consumes and returns normalized tensors."""
    model.eval()
    return accuracy_from_adv_fn(model, loader, attack_fn)


def seed_averaged_metrics(name, seeds, fn):
    """Run a metric for multiple seeds and return mean/std summaries."""
    accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        accs.append(fn(seed))
    mean = float(np.mean(accs))
    return {
        name: mean,
        f"{name}_mean": mean,
        f"{name}_std": float(np.std(accs)),
    }


def _seed_restart(seed):
    """Seed every RNG used by an explicitly identified attack restart."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _aggregate_restarts(name, accuracies, vectors, return_vector=False):
    """Aggregate seeded restarts by per-example worst-case correctness."""
    if not vectors:
        raise ValueError(f"{name} requires at least one restart seed")
    worst_vector = np.logical_and.reduce(vectors)
    out = {
        name: float(worst_vector.mean()),
        f"{name}_mean": float(np.mean(accuracies)),
        f"{name}_std": float(np.std(accuracies)),
    }
    if return_vector:
        out["_vectors"] = {name: worst_vector}
    return out


def _run_seeded_pgd(
    model,
    loader,
    *,
    eps,
    alpha,
    steps,
    random_start,
    seeds,
    use_ste,
    return_vector,
):
    """Single implementation for seeded torchattacks PGD evaluation."""
    model.eval()
    accuracies, vectors = [], []
    with ste_mode(model, use_ste):
        for seed in seeds:
            _seed_restart(seed)
            attack = make_torchattack(
                torchattacks.PGD,
                model,
                eps=eps,
                alpha=alpha,
                steps=steps,
                random_start=random_start,
            )
            accuracy, vector = accuracy_under_attack(
                model, loader, attack, return_vector=True
            )
            accuracies.append(accuracy)
            vectors.append(vector)
    return _aggregate_restarts("PGD", accuracies, vectors, return_vector)


def run_pgd(
    model,
    loader,
    eps=DEFAULT_EPS,
    alpha=PGD_ALPHA,
    steps=PGD_STEPS,
    random_start=PGD_RANDOM_START,
    seeds=SEEDS,
    return_vector=False,
):
    """Evaluate hard-round PGD with the canonical seeded restart policy.

    Each seed is one restart. ``PGD`` is the intersection of per-example
    correctness across restarts; ``PGD_mean`` and ``PGD_std`` summarize the
    individual restart accuracies.
    """
    return _run_seeded_pgd(
        model,
        loader,
        eps=eps,
        alpha=alpha,
        steps=steps,
        random_start=random_start,
        seeds=seeds,
        use_ste=False,
        return_vector=return_vector,
    )


def run_fgsm_pgd(
    model, loader, eps=DEFAULT_EPS, seeds=SEEDS, return_vectors=False, use_ste=False
):
    """Run FGSM and multi-restart PGD, aggregating PGD by worst-case correctness.

    By default this attacks the actual hard-round quantized model, matching
    ``pgd_steps_ablation``. Set ``use_ste=True`` only for an explicitly
    STE-based experiment; BPDA results are reported separately by ``run_bpda``.
    """
    model.eval()
    with ste_mode(model, use_ste):
        fgsm = make_torchattack(torchattacks.FGSM, model, eps=eps)
        out = {}

        def attack_vector(attack):
            def adv_fn(x, y):
                return attack(x, y)

            return accuracy_from_adv_fn(model, loader, adv_fn, return_vector=True)

        fgsm_acc, fgsm_vector = attack_vector(fgsm)
        out["FGSM"] = fgsm_acc
        pgd_out = (
            run_pgd(
                model, loader, eps=eps, seeds=seeds, return_vector=return_vectors
            )
            if not use_ste
            else _run_seeded_pgd(
                model,
                loader,
                eps=eps,
                alpha=PGD_ALPHA,
                steps=PGD_STEPS,
                random_start=PGD_RANDOM_START,
                seeds=seeds,
                use_ste=True,
                return_vector=return_vectors,
            )
        )
        pgd_vectors = pgd_out.pop("_vectors", {})
        out.update(pgd_out)
        if return_vectors:
            out["_vectors"] = {"FGSM": fgsm_vector, **pgd_vectors}
        return out


def run_autoattack(model, loader, eps=DEFAULT_EPS, return_vector=False, use_ste=False):
    """Run AutoAttack in pixel space while reporting normalized-model accuracy."""
    model.eval()
    with ste_mode(model, use_ste):
        pixel_model = PixelSpaceModel(model).to(device).eval()
        adversary = AutoAttack(
            pixel_model,
            norm=AUTOATTACK_NORM,
            eps=eps,
            version=AUTOATTACK_VERSION,
            device=device,
            verbose=AUTOATTACK_VERBOSE,
        )
        adversary.seed = AUTOATTACK_SEED

        def adv_fn(x, y):
            x_pixels = denormalize_inputs(x).clamp(0.0, 1.0)
            return normalize_pixels(
                adversary.run_standard_evaluation(x_pixels, y, bs=x.size(0))
            )

        return accuracy_from_adv_fn(model, loader, adv_fn, return_vector=return_vector)


def run_extra_whitebox_attacks(
    model,
    loader,
    eps=DEFAULT_EPS,
    jsma_max_images=JSMA_MAX_IMAGES,
    use_ste=False,
):
    model.eval()
    with ste_mode(model, use_ste):
        out = {}
        cw = make_torchattack(
            torchattacks.CW, model, c=CW_C, kappa=CW_KAPPA, steps=CW_STEPS, lr=CW_LR
        )
        out["CW"] = accuracy_under_attack(model, loader, cw)
        deepfool = make_torchattack(
            torchattacks.DeepFool,
            model,
            steps=DEEPFOOL_STEPS,
            overshoot=DEEPFOOL_OVERSHOOT,
        )
        out["DeepFool"] = accuracy_under_attack(model, loader, deepfool)
        jsma = make_torchattack(
            torchattacks.JSMA, model, theta=JSMA_THETA, gamma=JSMA_GAMMA
        )
        out["JSMA"] = accuracy_under_attack(
            model, loader, jsma, max_images=jsma_max_images
        )
        return out


def transfer_attack(
    source_model,
    target_model,
    loader,
    eps=DEFAULT_EPS,
    return_vector=False,
    use_ste=False,
):
    """Generate PGD examples on ``source_model`` and evaluate ``target_model``."""
    with ste_mode(source_model, use_ste):
        pgd = make_torchattack(
            torchattacks.PGD,
            source_model,
            eps=eps,
            alpha=PGD_ALPHA,
            steps=PGD_STEPS,
            random_start=PGD_RANDOM_START,
        )
        return accuracy_under_attack(
            source_model,
            loader,
            pgd,
            target_model=target_model,
            return_vector=return_vector,
        )


def transfer_attack_mim(
    source_model,
    target_model,
    loader,
    eps=DEFAULT_EPS,
    return_vector=False,
    use_ste=False,
):
    with ste_mode(source_model, use_ste):
        mim = make_torchattack(
            torchattacks.MIFGSM,
            source_model,
            eps=eps,
            alpha=PGD_ALPHA,
            steps=PGD_STEPS,
            decay=MIFGSM_DECAY,
        )
        return accuracy_under_attack(
            source_model,
            loader,
            mim,
            target_model=target_model,
            return_vector=return_vector,
        )


def build_uap(
    model,
    loader,
    eps=DEFAULT_EPS,
    delta=UAP_DELTA,
    max_iter=UAP_MAX_ITER,
    deepfool_steps=UAP_DEEPFOOL_STEPS,
    overshoot=UAP_OVERSHOOT,
    max_images=UAP_MAX_IMAGES,
):
    model.eval()
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)
    sample_x, _ = next(iter(loader))
    v = torch.zeros(1, *sample_x.shape[1:], device=device)
    deepfool = make_torchattack(
        torchattacks.DeepFool, model, steps=deepfool_steps, overshoot=overshoot
    )
    fooling_rate, it = 0.0, 0
    while fooling_rate < (1 - delta) and it < max_iter:
        n_seen, n_fooled = 0, 0
        for x, y in loader:
            if n_seen >= max_images:
                break
            x, y = x.to(device, non_blocking=NON_BLOCKING_TRANSFER), y.to(
                device, non_blocking=NON_BLOCKING_TRANSFER
            )
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


def _run_uap_attack(
    source_model, target_model, loader, eps=DEFAULT_EPS, max_images=UAP_MAX_IMAGES
):
    v = build_uap(source_model, loader, eps=eps, max_images=max_images)
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)

    def adv_fn(x, y):
        return torch.max(torch.min(x + v, clip_max), clip_min)

    return accuracy_from_adv_fn(source_model, loader, adv_fn, target_model=target_model)


def run_uap_attack(model, loader, eps=DEFAULT_EPS, max_images=UAP_MAX_IMAGES):
    return _run_uap_attack(model, model, loader, eps=eps, max_images=max_images)


def transfer_uap_attack(
    source_model, target_model, loader, eps=DEFAULT_EPS, max_images=UAP_MAX_IMAGES
):
    return _run_uap_attack(
        source_model, target_model, loader, eps=eps, max_images=max_images
    )


def projected_pgd_attack(x, y, eps, alpha, steps, grad_fn):
    """PGD over normalized tensors with an externally supplied gradient function."""
    clip_min = CLIP_MIN.to(device)
    clip_max = CLIP_MAX.to(device)
    eps_normalized = eps / DATASET_STD.to(x.device)
    alpha_normalized = alpha / DATASET_STD.to(x.device)
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-1, 1) * eps_normalized
    x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    for _ in range(steps):
        grad = grad_fn(x_adv, y)
        grad = torch.zeros_like(x_adv) if grad is None else grad
        x_adv = x_adv.detach() + alpha_normalized * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps_normalized), x + eps_normalized)
        x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
    return x_adv.detach()


def bpda_pgd_attack(
    model, x, y, eps=DEFAULT_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS, backward_model=None
):
    """Run PGD with STE/BPDA gradients through non-differentiable defenses."""
    backward_model = backward_model if backward_model is not None else model

    def grad_fn(x_adv, labels):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(backward_model(x_adv), labels)
        grad = torch.autograd.grad(loss, x_adv, allow_unused=True)[0]
        return grad

    with ste_mode(model, True):
        if backward_model is not model:
            with ste_mode(backward_model, True):
                return projected_pgd_attack(x, y, eps, alpha, steps, grad_fn)
        return projected_pgd_attack(x, y, eps, alpha, steps, grad_fn)


def _run_bpda_once(
    model,
    loader,
    eps,
    n_restarts,
    alpha=PGD_ALPHA,
    steps=PGD_STEPS,
    return_vector=False,
):
    correct_masks = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=NON_BLOCKING_TRANSFER), y.to(
            device, non_blocking=NON_BLOCKING_TRANSFER
        )
        worst_correct = torch.ones(y.size(0), dtype=torch.bool, device=device)
        for _ in range(n_restarts):
            x_adv = bpda_pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
            worst_correct &= pred == y
        correct_masks.append(worst_correct)
    all_correct = torch.cat(correct_masks)
    accuracy = all_correct.float().mean().item()
    return (
        (accuracy, all_correct.cpu().numpy().astype(bool))
        if return_vector
        else accuracy
    )


def run_bpda(
    model,
    loader,
    eps=DEFAULT_EPS,
    n_restarts=BPDA_RESTARTS_DEFAULT,
    seeds=SEEDS,
    return_vector=False,
    alpha=PGD_ALPHA,
    steps=PGD_STEPS,
):
    """Evaluate BPDA-PGD with seed-level worst-case correctness aggregation."""
    vectors, accuracies = [], []
    for seed in seeds:
        _seed_restart(seed)
        accuracy, vector = _run_bpda_once(
            model,
            loader,
            eps,
            n_restarts,
            alpha=alpha,
            steps=steps,
            return_vector=True,
        )
        accuracies.append(accuracy)
        vectors.append(vector)
    return _aggregate_restarts("BPDA_PGD", accuracies, vectors, return_vector)


def adaptive_pgd_attack(
    model,
    x,
    y,
    loss_fn,
    eps=DEFAULT_EPS,
    alpha=PGD_ALPHA,
    steps=PGD_STEPS,
    use_ste=False,
):
    def grad_fn(x_adv, labels):
        x_adv.requires_grad_(True)
        loss = loss_fn(x_adv, labels)
        grad = torch.autograd.grad(loss, x_adv, allow_unused=True)[0]
        return grad

    with ste_mode(model, use_ste):
        return projected_pgd_attack(x, y, eps, alpha, steps, grad_fn)


def run_sanitized_bpda(model, loader, eps=DEFAULT_EPS):
    defense_model = unwrap_model(model)
    backward_model = defense_model.model if hasattr(defense_model, "model") else model
    acc = evaluate_normalized_attack(
        model,
        loader,
        lambda x, y: bpda_pgd_attack(
            model, x, y, eps=eps, backward_model=backward_model
        ),
    )
    return {"BPDA_Adaptive": acc}


def run_eot_pgd(model, loader, eps=DEFAULT_EPS, eot_samples=ADAPTIVE_EOT_SAMPLES):
    defense_model = unwrap_model(model)

    def attack(x, y):
        def loss_fn(x_adv, labels):
            loss = 0.0
            base_model = (
                defense_model.model if hasattr(defense_model, "model") else model
            )
            sigma = getattr(defense_model, "sigma", 0.0)
            clip_min = CLIP_MIN.to(x_adv.device)
            clip_max = CLIP_MAX.to(x_adv.device)
            for _ in range(eot_samples):
                noisy = (x_adv + torch.randn_like(x_adv) * sigma).clamp(
                    clip_min, clip_max
                )
                loss = loss + F.cross_entropy(base_model(noisy), labels)
            return loss / eot_samples

        return adaptive_pgd_attack(model, x, y, loss_fn, eps=eps)

    return {"EOT_PGD": evaluate_normalized_attack(model, loader, attack)}


def run_adaptive_guardrail(
    model, loader, eps=DEFAULT_EPS, lam=ADAPTIVE_GUARDRAIL_LAMBDA
):
    defense_model = unwrap_model(model)

    def attack(x, y):
        def loss_fn(x_adv, labels):
            logits = defense_model.model(x_adv)
            conf = F.softmax(logits, dim=1).max(dim=1).values
            threshold = getattr(defense_model, "conf_threshold", 0.55)
            guardrail_penalty = F.softplus(
                ADAPTIVE_GUARDRAIL_SCALE * (threshold - conf)
            ).mean()
            return F.cross_entropy(logits, labels) - lam * guardrail_penalty

        return adaptive_pgd_attack(model, x, y, loss_fn, eps=eps)

    return {"Adaptive_Guardrail": evaluate_normalized_attack(model, loader, attack)}


def run_adaptive_detect_guard(
    model, loader, eps=DEFAULT_EPS, lam=ADAPTIVE_DETECTOR_LAMBDA
):
    defense_model = unwrap_model(model)

    def attack(x, y):
        def loss_fn(x_adv, labels):
            logits = defense_model.model(x_adv)
            benign = torch.zeros(labels.size(0), dtype=torch.long, device=labels.device)
            detector_penalty = F.cross_entropy(defense_model.detector(x_adv), benign)
            return F.cross_entropy(logits, labels) - lam * detector_penalty

        return adaptive_pgd_attack(model, x, y, loss_fn, eps=eps)

    return {"Adaptive_DetectGuard": evaluate_normalized_attack(model, loader, attack)}


def run_defense_adaptive_attacks(model, loader, eps=DEFAULT_EPS):
    defense_model = unwrap_model(model)
    if isinstance(defense_model, dfn.SanitizedModel):
        return run_sanitized_bpda(model, loader, eps=eps)
    if isinstance(defense_model, dfn.SmoothedModel):
        return run_eot_pgd(model, loader, eps=eps)
    if isinstance(defense_model, dfn.GuardrailModel):
        return run_adaptive_guardrail(model, loader, eps=eps)
    if isinstance(defense_model, dfn.DetectGuardModel):
        return run_adaptive_detect_guard(model, loader, eps=eps)
    return {}


def nes_estimate_gradient(
    model,
    x,
    y,
    n_samples=NES_SAMPLES_DEFAULT,
    sigma=NES_SIGMA,
    query_chunk=NES_QUERY_CHUNK,
):
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
            loss_plus = F.cross_entropy(model(x_plus), y_rep, reduction="none").view(
                chunk, B
            )
            loss_minus = F.cross_entropy(model(x_minus), y_rep, reduction="none").view(
                chunk, B
            )
        weight = (loss_plus - loss_minus).view(chunk, B, 1, 1, 1)
        grad_acc += (weight * u).sum(dim=0)
        remaining -= chunk
    return grad_acc / (2 * n_pairs * sigma)


def nes_pgd_attack(
    model,
    x,
    y,
    eps=DEFAULT_EPS,
    alpha=PGD_ALPHA,
    steps=NES_STEPS,
    n_samples=NES_SAMPLES_DEFAULT,
    sigma=NES_SIGMA,
    query_chunk=NES_QUERY_CHUNK,
):
    def grad_fn(x_adv, labels):
        return nes_estimate_gradient(
            model,
            x_adv,
            labels,
            n_samples=n_samples,
            sigma=sigma,
            query_chunk=query_chunk,
        )

    return projected_pgd_attack(x, y, eps, alpha, steps, grad_fn)


def nes_attack(
    model,
    loader,
    eps=DEFAULT_EPS,
    n_samples=NES_SAMPLES_DEFAULT,
    sigma=NES_SIGMA,
    alpha=PGD_ALPHA,
    steps=NES_STEPS,
    seed=None,
    query_chunk=NES_QUERY_CHUNK,
):
    """Single-seed NES attack accuracy over the whole loader."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()

    def adv_fn(x, y):
        return nes_pgd_attack(
            model,
            x,
            y,
            eps=eps,
            alpha=alpha,
            steps=steps,
            n_samples=n_samples,
            sigma=sigma,
            query_chunk=query_chunk,
        )

    return accuracy_from_adv_fn(model, loader, adv_fn)


def run_nes_attack(model, loader, eps=DEFAULT_EPS, seeds=SEEDS, **kwargs):
    return seed_averaged_metrics(
        "NES",
        seeds,
        lambda seed: nes_attack(model, loader, eps=eps, seed=seed, **kwargs),
    )


class SubstituteCNN(nn.Module):
    def __init__(self, num_classes=SUBSTITUTE_NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                DATASET_INPUT_CHANNELS,
                SUBSTITUTE_CONV1_CHANNELS,
                SUBSTITUTE_KERNEL_SIZE,
                padding=SUBSTITUTE_CONV_PADDING,
            ),
            nn.ReLU(inplace=SUBSTITUTE_RELU_INPLACE),
            nn.Conv2d(
                SUBSTITUTE_CONV1_CHANNELS,
                SUBSTITUTE_CONV1_CHANNELS,
                SUBSTITUTE_KERNEL_SIZE,
                padding=SUBSTITUTE_CONV_PADDING,
            ),
            nn.ReLU(inplace=SUBSTITUTE_RELU_INPLACE),
            nn.MaxPool2d(SUBSTITUTE_POOL_KERNEL),
            nn.Conv2d(
                SUBSTITUTE_CONV1_CHANNELS,
                SUBSTITUTE_CONV2_CHANNELS,
                SUBSTITUTE_KERNEL_SIZE,
                padding=SUBSTITUTE_CONV_PADDING,
            ),
            nn.ReLU(inplace=SUBSTITUTE_RELU_INPLACE),
            nn.Conv2d(
                SUBSTITUTE_CONV2_CHANNELS,
                SUBSTITUTE_CONV2_CHANNELS,
                SUBSTITUTE_KERNEL_SIZE,
                padding=SUBSTITUTE_CONV_PADDING,
            ),
            nn.ReLU(inplace=SUBSTITUTE_RELU_INPLACE),
            nn.MaxPool2d(SUBSTITUTE_POOL_KERNEL),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                SUBSTITUTE_CONV2_CHANNELS
                * SUBSTITUTE_LINEAR_FEATURE_MAP
                * SUBSTITUTE_LINEAR_FEATURE_MAP,
                SUBSTITUTE_HIDDEN_DIM,
            ),
            nn.ReLU(inplace=SUBSTITUTE_RELU_INPLACE),
            nn.Linear(SUBSTITUTE_HIDDEN_DIM, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_substitute(
    target_model,
    seed_x,
    rounds=SUBSTITUTE_ROUNDS,
    epochs_per_round=SUBSTITUTE_EPOCHS_PER_ROUND,
    lr=SUBSTITUTE_LR,
    lam=SUBSTITUTE_LAMBDA,
    batch_size=SUBSTITUTE_BATCH_SIZE,
):
    target_model.eval()
    substitute = SubstituteCNN().to(device)
    opt = torch.optim.Adam(substitute.parameters(), lr=lr)
    x = seed_x.clone().to(device)
    for r in range(rounds):
        with torch.no_grad():
            y = target_model(x).argmax(dim=1)
        substitute.train()
        ds = torch.utils.data.TensorDataset(x, y)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=TRAIN_SHUFFLE
        )
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


def run_surrogate_attack(
    model, loader, eps=DEFAULT_EPS, seed_n=SURROGATE_SEED_N, rounds=SUBSTITUTE_ROUNDS
):
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


def boundary_attack_single(
    model,
    x_orig,
    y_true,
    clip_min,
    clip_max,
    steps=BOUNDARY_STEPS_DEFAULT,
    spherical_step=BOUNDARY_SPHERICAL_STEP,
    source_step=BOUNDARY_SOURCE_STEP,
    step_adapt=BOUNDARY_STEP_ADAPT,
    init_tries=BOUNDARY_INIT_TRIES,
    init_chunk=BOUNDARY_INIT_CHUNK,
):
    """
    Decision-based Boundary Attack (Brendel, Bethge, 2018) for a src
    already-correctly-classified image.
    """
    clip_min = clip_min.to(x_orig.device)
    clip_max = clip_max.to(x_orig.device)
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
        return x_orig.clone(), False
    sph_step, src_step = spherical_step, source_step
    sph_hist, src_hist = [], []
    for i in range(steps):
        diff = x_orig - x_adv
        dist = diff.norm()
        if dist.item() < BOUNDARY_MIN_DIST:
            break
        # random move orthogonal to the direction toward x_orig, same radius
        perturb = torch.randn_like(x_adv)
        perturb = perturb - (perturb * diff).sum() / (dist**2) * diff
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
                sph_step *= (
                    step_adapt
                    if rate > BOUNDARY_SPH_SUCCESS_HIGH
                    else (1 / step_adapt if rate < BOUNDARY_SPH_SUCCESS_LOW else 1.0)
                )
            if src_hist:
                rate = np.mean(src_hist[-10:])
                src_step *= (
                    step_adapt
                    if rate > BOUNDARY_SPH_SUCCESS_HIGH
                    else (1 / step_adapt if rate < BOUNDARY_SPH_SUCCESS_LOW else 1.0)
                )
    return x_adv.detach(), True


def run_boundary_attack(
    model,
    loader,
    eps=DEFAULT_EPS,
    max_images=BOUNDARY_MAX_IMAGES_DEFAULT,
    steps=BOUNDARY_STEPS_DEFAULT,
    seed=BOUNDARY_SEED,
):
    """
    Runs the Boundary Attack on up to `max_images` loader examples and reports
    estimated robust accuracy over that same subset.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    clip_min, clip_max = CLIP_MIN.squeeze(0).to(device), CLIP_MAX.squeeze(0).to(device)
    dists = []
    total_seen = 0
    clean_correct = 0
    robust_correct = 0
    init_failed = 0  # random search never found a misclassified starting point
    for x, y in loader:
        if total_seen >= max_images:
            break
        x, y = x.to(device, non_blocking=NON_BLOCKING_TRANSFER), y.to(
            device, non_blocking=NON_BLOCKING_TRANSFER
        )
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
        for i in range(x.size(0)):
            if total_seen >= max_images:
                break
            total_seen += 1
            if (pred[i] != y[i]).item():
                continue
            clean_correct += 1
            x_adv, init_ok = boundary_attack_single(
                model, x[i], y[i], clip_min, clip_max, steps=steps
            )
            if not init_ok:
                init_failed += 1
                continue
            dist = (
                (
                    denormalize_inputs(x_adv.unsqueeze(0))
                    - denormalize_inputs(x[i].unsqueeze(0))
                )
                .abs()
                .max()
                .item()
            )
            adv_found = (_predict_batch(model, x_adv.unsqueeze(0))[0] != y[i]).item()
            if adv_found:
                dists.append(dist)
            if (not adv_found) or dist > eps:
                robust_correct += 1
    if total_seen == 0:
        return {
            "Boundary_acc": None,
            "Boundary_mean_Linf": None,
            "Boundary_median_Linf": None,
            "Boundary_min_Linf": None,
            "Boundary_max_Linf": None,
            "Boundary_std_Linf": None,
            "Boundary_n": 0,
            "Boundary_init_failed": 0,
            "Boundary_init_failed_rate": None,
        }
    evaluated = clean_correct - init_failed
    dists = np.array(dists, dtype=float)
    boundary_acc = (robust_correct / evaluated) if evaluated > 0 else None
    clean_subset_acc = clean_correct / total_seen
    if boundary_acc is not None and boundary_acc > clean_subset_acc + 1e-12:
        warnings.warn(
            "Boundary_acc exceeded clean accuracy on the evaluated subset; the old subset-only metric could do this because it divided only by clean-correct attacked samples.",
            RuntimeWarning,
        )
    init_failed_rate = (init_failed / clean_correct) if clean_correct > 0 else None
    if (
        init_failed_rate is not None and init_failed_rate > 0.2
    ):  # >20% failed inits -> Boundary_acc is unreliable
        warnings.warn(
            f"Boundary attack init search failed on {init_failed_rate:.1%} of clean-correct samples "
            f"({init_failed}/{clean_correct}); Boundary_acc is computed only over the remaining "
            f"{evaluated} samples and may not be a reliable robustness estimate.",
            RuntimeWarning,
        )
    return {
        "Boundary_acc": float(boundary_acc) if boundary_acc is not None else None,
        "Boundary_mean_Linf": float(dists.mean()) if dists.size else None,
        "Boundary_median_Linf": float(np.median(dists)) if dists.size else None,
        "Boundary_min_Linf": float(dists.min()) if dists.size else None,
        "Boundary_max_Linf": float(dists.max()) if dists.size else None,
        "Boundary_std_Linf": float(dists.std()) if dists.size else None,
        "Boundary_n": int(evaluated),
        "Boundary_init_failed": int(init_failed),
        "Boundary_init_failed_rate": (
            float(init_failed_rate) if init_failed_rate is not None else None
        ),
    }


def random_noise_attack(
    model, loader, eps=DEFAULT_EPS, n_restarts=1, seed=None, return_vector=False
):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)
    eps_normalized = eps / DATASET_STD.to(device)
    correct, total, vectors = 0, 0, []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=NON_BLOCKING_TRANSFER), y.to(
                device, non_blocking=NON_BLOCKING_TRANSFER
            )
            worst_correct = torch.ones(y.size(0), dtype=torch.bool, device=device)
            for _ in range(n_restarts):
                noise = torch.empty_like(x).uniform_(-1, 1) * eps_normalized
                x_adv = torch.max(torch.min(x + noise, clip_max), clip_min)
                pred = model(x_adv).argmax(dim=1)
                worst_correct &= pred == y
            correct += worst_correct.sum().item()
            total += y.size(0)
            if return_vector:
                vectors.append(worst_correct.cpu())
    accuracy = correct / total
    if return_vector:
        return accuracy, torch.cat(vectors).numpy().astype(bool)
    return accuracy


def run_random_noise_seeded(
    model, loader, eps=DEFAULT_EPS, seeds=SEEDS, return_vector=False
):
    accuracies, vectors = [], []
    for seed in seeds:
        _seed_restart(seed)
        accuracy, vector = random_noise_attack(
            model, loader, eps=eps, seed=seed, return_vector=True
        )
        accuracies.append(accuracy)
        vectors.append(vector)
    return _aggregate_restarts("Random_Noise", accuracies, vectors, return_vector)


def pgd_steps_ablation(
    model, loader, eps=DEFAULT_EPS, step_list=PGD_ABLATION_STEPS, use_ste=False
):
    """Accuracy vs. PGD step count, at an explicit, chosen gradient regime."""
    model.eval()
    out = {}
    for steps in step_list:
        if not use_ste:
            result = run_pgd(model, loader, eps=eps, steps=steps, seeds=SEEDS)
            out[steps] = result["PGD"]
        else:
            result = run_bpda(
                model,
                loader,
                eps=eps,
                n_restarts=1,
                seeds=SEEDS,
                steps=steps,
            )
            out[steps] = result["BPDA_PGD"]
    return out


def pgd_trajectory_diagnostics(
    model,
    loader,
    eps=DEFAULT_EPS,
    alpha=PGD_ALPHA,
    steps=PGD_STEPS,
    max_batches=TRAJECTORY_MAX_BATCHES,
    use_ste=False,
):
    """Per-step gradient norm and movement along a PGD trajectory."""
    model.eval()
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)
    eps_normalized = eps / DATASET_STD.to(device)
    alpha_normalized = alpha / DATASET_STD.to(device)
    step_grad_norms = [0.0] * steps
    step_movement = [0.0] * steps
    n_batches = 0
    with ste_mode(model, use_ste):
        for bi, (x, y) in enumerate(loader):
            if bi >= max_batches:
                break
            x, y = x.to(device, non_blocking=NON_BLOCKING_TRANSFER), y.to(
                device, non_blocking=NON_BLOCKING_TRANSFER
            )
            noise = torch.empty_like(x).uniform_(-1, 1) * eps_normalized
            x_start = torch.max(torch.min(x + noise, clip_max), clip_min).detach()
            x_adv = x_start.clone()
            for s in range(steps):
                x_adv.requires_grad_(True)
                loss = F.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss, x_adv)[0]
                step_grad_norms[s] += grad.flatten(1).norm(dim=1).mean().item()
                x_adv = x_adv.detach() + alpha_normalized * grad.sign()
                x_adv = torch.min(
                    torch.max(x_adv, x - eps_normalized), x + eps_normalized
                )
                x_adv = torch.max(torch.min(x_adv, clip_max), clip_min).detach()
                step_movement[s] += (
                    (x_adv - x_start).flatten(1).abs().max(dim=1).values.mean().item()
                )
            n_batches += 1
    return {
        "grad_norm_per_step": [g / n_batches for g in step_grad_norms],
        "movement_from_random_start_per_step": [m / n_batches for m in step_movement],
    }


def staircase_diagnostic(
    model, loader, radius=STAIRCASE_RADIUS, n_points=STAIRCASE_N_POINTS
):
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    direction = torch.randn_like(x)
    flat_norm = direction.flatten(1).norm(dim=1).view(-1, *([1] * (x.dim() - 1)))
    direction = direction / flat_norm
    clip_min, clip_max = CLIP_MIN.to(device), CLIP_MAX.to(device)
    radius_normalized = radius / DATASET_STD.to(device)
    with torch.no_grad():
        prev_logits = model(x)
        plateau_hits = 0.0
        for i in range(1, n_points + 1):
            step = x + direction * radius_normalized * (i / n_points)
            step = torch.max(torch.min(step, clip_max), clip_min)
            logits = model(step)
            plateau_hits += (logits == prev_logits).all(dim=1).float().mean().item()
            prev_logits = logits
    return {"plateau_fraction": plateau_hits / n_points}


def confidence_margin_diagnostic(
    model,
    loader,
    eps=DEFAULT_EPS,
    steps=MARGIN_STEPS,
    max_batches=MARGIN_MAX_BATCHES,
    use_ste=False,
):
    """Compare top-2 softmax margins clean vs. under PGD, at an explicit regime."""
    model.eval()
    with ste_mode(model, use_ste):
        pgd = make_torchattack(
            torchattacks.PGD,
            model,
            eps=eps,
            alpha=PGD_ALPHA,
            steps=steps,
            random_start=PGD_RANDOM_START,
        )
        clean_margins, adv_margins = [], []
        for bi, (x, y) in enumerate(loader):
            if bi >= max_batches:
                break
            x, y = x.to(device, non_blocking=NON_BLOCKING_TRANSFER), y.to(
                device, non_blocking=NON_BLOCKING_TRANSFER
            )
            with torch.no_grad():
                top2 = F.softmax(model(x), dim=1).topk(2, dim=1).values
            clean_margins.extend((top2[:, 0] - top2[:, 1]).cpu().tolist())
            x_adv = pgd(x, y)
            with torch.no_grad():
                top2_adv = F.softmax(model(x_adv), dim=1).topk(2, dim=1).values
            adv_margins.extend((top2_adv[:, 0] - top2_adv[:, 1]).cpu().tolist())
    return {"clean_margins": clean_margins, "adv_margins": adv_margins}


def run_epsilon_sweep_for_model(
    model, loader, name, epsilons, count_quant_layers_fn=None, safe_set=None
):
    """Evaluate epsilon-sweep metrics with the main suite implementations.

    Shared metrics use the same seeded, per-example worst-case aggregation as
    ``run_suite``, so matching model/epsilon rows are directly comparable.
    """
    if safe_set is None:

        def safe_set(target, key, fn, warning, *, context=None, default=None):
            try:
                target[key] = fn()
            except Exception as exc:
                suffix = f" for {context}" if context else ""
                print(f"  [WARN] {warning}{suffix}: {exc}")
                target[key] = default
            return target[key]

    is_quant = count_quant_layers_fn(model) > 0 if count_quant_layers_fn else False
    rows = []
    for eps in epsilons:
        row = {"model": name, "epsilon": eps}
        context = f"{name} eps={eps:.4f}"

        def add_shared_metrics(result, source_name, output_name):
            row[output_name] = result[source_name] if result is not None else None
            row[f"{source_name}_mean"] = (
                result[f"{source_name}_mean"] if result is not None else None
            )
            row[f"{source_name}_std"] = (
                result[f"{source_name}_std"] if result is not None else None
            )

        pgd = safe_set(
            {},
            "result",
            lambda: run_pgd(model, loader, eps=eps, seeds=SEEDS),
            "PGD (hard-round) sweep failed",
            context=context,
        )
        add_shared_metrics(pgd, "PGD", "PGD_acc")

        noise = safe_set(
            {},
            "result",
            lambda: run_random_noise_seeded(model, loader, eps=eps, seeds=SEEDS),
            "random_noise sweep failed",
            context=context,
        )
        add_shared_metrics(noise, "Random_Noise", "Random_Noise_acc")

        if is_quant:
            bpda = safe_set(
                {},
                "result",
                lambda: run_bpda(model, loader, eps=eps, n_restarts=1, seeds=SEEDS),
                "BPDA sweep failed",
                context=context,
            )
            add_shared_metrics(bpda, "BPDA_PGD", "BPDA_acc")
        rows.append(row)
    return rows
