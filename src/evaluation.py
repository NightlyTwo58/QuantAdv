"""
Evaluation routines: clean-accuracy checks, adversarial-attack sweeps
(FGSM/PGD, AutoAttack, BPDA, random-noise), and epsilon sweeps. These build
on the primitive attacks in `quantadv.attacks` and the `Model` quantization
wrapper's static helpers.
"""
import os
import traceback

import numpy as np
import pandas as pd
import torch

from autoattack import AutoAttack

from .config import device, SEEDS, CLIP_MIN_DEV, CLIP_MAX_DEV
from .model import Model as QuantModel
from .attacks import (
    amp_ctx,
    tolerate_masked_gradients,
    pgd_step,
    fgsm_attack,
    pgd_attack,
    bpda_pgd_attack,
    random_noise_attack,
    evaluate_under_attack,
)


def sanity_check_accuracy(model, loader):
    """Delegate to QuantModel.clean_accuracy."""
    return QuantModel.clean_accuracy(model, loader)


def count_quant_layers(model):
    """Count quantized layers using the QuantModel helper."""
    return QuantModel._count_quant_layers(model)


def safe_run(fn, name, label):
    try:
        return fn(), None
    except Exception as e:
        print(f"  [WARN] {label} failed for {name}: {e}")
        traceback.print_exc()
        return None, e


def seeded_average(fn, seeds, key):
    vals = [fn(s) for s in seeds]
    return {key: float(np.mean(vals)), f"{key}_mean": float(np.mean(vals)), f"{key}_std": float(np.std(vals))}


def run_fgsm_pgd(model, loader, eps=8 / 255, seeds=SEEDS):
    model.eval()
    out = {"FGSM": evaluate_under_attack(model, loader, lambda x, y: fgsm_attack(model, x, y, eps))}

    def run_seed(seed):
        torch.manual_seed(seed)
        return evaluate_under_attack(
            model, loader,
            lambda x, y: pgd_attack(model, x, y, eps, alpha=2 / 255, steps=20, random_start=True))

    out.update(seeded_average(run_seed, seeds, "PGD"))
    return out


def run_autoattack(model, loader, eps=8 / 255, aa_batch_size=256):
    model.eval()
    adversary = AutoAttack(model, norm="Linf", eps=eps, version="custom", device=device, verbose=False)
    adversary.attacks_to_run = ["apgd-ce", "apgd-t"]

    x_all = torch.cat([x for x, _ in loader], dim=0)
    y_all = torch.cat([y for _, y in loader], dim=0)
    bs = min(aa_batch_size, x_all.size(0))

    with amp_ctx(), tolerate_masked_gradients():
        x_adv = adversary.run_standard_evaluation(x_all, y_all, bs=bs)
        with torch.no_grad():
            pred = model(x_adv).argmax(1)
    correct = (pred == y_all).sum().item()
    total = y_all.size(0)
    return correct / total


def _run_bpda_once(model, loader, eps, n_restarts, backward_model=None):
    correct_masks = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        worst_correct = torch.ones(y.size(0), dtype=torch.bool, device=device)
        for _ in range(n_restarts):
            x_adv = bpda_pgd_attack(model, x, y, eps=eps, backward_model=backward_model)
            with torch.no_grad(), amp_ctx():
                pred = model(x_adv).argmax(dim=1)
            worst_correct &= (pred == y)
        correct_masks.append(worst_correct)
    all_correct = torch.cat(correct_masks)
    return all_correct.float().mean().item()


def run_bpda(model, loader, eps=8 / 255, n_restarts=1, seeds=SEEDS, backward_model=None):
    """
    backward_model: differentiable surrogate used to compute BPDA gradients
    (see quantadv.attacks.bpda_pgd_attack). For a quantized `model`, this
    should be the full-precision shadow network it was derived from -- e.g.
    the `fp32_ref` / `ref` model already tracked in experiment.py -- so the
    hard round()/quantization step doesn't just zero out the gradient.

    If omitted, this call degrades to plain PGD against `model`'s own
    forward pass, and any masked-gradient RuntimeWarnings raised by
    attacks.py apply: a quantized model's true int8 kernels commonly have
    no (or a zeroed) backward, so an unset backward_model here typically
    means "BPDA_acc" is not actually BPDA, just PGD with little or no
    gradient signal.
    """
    def run_seed(seed):
        torch.manual_seed(seed)
        return _run_bpda_once(model, loader, eps, n_restarts, backward_model=backward_model)

    return seeded_average(run_seed, seeds, "BPDA_PGD")


def run_random_noise_seeded(model, loader, eps=8 / 255, seeds=SEEDS):
    return seeded_average(lambda s: random_noise_attack(model, loader, eps=eps, seed=s), seeds, "Random_Noise")


def run_pgd_epsilon_sweep_shared(model, loader, epsilons, alpha=2 / 255, steps=20):
    """
    Epsilon-projection PGD sweep (item #4). Instead of a full 20-step PGD
    run for every epsilon (5 epsilons x 20 steps = 100 PGD iterations per
    model), this runs ONE 20-step PGD attack at eps_max and, for every
    smaller epsilon in `epsilons`, projects the final perturbation onto
    that epsilon's L_inf ball. This is the amortized epsilon-sweep
    approximation used by several robustness libraries.
    """
    eps_max = max(epsilons)

    correct_per_eps = {eps: 0 for eps in epsilons}
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps_max, eps_max)
        x_adv = torch.max(torch.min(x_adv, CLIP_MAX_DEV), CLIP_MIN_DEV).detach()

        for _ in range(steps):
            x_adv = pgd_step(model, x_adv, x, y, eps_max, alpha)

        with torch.no_grad(), amp_ctx():
            for eps in epsilons:
                delta = torch.clamp(x_adv - x, -eps, eps)
                x_eps = torch.max(torch.min(x + delta, CLIP_MAX_DEV), CLIP_MIN_DEV)
                pred = model(x_eps).argmax(dim=1)
                correct_per_eps[eps] += (pred == y).sum().item()
        total += y.size(0)

    return {eps: correct_per_eps[eps] / total for eps in epsilons}


def run_epsilon_sweep_for_model(model, loader, name, epsilons, backward_model=None):
    """
    backward_model: fp32 shadow network (see `run_bpda`), forwarded into
    the per-epsilon BPDA_acc calls below when `model` is quantized. Callers
    (experiment.py) already build and compile this reference model
    alongside each quantized variant -- pass it in here, don't drop it.
    """
    rows = []
    is_quant = count_quant_layers(model) > 0
    if is_quant and backward_model is None:
        print(f"  [WARN] {name} is quantized but no backward_model/fp32 shadow was supplied "
              f"for BPDA. Gradients through the quantization step are likely masked, so "
              f"BPDA_acc below will silently degrade toward plain-PGD accuracy rather than "
              f"reflecting a real BPDA attack.")

    pgd_acc_by_eps, _ = safe_run(
        lambda: run_pgd_epsilon_sweep_shared(model, loader, epsilons), name, "shared-trajectory PGD sweep")
    pgd_acc_by_eps = pgd_acc_by_eps or {}

    for eps in epsilons:
        row = {"model": name, "epsilon": eps}
        row["PGD_acc"] = pgd_acc_by_eps.get(eps)
        row["Random_Noise_acc"], _ = safe_run(
            lambda: random_noise_attack(model, loader, eps=eps), name, f"random_noise sweep eps={eps:.4f}")

        if is_quant:
            row["BPDA_acc"], _ = safe_run(
                lambda: _run_bpda_once(model, loader, eps=eps, n_restarts=3, backward_model=backward_model),
                name, f"BPDA sweep eps={eps:.4f}")
        rows.append(row)
    return rows


def load_resumable_csv(path, key_fn, empty_columns=("model",)):
    if os.path.exists(path):
        df = pd.read_csv(path)
        done = set(key_fn(row) for row in df.to_dict("records"))
        rows = df.to_dict("records")
    else:
        df = pd.DataFrame(columns=list(empty_columns))
        done = set()
        rows = []
    return df, done, rows
