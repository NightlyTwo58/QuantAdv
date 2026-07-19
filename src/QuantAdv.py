#!/usr/bin/env python
# coding: utf-8
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
import os
import json
import traceback
import sys
import warnings
import time
import threading
import psutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pytorchcv.model_provider import get_model as ptcv_get_model

import torchattacks
from autoattack import AutoAttack

import defense as dfn
from src.graphs import data as report_data
from src.graphs.data import csv_path, defense_summary_csv_path, json_path
import stats as qstats

from config import *
from quantization import *
from ResourceMonitor import ResourceMonitor
from attack import *
from attack_cache import AttackResultCache

"""QuantAdv experiment runner for quantization and adversarial robustness.

This module builds FP32, post-training quantized, quantization-aware trained,
and optional defended image classifiers, then evaluates them with white-box,
transfer, adaptive, black-box, and diagnostic attacks.  Quantized layers are
fake-quantized float modules: they simulate integer rounding during forward
passes while optionally using straight-through gradients for attacks and QAT.
"""




def check_environment():
    """Validate runtime packages and configured local data before a full run."""
    missing = [
        pkg
        for pkg in ("torchattacks", "autoattack", "pytorchcv", "torchao")
        if importlib.util.find_spec(pkg) is None
    ]
    if missing:
        raise ImportError(
            f"Missing packages: {missing}.\nInstall via: pip install -r requirements.txt"
        )
    print("All required packages are available.")
    print("device:", device)
    if not DATASET_DOWNLOAD and not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Expected extracted {DATASET_NAME} data at {DATASET_DIR!r}"
        )


def get_dataloaders(
    batch_size=DEFAULT_BATCH_SIZE, eval_n=DEFAULT_EVAL_N, finetune_n=DEFAULT_FINETUNE_N
):
    """Build fine-tuning and evaluation loaders for the configured dataset."""
    train_full = DATASET_CLASS(
        root=DATASET_ROOT,
        download=DATASET_DOWNLOAD,
        transform=DATASET_TRAIN_TRANSFORM,
        **DATASET_TRAIN_KWARGS,
    )
    test_full = DATASET_CLASS(
        root=DATASET_ROOT,
        download=DATASET_DOWNLOAD,
        transform=DATASET_TEST_TRANSFORM,
        **DATASET_TEST_KWARGS,
    )
    finetune_subset = torch.utils.data.Subset(train_full, list(range(finetune_n)))
    eval_subset = torch.utils.data.Subset(test_full, list(range(eval_n)))
    workers = min(MAX_DATA_WORKERS, os.cpu_count() or 1)
    finetune_loader = torch.utils.data.DataLoader(
        finetune_subset,
        batch_size=batch_size,
        shuffle=TRAIN_SHUFFLE,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=workers > 0,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_subset,
        batch_size=DEFAULT_EVAL_BATCH_SIZE,
        shuffle=EVAL_SHUFFLE,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=workers > 0,
    )
    return finetune_loader, eval_loader


def load_pretrained(arch_key):
    """Load a configured pretrained TorchCV architecture onto the active device."""
    if ptcv_get_model is None:
        raise ImportError(
            "Missing package 'pytorchcv'. Install via: pip install -r requirements.txt"
        )
    try:
        model_name = PRETRAINED_NAMES[arch_key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown architecture {arch_key!r}; configured choices are "
            f"{tuple(PRETRAINED_NAMES)}"
        ) from exc
    model = ptcv_get_model(model_name, **PRETRAINED_MODEL_KWARGS)
    return model.to(device).eval()


def sanity_check_accuracy(model, loader):
    """Compute clean accuracy with the same evaluation path as attack metrics."""
    model.eval()
    return accuracy_from_adv_fn(model, loader, use_autocast=True)


def with_image_compression(
    model, size=COMPRESS_IMAGE_SIZE, bits=COMPRESS_IMAGE_BITS, mode=COMPRESS_IMAGE_MODE
):
    """Return an eval-mode copy of ``model`` behind input compression."""
    return (
        CompressedInputModel(copy.deepcopy(model), size=size, bits=bits, mode=mode)
        .to(device)
        .eval()
    )


def add_paired_fp32_mcnemar_tests(df_results):
    """Post-hoc paired tests between each variant and its architecture's FP32 model."""
    return qstats.add_paired_mcnemar_tests(
        df_results, baseline_name=qstats.fp32_baseline_name
    )


def gradient_diagnostics(
    model, loader, fp32_ref=None, max_batches=GRAD_DIAG_MAX_BATCHES
):
    """Ground-truth masking check: compare hard-round vs. STE input gradients."""
    frac_zero_hard, norm_hard = [], []
    frac_zero_ste, norm_ste = [], []
    cos_sims = []
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with ste_mode(model, False):
            x_in = x.clone().requires_grad_(True)
            loss = F.cross_entropy(model(x_in), y)
            g_hard = torch.autograd.grad(loss, x_in)[0].flatten()
        frac_zero_hard.append(
            (g_hard.abs() < GRAD_ZERO_THRESHOLD).float().mean().item()
        )
        norm_hard.append(g_hard.norm().item())
        with ste_mode(model, True):
            x_in2 = x.clone().requires_grad_(True)
            loss2 = F.cross_entropy(model(x_in2), y)
            g_ste = torch.autograd.grad(loss2, x_in2)[0].flatten()
        frac_zero_ste.append((g_ste.abs() < GRAD_ZERO_THRESHOLD).float().mean().item())
        norm_ste.append(g_ste.norm().item())
        if fp32_ref is not None:
            fp32_ref.eval()
            x_ref = x.clone().requires_grad_(True)
            loss_ref = F.cross_entropy(fp32_ref(x_ref), y)
            g_ref = torch.autograd.grad(loss_ref, x_ref)[0].flatten()
            cos_sims.append(
                F.cosine_similarity(g_ste.unsqueeze(0), g_ref.unsqueeze(0)).item()
            )
    diagnostics = {
        "frac_zero_grad_hard": float(np.mean(frac_zero_hard)),
        "frac_zero_grad_ste": float(np.mean(frac_zero_ste)),
        "grad_norm_hard": float(np.mean(norm_hard)),
        "grad_norm_ste": float(np.mean(norm_ste)),
    }
    if cos_sims:
        diagnostics["grad_cosine_sim_with_FP32"] = float(np.mean(cos_sims))
    return diagnostics


def layerwise_grad_profile(model, loader, use_ste, max_batches=LAYERWISE_MAX_BATCHES):
    """Collect per-layer gradient statistics for a model and loader."""
    quant_layers = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, (QuantConv2d, QuantLinear))
    ]
    norms = {n: [] for n, _ in quant_layers}
    handles = []

    def make_hook(name):
        """Create a forward hook that tracks gradients on a layer's input tensor."""

        def hook(module, inputs):
            """Attach directly to the activation without backward-hook view wrapping."""
            activation = inputs[0]
            if activation.requires_grad:
                activation.register_hook(
                    lambda grad: norms[name].append(
                        grad.flatten(1).norm(dim=1).mean().item()
                    )
                )

        return hook

    try:
        for n, m in quant_layers:
            handles.append(m.register_forward_pre_hook(make_hook(n)))
        model.eval()
        with ste_mode(model, use_ste):
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
    ordered_names = [n for n, _ in quant_layers]
    return {
        n: (float(np.mean(norms[n])) if len(norms[n]) else None) for n in ordered_names
    }


def component_ablation_row(
    name, label, quant_weight, quant_act, clean_acc, pgd, bpda, frac_zero_grad_hard
):
    """Build one consistently shaped quantization-component ablation row."""
    pgd_acc = pgd.get("PGD")
    bpda_acc = bpda.get("BPDA_PGD")
    return {
        "model": name,
        "config": label,
        "quant_weight": quant_weight,
        "quant_act": quant_act,
        "clean_acc": clean_acc,
        "PGD_acc": pgd_acc,
        "PGD_hard_acc": pgd_acc,
        "PGD_mean": pgd.get("PGD_mean"),
        "PGD_std": pgd.get("PGD_std"),
        "BPDA_acc": bpda_acc,
        "BPDA_mean": bpda.get("BPDA_PGD_mean"),
        "BPDA_std": bpda.get("BPDA_PGD_std"),
        "PGD_minus_BPDA": (
            pgd_acc - bpda_acc
            if pgd_acc is not None and bpda_acc is not None
            else None
        ),
        "frac_zero_grad_hard": frac_zero_grad_hard,
    }


def main_both_component_ablation_row(name, results):
    """Reuse main-suite metrics for the already evaluated fully quantized model."""
    return component_ablation_row(
        name,
        "both",
        True,
        True,
        results.get("clean_acc"),
        {
            key: results.get(key)
            for key in ("PGD", "PGD_mean", "PGD_std")
        },
        {
            key: results.get(key)
            for key in ("BPDA_PGD", "BPDA_PGD_mean", "BPDA_PGD_std")
        },
        results.get("frac_zero_grad_hard"),
    )


def run_quant_component_ablation(model, loader, name, eps=DEFAULT_EPS):
    """Evaluate which quantized components are responsible for observed effects.

    The final artifact distinguishes weight-only, activation-only, and fully
    quantized models. This function computes only the first two; the ``both``
    row is populated from the main suite rather than attacked a second time.
    For each interpretation the experiment reports ordinary hard-round PGD and
    a budget-matched BPDA-PGD. A large PGD vs BPDA-PGD gap is evidence
    of gradient masking, not evidence that quantization itself improves
    robustness.

    Weight quantization alone cannot produce this gap: rounding a weight
    doesn't touch the gradient path back to the input (the conv/linear op is
    still linear in ``x`` for whatever rounded weight value it has), so
    ``weight_only`` should show ``frac_zero_grad_hard`` near the background
    rate and a small PGD-vs-BPDA gap. Only ``act_only``/``both`` quantize a
    tensor that ``x`` actually flows through, so only those can mask.
    """
    configs = [("weight_only", True, False), ("act_only", False, True)]
    rows = []
    try:
        for label, qw, qa in configs:
            set_quant_components(model, qw, qa)
            clean_acc = sanity_check_accuracy(model, loader)

            pgd_hard = run_pgd(model, loader, eps=eps, seeds=SEEDS)
            with ste_mode(model, False):
                x, y = next(iter(loader))
                x, y = x.to(device), y.to(device)
                x_in = x.clone().requires_grad_(True)
                loss = F.cross_entropy(model(x_in), y)
                g_hard = torch.autograd.grad(loss, x_in)[0].flatten()
                frac_zero = (
                    (g_hard.abs() < GRAD_ZERO_THRESHOLD).float().mean().item()
                )

            bpda = run_bpda(
                model,
                loader,
                eps=eps,
                n_restarts=1,
                seeds=SEEDS,
            )
            rows.append(
                component_ablation_row(
                    name, label, qw, qa, clean_acc, pgd_hard, bpda, frac_zero
                )
            )
    finally:
        # Never leave the shared registry model in an ablated state after failure.
        set_quant_components(model, True, True)
    return rows


def run_chunk_quantization_attacks(
    fp32_model,
    loader,
    name,
    bits=8,
    n_chunks=CHUNK_QUANT_NUM_CHUNKS,
    eps=DEFAULT_EPS,
):
    """Evaluate attacks as contiguous chunks of layers are quantized."""
    layer_names = quantizable_layer_names(fp32_model)
    chunks = quant_layer_chunks(layer_names, n_chunks)
    rows = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_model = convert_layer_chunk_to_quant(
            fp32_model, chunk, bits=bits, quant_weight=True, quant_act=True
        )
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
            print(
                f"  [WARN] chunk clean_acc failed for {name} {row['chunk_label']}: {e}"
            )
            row["clean_acc"] = None
        try:
            pgd = run_pgd(chunk_model, loader, eps=eps, seeds=SEEDS)
            row["PGD_acc"] = pgd["PGD"]
            row["PGD_mean"] = pgd["PGD_mean"]
            row["PGD_std"] = pgd["PGD_std"]
        except Exception as e:
            print(f"  [WARN] chunk PGD failed for {name} {row['chunk_label']}: {e}")
            row["PGD_acc"] = None
        rows.append(row)
        del chunk_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def safe_call(
    fn, warning, *, context=None, default=None, show_traceback=False, level="WARN"
):
    """Run an optional experiment stage with consistent warning and fallback handling."""
    try:
        return fn()
    except Exception as exc:
        suffix = f" for {context}" if context else ""
        print(f"  [{level}] {warning}{suffix}: {exc}")
        if show_traceback:
            traceback.print_exc()
        return default() if callable(default) else default


def safe_set(target, key, fn, warning, *, context=None, default=None):
    """Run a metric function and store its value with warning-based fallback."""
    target[key] = safe_call(fn, warning, context=context, default=default)
    return target[key]


def safe_update(target, fn, warning, *, context=None, defaults=None):
    """Merge a metric dictionary into a target dictionary with warning fallback."""
    update = safe_call(fn, warning, context=context, default=None)
    if update is not None:
        target.update(update)
        return True
    if defaults:
        target.update(
            {
                key: value() if callable(value) else value
                for key, value in defaults.items()
            }
        )
    return False


def safe_set_vector(results, vectors, metric, fn, warning, *, context=None):
    """Run a vector-producing metric and store values plus correctness vectors."""
    pair = safe_call(fn, warning, context=context, default=None)
    if pair is None:
        results[metric] = None
        return False
    results[metric], vectors[metric] = pair
    return True


def safe_update_vectors(results, vectors, fn, warning, *, context=None, defaults=None):
    """Merge metric and vector outputs into their destination dictionaries."""
    update = safe_call(fn, warning, context=context, default=None)
    if update is None:
        if defaults:
            results.update(defaults)
        return False
    vectors.update(update.pop("_vectors", {}))
    results.update(update)
    return True


def save_json(path, data, *, indent=None):
    """Serialize a Python object as pretty-printed JSON."""
    with open(path, "w") as handle:
        json.dump(data, handle, indent=indent)


def run_suite(
    model, loader, name, fp32_ref=None, eps=DEFAULT_EPS, attack_cache=None
):
    """Run the full attack and diagnostic suite for one model."""
    model.eval()
    results = {"model": name}
    vectors = {}
    safe_set_vector(
        results,
        vectors,
        "clean_acc",
        lambda: accuracy_from_adv_fn(
            model, loader, use_autocast=True, return_vector=True
        ),
        "clean_acc failed",
        context=name,
    )
    safe_update_vectors(
        results,
        vectors,
        lambda: run_fgsm_pgd(
            model, loader, eps=eps, return_vectors=True, use_ste=False,
            cache=attack_cache,
        ),
        "FGSM/PGD failed",
        context=name,
        defaults={"FGSM": None, "PGD": None},
    )
    safe_set_vector(
        results,
        vectors,
        "AutoAttack",
        lambda: run_autoattack(
            model, loader, eps=eps, return_vector=True, use_ste=False
        ),
        "AutoAttack failed",
        context=name,
    )
    if RUN_EXTRA_WHITEBOX_ATTACKS:
        safe_update(
            results,
            lambda: run_extra_whitebox_attacks(model, loader, eps=eps, use_ste=False),
            "CW/DeepFool/JSMA failed",
            context=name,
        )
    if RUN_UAP_ATTACKS:
        safe_set(
            results,
            "UAP",
            lambda: run_uap_attack(model, loader, eps=eps),
            "UAP attack failed",
            context=name,
        )
    if RUN_SURROGATE_ATTACK:
        safe_set(
            results,
            "Surrogate_Transfer",
            lambda: run_surrogate_attack(model, loader, eps=eps),
            "surrogate attack failed",
            context=name,
        )
    if fp32_ref is not None:
        safe_set_vector(
            results,
            vectors,
            "Transfer_from_FP32",
            lambda: transfer_attack(
                fp32_ref,
                model,
                loader,
                eps=eps,
                return_vector=True,
                use_ste=False,
                cache=attack_cache,
            ),
            "transfer_attack failed",
            context=name,
        )
        safe_set_vector(
            results,
            vectors,
            "MIM_Transfer",
            lambda: transfer_attack_mim(
                fp32_ref,
                model,
                loader,
                eps=eps,
                return_vector=True,
                use_ste=False,
                cache=attack_cache,
            ),
            "MIM transfer_attack failed",
            context=name,
        )
        if RUN_UAP_ATTACKS:
            safe_set(
                results,
                "UAP_Transfer",
                lambda: transfer_uap_attack(
                    fp32_ref, model, loader, eps=eps, cache=attack_cache
                ),
                "UAP transfer_attack failed",
                context=name,
            )
        if RUN_REVERSE_TRANSFERS and count_quant_layers(model) > 0:
            safe_set(
                results,
                "Transfer_to_FP32",
                lambda: transfer_attack(
                    model, fp32_ref, loader, eps=eps, use_ste=False
                ),
                "reverse transfer_attack failed",
                context=name,
            )
            safe_set(
                results,
                "MIM_Transfer_to_FP32",
                lambda: transfer_attack_mim(
                    model, fp32_ref, loader, eps=eps, use_ste=False
                ),
                "reverse MIM transfer_attack failed",
                context=name,
            )
            safe_set(
                results,
                "UAP_Transfer_to_FP32",
                lambda: transfer_uap_attack(model, fp32_ref, loader, eps=eps),
                "reverse UAP transfer_attack failed",
                context=name,
            )
    safe_update_vectors(
        results,
        vectors,
        lambda: run_random_noise_seeded(
            model, loader, eps=eps, return_vector=True, cache=attack_cache
        ),
        "random_noise_attack failed",
        context=name,
        defaults={"Random_Noise": None},
    )
    defense_model = unwrap_model(model)
    adaptive_defaults = {}
    if isinstance(defense_model, dfn.SanitizedModel):
        adaptive_defaults["BPDA_Adaptive"] = None
    elif isinstance(defense_model, dfn.SmoothedModel):
        adaptive_defaults["EOT_PGD"] = None
    elif isinstance(defense_model, dfn.GuardrailModel):
        adaptive_defaults["Adaptive_Guardrail"] = None
    elif isinstance(defense_model, dfn.DetectGuardModel):
        adaptive_defaults["Adaptive_DetectGuard"] = None
    safe_update(
        results,
        lambda: run_defense_adaptive_attacks(model, loader, eps=eps),
        "adaptive defense attack failed",
        context=name,
        defaults=adaptive_defaults,
    )

    if any(
        name == f"{arch_key}_{variant}"
        for arch_key in PRETRAINED_NAMES
        for variant in PGD_ABLATION_VARIANTS
    ):

        def save_pgd_ablation():
            """Persist PGD and BPDA step-ablation diagnostics."""
            rows = []
            for attack_name, use_ste in (("PGD", False), ("BPDA_PGD", True)):
                ablation = pgd_steps_ablation(model, loader, eps=eps, use_ste=use_ste)
                rows.extend(
                    {
                        "model": name,
                        "attack": attack_name,
                        "steps": steps,
                        "acc": accuracy,
                    }
                    for steps, accuracy in ablation.items()
                )
            pd.DataFrame(rows).to_csv(csv_path(name, "ablation"), index=False)

        safe_call(save_pgd_ablation, "pgd_steps_ablation failed", context=name)

    if count_quant_layers(model) > 0:
        safe_update_vectors(
            results,
            vectors,
            lambda: run_bpda(
                model,
                loader,
                eps=eps,
                n_restarts=1,
                return_vector=True,
                cache=attack_cache,
            ),
            "BPDA failed",
            context=name,
            defaults={"BPDA_PGD": None},
        )
        safe_update(
            results,
            lambda: gradient_diagnostics(
                model, loader, fp32_ref=fp32_ref, max_batches=GRAD_DIAG_MAX_BATCHES
            ),
            "gradient_diagnostics failed",
            context=name,
        )
        safe_update(
            results,
            lambda: staircase_diagnostic(model, loader),
            "staircase_diagnostic failed",
            context=name,
        )
        boundary_defaults = {
            "Boundary_acc": None,
            "Boundary_mean_Linf": None,
            "Boundary_median_Linf": None,
            "Boundary_min_Linf": None,
            "Boundary_max_Linf": None,
            "Boundary_std_Linf": None,
            "Boundary_n": 0,
            "Boundary_init_failed": None,
            "Boundary_init_failed_rate": None,
        }
        if RUN_BOUNDARY_ATTACK:
            safe_update(
                results,
                lambda: run_boundary_attack(
                    model,
                    loader,
                    eps=eps,
                    max_images=BOUNDARY_MAX_IMAGES_SUITE,
                    steps=BOUNDARY_STEPS_SUITE,
                    seed=BOUNDARY_SEED,
                ),
                "boundary_attack failed",
                context=name,
                defaults=boundary_defaults,
            )
        else:
            results.update(boundary_defaults)
        if RUN_NES_ATTACK:
            safe_update(
                results,
                lambda: run_nes_attack(
                    model,
                    loader,
                    eps=eps,
                    seeds=SEEDS,
                    n_samples=NES_SAMPLES_SUITE,
                    query_chunk=NES_QUERY_CHUNK,
                ),
                "NES attack failed",
                context=name,
                defaults={"NES": None},
            )

        if RUN_PGD_TRAJECTORY:

            def save_trajectory():
                """Persist PGD trajectory diagnostics for the current model."""
                traj = pgd_trajectory_diagnostics(
                    model, loader, eps=eps, max_batches=TRAJECTORY_MAX_BATCHES
                )
                save_json(json_path(name, "trajectory"), traj, indent=2)

            safe_call(
                save_trajectory, "pgd_trajectory_diagnostics failed", context=name
            )
        if RUN_LAYERWISE_PROFILE:

            def save_layerwise_profile():
                """Persist layerwise gradient diagnostics for the current model."""
                prof_hard = layerwise_grad_profile(model, loader, use_ste=False)
                prof_ste = layerwise_grad_profile(model, loader, use_ste=True)
                rows = [
                    {
                        "model": name,
                        "layer": n,
                        "grad_norm_hard": prof_hard.get(n),
                        "grad_norm_ste": prof_ste.get(n),
                    }
                    for n in prof_hard
                ]
                pd.DataFrame(rows).to_csv(csv_path(name, "layerwise"), index=False)

            safe_call(
                save_layerwise_profile, "layerwise_grad_profile failed", context=name
            )
        # weight-only / activation-only / both ablation
        if RUN_COMPONENT_ABLATION:

            def save_component_ablation():
                """Persist quantization component-ablation diagnostics for the current model."""
                rows = run_quant_component_ablation(model, loader, name, eps=eps)
                rows.append(main_both_component_ablation_row(name, results))
                pd.DataFrame(rows).to_csv(
                    csv_path(name, "component_ablation"), index=False
                )

            safe_call(
                save_component_ablation,
                "run_quant_component_ablation failed",
                context=name,
            )
        if RUN_CONFIDENCE_MARGIN:

            def save_confidence_margins():
                """Persist confidence-margin diagnostics for the current model."""
                margins = confidence_margin_diagnostic(
                    model, loader, eps=eps, max_batches=MARGIN_MAX_BATCHES
                )
                save_json(json_path(name, "margin"), margins)

            safe_call(
                save_confidence_margins,
                "confidence_margin_diagnostic failed",
                context=name,
            )
    for metric, vector in vectors.items():
        qstats.add_binomial_statistics(
            results, metric, vector, confidence=CI_CONFIDENCE
        )
    if "PGD" in vectors and "BPDA_PGD" in vectors:
        test = qstats.mcnemar_exact(vectors["PGD"], vectors["BPDA_PGD"])
        results.update(
            {f"McNemar_PGD_vs_BPDA_{key}": value for key, value in test.items()}
        )
    if vectors:
        results["correctness_vectors_path"] = qstats.save_correctness_vectors(
            name, vectors, PER_EXAMPLE_DIR
        )
    return results


def run_epsilon_sweep_for_model_wrapped(
    model, loader, name, epsilons, attack_cache=None
):
    """Run and annotate an epsilon sweep for one named model."""
    return run_epsilon_sweep_for_model(
        model,
        loader,
        name,
        epsilons,
        count_quant_layers_fn=count_quant_layers,
        safe_set=safe_set,
        cache=attack_cache,
    )


def completed_sweep_keys(df_sweep):
    """Return only rows produced by the shared seeded sweep implementation."""
    required = {"model", "epsilon", "PGD_acc", "PGD_mean", "PGD_std"}
    if df_sweep.empty or not required.issubset(df_sweep.columns):
        return set()
    complete = df_sweep[["PGD_acc", "PGD_mean", "PGD_std"]].notna().all(axis=1)
    return set(
        zip(
            df_sweep.loc[complete, "model"].astype(str),
            df_sweep.loc[complete, "epsilon"].round(6),
        )
    )


def run_defense_suite(model_registry, finetune_loader, eval_loader):
    """Build defended models and evaluate them under adaptive attacks."""
    summary_rows = []
    arch_keys = sorted(
        {name.split("_FP32")[0] for name in model_registry if name.endswith("_FP32")}
    )
    for arch_key in arch_keys:
        fp32_entry = model_registry.get(f"{arch_key}_FP32")
        int8_qat_entry = model_registry.get(f"{arch_key}_int8_QAT")
        int4_qat_entry = model_registry.get(f"{arch_key}_int4_QAT")
        if fp32_entry is None:
            continue
        fp32_model = fp32_entry[0]

        def add_fp32_at():
            """Add the adversarially trained FP32 defense to the registry when available."""
            fp32_at = dfn.prepare_adversarial_training(
                fp32_model, finetune_loader, bits=None
            )
            model_registry[f"{arch_key}_FP32_AT"] = (fp32_at, fp32_model)

        safe_call(
            add_fp32_at,
            "adversarial training (FP32) failed",
            context=arch_key,
            show_traceback=True,
            level="FAIL",
        )

        def add_int8_at():
            """Add the adversarially trained INT8 defense to the registry when available."""
            int8_at = dfn.prepare_adversarial_training(
                fp32_model, finetune_loader, bits=8
            )
            model_registry[f"{arch_key}_int8_QAT_AT"] = (int8_at, fp32_model)

        safe_call(
            add_int8_at,
            "adversarial training (int8) failed",
            context=arch_key,
            show_traceback=True,
            level="FAIL",
        )
        wrap_targets = [("FP32", fp32_model)]
        if int8_qat_entry is not None:
            wrap_targets.append(("int8_QAT", int8_qat_entry[0]))
        if int4_qat_entry is not None:
            wrap_targets.append(("int4_QAT", int4_qat_entry[0]))
        detector = safe_call(
            lambda: dfn.train_adversarial_detector(fp32_model, finetune_loader),
            "adversarial detector training failed",
            context=arch_key,
            show_traceback=True,
            level="FAIL",
        )
        for tag, base_model in wrap_targets:
            entry_name = f"{arch_key}_{tag}"

            def add_sanitized():
                """Add the input-sanitization defense to the registry."""
                sanitized = dfn.SanitizedModel(base_model).to(device).eval()
                model_registry[f"{entry_name}_Sanitized"] = (sanitized, fp32_model)

            safe_call(
                add_sanitized, "SanitizedModel failed", context=entry_name, level="FAIL"
            )

            def add_smoothed():
                """Add the randomized-smoothing defense to the registry."""
                smoothed = dfn.SmoothedModel(base_model).to(device).eval()
                model_registry[f"{entry_name}_Smoothed"] = (smoothed, fp32_model)
                cert_stats = dfn.run_certified_accuracy(smoothed, eval_loader)
                summary_rows.append(
                    {
                        "model": entry_name,
                        "defense": "randomized_smoothing",
                        **cert_stats,
                    }
                )

            safe_call(
                add_smoothed,
                "SmoothedModel/certification failed",
                context=entry_name,
                show_traceback=True,
                level="FAIL",
            )

            def add_guardrail():
                """Add the confidence guardrail defense to the registry."""
                guardrail = dfn.GuardrailModel(base_model).to(device).eval()
                model_registry[f"{entry_name}_Guardrail"] = (guardrail, fp32_model)
                pgd_for_flagging = make_torchattack(
                    torchattacks.PGD,
                    guardrail,
                    eps=DEFAULT_EPS,
                    alpha=PGD_ALPHA,
                    steps=PGD_STEPS,
                    random_start=PGD_RANDOM_START,
                )
                flag_stats = dfn.run_guardrail_flagging_rate(
                    guardrail, eval_loader, attack=pgd_for_flagging
                )
                summary_rows.append(
                    {"model": entry_name, "defense": "guardrail", **flag_stats}
                )

            safe_call(
                add_guardrail,
                "GuardrailModel failed",
                context=entry_name,
                show_traceback=True,
                level="FAIL",
            )
            if detector is not None:

                def add_detect_guard():
                    """Add the detector-based guardrail defense to the registry."""
                    detect_guard = (
                        dfn.DetectGuardModel(base_model, detector).to(device).eval()
                    )
                    model_registry[f"{entry_name}_DetectGuard"] = (
                        detect_guard,
                        fp32_model,
                    )
                    pgd_for_detect = make_torchattack(
                        torchattacks.PGD,
                        detect_guard,
                        eps=DEFAULT_EPS,
                        alpha=PGD_ALPHA,
                        steps=PGD_STEPS,
                        random_start=PGD_RANDOM_START,
                    )
                    catch_stats = dfn.run_detector_catch_rate(
                        detect_guard, eval_loader, attack=pgd_for_detect
                    )
                    summary_rows.append(
                        {"model": entry_name, "defense": "detector", **catch_stats}
                    )

                safe_call(
                    add_detect_guard,
                    "DetectGuardModel failed",
                    context=entry_name,
                    show_traceback=True,
                    level="FAIL",
                )
    df_defense = pd.DataFrame(summary_rows)
    if not df_defense.empty:
        df_defense.to_csv(defense_summary_csv_path(), index=False)
    return model_registry, df_defense


def parallelize(model):
    """Wrap a model in DataParallel when multiple CUDA devices are available."""
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        return nn.DataParallel(model)
    return model


def run_chaotic_dither_sweep(fp32_model, loader, arch_key, bits=8):
    """Evaluate chaotic quantization over dither-amplitude settings."""
    rows = []
    for amplitude in CHAOTIC_DITHER_AMPLITUDES:
        model = (
            convert_to_chaotic_quant(
                fp32_model,
                bits=bits,
                quant_weight=True,
                quant_act=True,
                dither_amplitude=amplitude,
            )
            .to(device)
            .eval()
        )
        clean = sanity_check_accuracy(model, loader)
        pgd = run_pgd(model, loader, eps=DEFAULT_EPS, seeds=SEEDS)
        rows.append(
            {
                "model": arch_key,
                "bits": bits,
                "dither_amplitude": amplitude,
                "clean_acc": clean,
                "PGD_acc": pgd["PGD"],
                "PGD_mean": pgd["PGD_mean"],
                "PGD_std": pgd["PGD_std"],
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def add_general_scores(df_results):
    """Compute comparable aggregate scores solely from already-produced metrics."""
    robust_candidates = [
        c
        for c in (
            "AutoAttack",
            "BPDA_PGD",
            "Transfer_from_FP32",
            "MIM_Transfer",
            "BPDA_Adaptive",
            "EOT_PGD",
            "Adaptive_Guardrail",
            "Adaptive_DetectGuard",
        )
        if c in df_results.columns
    ]
    if robust_candidates:
        df_results["General_Robustness"] = df_results[robust_candidates].min(
            axis=1, skipna=True
        )
    if {"clean_acc", "General_Robustness"}.issubset(df_results.columns):
        denom = df_results["clean_acc"].replace(0, np.nan)
        df_results["General_Robustness_Retention"] = (
            df_results["General_Robustness"] / denom
        )
    if {"PGD", "General_Robustness"}.issubset(df_results.columns):
        gap = (df_results["PGD"] - df_results["General_Robustness"]).clip(lower=0)
        denom = (
            df_results["clean_acc"].replace(0, np.nan)
            if "clean_acc" in df_results
            else 1.0
        )
        df_results["General_Masking_Score"] = (gap / denom).clip(lower=0, upper=1)
    return df_results


def main():
    """Run the command-line entry point for this module."""
    check_environment()
    finetune_loader, eval_loader = get_dataloaders()
    if os.path.exists(RESULTS_CSV):
        df_results = report_data.read_table(RESULTS_CSV)
        done = set(
            df_results.loc[
                df_results.get(
                    "clean_acc_n", pd.Series(index=df_results.index)
                ).notna(),
                "model",
            ].astype(str)
        )
    else:
        df_results = pd.DataFrame(columns=["model"])
        done = set()
    if os.path.exists(PERFORMANCE_CSV):
        existing_run_metrics = pd.read_csv(PERFORMANCE_CSV)
        run_metrics = (
            existing_run_metrics.to_dict("records")
            if "run_seconds" in existing_run_metrics.columns
            else []
        )
    else:
        run_metrics = []
    if RUN_EPSILON_SWEEP and os.path.exists(SWEEP_CSV):
        df_sweep = pd.read_csv(SWEEP_CSV)
        sweep_done = completed_sweep_keys(df_sweep)
    else:
        df_sweep = pd.DataFrame()
        sweep_done = set()
    chunk_model_names = []
    dither_rows = []
    defense_summary_frames = []

    def run_pending_epsilon_sweep(name, model):
        """Run an epsilon sweep for a model that still needs one."""
        nonlocal df_sweep, sweep_done
        if not RUN_EPSILON_SWEEP:
            return
        pending_eps = [
            eps for eps in SWEEP_EPSILONS if (name, round(eps, 6)) not in sweep_done
        ]
        if not pending_eps:
            print(f"  Skipping epsilon sweep for {name} (already done)")
            return
        print(f"\nSweeping {name} ...")

        def save_epsilon_sweep():
            nonlocal df_sweep, sweep_done
            rows = run_epsilon_sweep_for_model_wrapped(
                model, eval_loader, name, pending_eps, attack_cache=attack_cache
            )
            if rows:
                df_sweep = report_data.upsert_table(
                    SWEEP_CSV, pd.DataFrame(rows), ["model", "epsilon"]
                )
                sweep_done = completed_sweep_keys(df_sweep)

        safe_call(
            save_epsilon_sweep,
            "epsilon sweep failed",
            context=name,
            show_traceback=True,
        )

    for arch_key in PRETRAINED_NAMES:
        # Keep only one architecture's family resident at a time.  References
        # among variants still keep their shared FP32 baseline alive as needed.
        model_registry = {}
        attack_cache = AttackResultCache()
        print(f"\n>>> {arch_key} <<<")
        try:
            fp32 = load_pretrained(arch_key)
            fp32_layer_names = quantizable_layer_names(fp32)
            print(
                f"  FP32 quantizable nn.Conv2d/nn.Linear layers: {len(fp32_layer_names)}"
            )
            print(f"  first quantizable layers: {fp32_layer_names[:8]}")
            if not fp32_layer_names:
                raise RuntimeError(
                    f"{arch_key} exposes zero FP32 nn.Conv2d/nn.Linear layers."
                )
            acc = sanity_check_accuracy(fp32, eval_loader)
            print(f"  loaded pretrained {arch_key}, clean acc: {acc:.3f}")
            model_registry[f"{arch_key}_FP32"] = (fp32, None)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [FAIL] could not load {arch_key}: {e}")
            traceback.print_exc()
            continue
        try:
            int8_ptq = convert_to_quant(fp32, bits=8, quant_weight=True, quant_act=True)
            verify_quantization_layers(
                arch_key, fp32, int8_ptq, "int8 PTQ", fp32_layer_names
            )
            model_registry[f"{arch_key}_int8_PTQ"] = (int8_ptq, fp32)
        except Exception as e:
            print(f"  [FAIL] int8 PTQ for {arch_key}: {e}")
            traceback.print_exc()
            raise
        try:
            int4_ptq = convert_to_quant(fp32, bits=4, quant_weight=True, quant_act=True)
            verify_quantization_layers(
                arch_key, fp32, int4_ptq, "int4 PTQ", fp32_layer_names
            )
            model_registry[f"{arch_key}_int4_PTQ"] = (int4_ptq, fp32)
        except Exception as e:
            print(f"  [FAIL] int4 PTQ for {arch_key}: {e}")
            traceback.print_exc()
            raise
        try:
            int8_qat = prepare_qat(
                fp32,
                bits=8,
                finetune_loader=finetune_loader,
                epochs=QAT_MAIN_EPOCHS,
            )
            verify_quantization_layers(
                arch_key, fp32, int8_qat, "int8 QAT", fp32_layer_names
            )
            model_registry[f"{arch_key}_int8_QAT"] = (int8_qat, fp32)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [FAIL] int8 QAT for {arch_key}: {e}")
            traceback.print_exc()
            raise
        try:
            int4_qat = prepare_qat(
                fp32,
                bits=4,
                finetune_loader=finetune_loader,
                epochs=QAT_MAIN_EPOCHS,
            )
            verify_quantization_layers(
                arch_key, fp32, int4_qat, "int4 QAT", fp32_layer_names
            )
            model_registry[f"{arch_key}_int4_QAT"] = (int4_qat, fp32)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [FAIL] int4 QAT for {arch_key}: {e}")
            traceback.print_exc()
            raise
        if QUANTIZATION_DEBUG_ONLY:
            print(
                "\nQUANTIZATION_DEBUG_ONLY=True; skipping defenses, attacks, sweeps, and plots."
            )
            print("Architecture registry built:", list(model_registry.keys()))
            del model_registry, attack_cache
            fp32 = int8_ptq = int4_ptq = int8_qat = int4_qat = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        try:
            model_registry[f"{arch_key}_FP32_Compressed"] = (
                with_image_compression(fp32),
                fp32,
            )
        except Exception as e:
            print(f"  [FAIL] compressed FP32 for {arch_key}: {e}")
        if RUN_CHAOTIC_COMPRESS:
            try:
                chaotic_int8_ptq = convert_to_chaotic_quant(
                    fp32, bits=8, quant_weight=True, quant_act=True
                )
                model_registry[f"{arch_key}_chaotic_int8_PTQ"] = (
                    chaotic_int8_ptq,
                    fp32,
                )
            except Exception as e:
                print(f"  [FAIL] chaotic int8 PTQ for {arch_key}: {e}")
            try:
                chaotic_int4_ptq = convert_to_chaotic_quant(
                    fp32, bits=4, quant_weight=True, quant_act=True
                )
                model_registry[f"{arch_key}_chaotic_int4_PTQ"] = (
                    chaotic_int4_ptq,
                    fp32,
                )
            except Exception as e:
                print(f"  [FAIL] chaotic int4 PTQ for {arch_key}: {e}")
            try:
                compressed_chaotic_int8 = with_image_compression(
                    convert_to_chaotic_quant(
                        fp32, bits=8, quant_weight=True, quant_act=True
                    )
                )
                model_registry[f"{arch_key}_chaotic_int8_PTQ_Compressed"] = (
                    compressed_chaotic_int8,
                    fp32,
                )
            except Exception as e:
                print(f"  [FAIL] compressed chaotic int8 PTQ for {arch_key}: {e}")
            try:
                chaotic_int8_qat = prepare_qat(
                    fp32,
                    bits=8,
                    finetune_loader=finetune_loader,
                    epochs=QAT_MAIN_EPOCHS,
                    chaotic=True,
                )
                model_registry[f"{arch_key}_chaotic_int8_QAT"] = (
                    chaotic_int8_qat,
                    fp32,
                )
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [FAIL] chaotic int8 QAT for {arch_key}: {e}")
                traceback.print_exc()
        if RUN_DEFENSE_SUITE:
            try:
                model_registry, df_defense_summary = run_defense_suite(
                    model_registry, finetune_loader, eval_loader
                )
                if not df_defense_summary.empty:
                    defense_summary_frames.append(df_defense_summary)
                    print("\nDefense summary:")
                    print(df_defense_summary.to_string(index=False))
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [FAIL] run_defense_suite failed for {arch_key}: {e}")
                traceback.print_exc()
        if RUN_CHAOTIC_DITHER_SWEEP:
            dither_rows.extend(run_chaotic_dither_sweep(fp32, eval_loader, arch_key))
        if RUN_CHUNK_QUANTIZATION:
            chunk_model_names.append(arch_key)
            out_path = csv_path(arch_key, "chunk_quant")
            if os.path.exists(out_path):
                print(f"Skipping chunk quantization for {arch_key} (already in {out_path})")
            else:
                print(f"\nChunk quantization sweep for {arch_key} ...")
                try:
                    rows = run_chunk_quantization_attacks(
                        fp32, eval_loader, arch_key, bits=8,
                        n_chunks=CHUNK_QUANT_NUM_CHUNKS, eps=DEFAULT_EPS,
                    )
                    report_data.upsert_table(out_path, pd.DataFrame(rows), ["model", "chunk_id"])
                except Exception as e:
                    print(f"  [FAIL] chunk quantization sweep failed for {arch_key}: {e}")
                    traceback.print_exc()

        print("\nArchitecture registry built:", list(model_registry.keys()))
        model_registry = {
            name: (parallelize(model), parallelize(ref) if ref else None)
            for name, (model, ref) in model_registry.items()
        }
        for name, (model, ref) in model_registry.items():
            if name in done:
                print(f"Skipping {name} (already in {RESULTS_CSV})")
                run_pending_epsilon_sweep(name, model)
                continue
            print(f"\nEvaluating {name} ...")
            monitor = ResourceMonitor(model, name)
            try:
                if RECORD_RUN_METRICS:
                    with monitor:
                        res = run_suite(model, eval_loader, name, fp32_ref=ref, attack_cache=attack_cache)
                else:
                    res = run_suite(model, eval_loader, name, fp32_ref=ref, attack_cache=attack_cache)
            except Exception as e:
                print(f"  [FAIL] run_suite failed for {name}: {e}")
                traceback.print_exc()
                res = {"model": name}
            finally:
                if RECORD_RUN_METRICS and monitor.metrics is not None:
                    run_metrics = [row for row in run_metrics if str(row.get("model")) != name]
                    run_metrics.append(monitor.metrics)
                    report_data.upsert_table(PERFORMANCE_CSV, pd.DataFrame([monitor.metrics]), ["model"])
            new_row = pd.DataFrame([res])
            df_results = report_data.upsert_table(RESULTS_CSV, new_row, ["model"])
            print("Result:")
            print(new_row.to_string(index=False))
            run_pending_epsilon_sweep(name, model)
            print("-" * 100)

        # Dropping this architecture-local registry (and cache) releases its
        # model family before the next architecture is constructed.
        del model_registry, attack_cache
        fp32 = int8_ptq = int4_ptq = int8_qat = int4_qat = None
        chaotic_int8_ptq = chaotic_int4_ptq = compressed_chaotic_int8 = None
        chaotic_int8_qat = model = ref = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if QUANTIZATION_DEBUG_ONLY:
        return
    if dither_rows:
        pd.DataFrame(dither_rows).to_csv(CHAOTIC_DITHER_SWEEP_CSV, index=False)
    if defense_summary_frames:
        pd.concat(defense_summary_frames, ignore_index=True).to_csv(
            defense_summary_csv_path(), index=False
        )
    print("\nFinal results:")
    print(df_results)
    adaptive_cols = [
        c
        for c in [
            "BPDA_PGD",
            "BPDA_Adaptive",
            "EOT_PGD",
            "Adaptive_Guardrail",
            "Adaptive_DetectGuard",
            "NES",
            "Boundary_acc",
            "AutoAttack",
        ]
        if c in df_results.columns
    ]
    if adaptive_cols:
        df_results["Worst_Robust_Acc"] = df_results[adaptive_cols].min(
            axis=1, skipna=True
        )
    if {"PGD", "Worst_Robust_Acc"}.issubset(df_results.columns):
        df_results["Gradient_Masking_Gap"] = (
            df_results["PGD"] - df_results["Worst_Robust_Acc"]
        )
    fp32_baseline = (
        df_results[df_results["model"].str.endswith("_FP32")]
        .assign(Architecture=lambda d: d["model"].str.replace("_FP32", "", regex=False))
        .set_index("Architecture")["Worst_Robust_Acc"]
    )
    df_results["Architecture"] = df_results["model"].str.replace(
        r"_(FP32|int8_PTQ|int4_PTQ|int8_QAT|int4_QAT).*", "", regex=True
    )
    df_results["FP32_Worst_Robust_Acc"] = df_results["Architecture"].map(fp32_baseline)
    if {
        "Worst_Robust_Acc",
        "FP32_Worst_Robust_Acc",
    }.issubset(df_results.columns):
        df_results["True_Robustness_Gain"] = (
            df_results["Worst_Robust_Acc"] - df_results["FP32_Worst_Robust_Acc"]
        )
    df_results = report_data.add_paired_tests(
        report_data.add_derived_metrics(df_results)
    )
    df_results.to_csv(RESULTS_CSV, index=False)
    if RUN_EPSILON_SWEEP:
        print("\nEpsilon sweep completed. Results saved to", SWEEP_CSV)

    report_data.generate_reports(report_data.DATA_DIR)
    print("All done.")


if __name__ == "__main__":
    main()
