"""
Full per-model evaluation suite: runs the complete battery of clean-accuracy,
attack, and diagnostic checks for a single (model, fp32_ref) pair, and tracks
which QuantModel instance owns which quantized sub-model (needed by the
component-ablation diagnostic).
"""
import json

import pandas as pd

from .attacks import (
    transfer_attack,
    transfer_attack_mim,
    transfer_uap_attack,
    run_surrogate_attack,
    run_boundary_attack,
)
from .evaluation import (
    safe_run,
    sanity_check_accuracy,
    count_quant_layers,
    run_fgsm_pgd,
    run_autoattack,
    run_extra_whitebox_attacks,
    run_random_noise_seeded,
    run_bpda,
    run_nes_attack,
)
from .diagnostics import (
    gradient_diagnostics_and_layerwise_profile,
    staircase_diagnostic,
    pgd_steps_ablation,
    pgd_trajectory_diagnostics,
    run_quant_component_ablation,
)
from .paths import (
    layerwise_csv_path,
    ablation_csv_path,
    trajectory_json_path,
    component_ablation_csv_path,
)

# Resolve the QuantModel instance that owns a given sub-model.
# Populated by quantadv.experiment.main().
_model_to_qat_instance = {}


def _get_qat_instance_for_model(target_model):
    """
    Find the QuantModel instance that owns *target_model*.
    We store the mapping in a global registry built during main().
    """
    candidates = [target_model]
    module = getattr(target_model, "module", None)
    if module is not None:
        candidates.append(module)
    orig_mod = getattr(target_model, "_orig_mod", None)
    if orig_mod is not None:
        candidates.append(orig_mod)
    if module is not None and getattr(module, "_orig_mod", None) is not None:
        candidates.append(module._orig_mod)

    for candidate in candidates:
        qat_instance = _model_to_qat_instance.get(id(candidate))
        if qat_instance is not None:
            return qat_instance
    return None


def run_suite(model, loader, name, fp32_ref=None, eps=8 / 255):
    model.eval()
    results = {"model": name}

    clean_acc, _ = safe_run(lambda: sanity_check_accuracy(model, loader), name, "clean_acc")
    results["clean_acc"] = clean_acc

    fgsm_pgd, _ = safe_run(lambda: run_fgsm_pgd(model, loader, eps=eps), name, "FGSM/PGD")
    if fgsm_pgd is not None:
        results.update(fgsm_pgd)
    else:
        results["FGSM"] = results.get("FGSM", None)
        results["PGD"] = results.get("PGD", None)

    results["AutoAttack"], _ = safe_run(lambda: run_autoattack(model, loader, eps=eps), name, "AutoAttack")

    extra_whitebox, _ = safe_run(
        lambda: run_extra_whitebox_attacks(model, loader, eps=eps), name, "CW/DeepFool/JSMA")
    if extra_whitebox is not None:
        results.update(extra_whitebox)

    results["Surrogate_Transfer"], _ = safe_run(
        lambda: run_surrogate_attack(model, loader, eps=eps), name, "surrogate_attack")

    if fp32_ref is not None:
        results["Transfer_from_FP32"], _ = safe_run(
            lambda: transfer_attack(fp32_ref, model, loader, eps=eps), name, "transfer_attack")
        results["MIM_Transfer"], _ = safe_run(
            lambda: transfer_attack_mim(fp32_ref, model, loader, eps=eps), name, "transfer_attack_mim")
        results["UAP_Transfer"], _ = safe_run(
            lambda: transfer_uap_attack(fp32_ref, model, loader, eps=eps), name, "transfer_uap_attack")

    random_noise, _ = safe_run(lambda: run_random_noise_seeded(model, loader, eps=eps), name, "random_noise_attack")
    if random_noise is not None:
        results.update(random_noise)
    else:
        results["Random_Noise"] = None

    if count_quant_layers(model) > 0:
        bpda, _ = safe_run(
            lambda: run_bpda(model, loader, eps=eps, n_restarts=5, backward_model=fp32_ref), name, "BPDA")
        if bpda is not None:
            results.update(bpda)
        else:
            results["BPDA_PGD"] = None

        nes, _ = safe_run(lambda: run_nes_attack(model, loader, eps=eps), name, "NES")
        if nes is not None:
            results.update(nes)
        else:
            results["NES"] = None

        boundary, _ = safe_run(
            lambda: run_boundary_attack(model, loader, eps=eps, max_images=30, steps=200, seed=0),
            name, "boundary_attack")
        if boundary is not None:
            results.update(boundary)
        else:
            results["Boundary_acc"] = None
            results["Boundary_mean_Linf"] = None

        diag_result, _ = safe_run(
            lambda: gradient_diagnostics_and_layerwise_profile(model, loader, fp32_ref=fp32_ref, max_batches=5),
            name, "gradient_diagnostics_and_layerwise_profile")
        if diag_result is not None:
            diag, layer_profile = diag_result
            results.update(diag)
            rows = [{"model": name, "layer": n, "grad_norm_hard": v, "grad_norm_ste": v}
                    for n, v in layer_profile.items()]
            pd.DataFrame(rows).to_csv(layerwise_csv_path(name), index=False)

        staircase, _ = safe_run(lambda: staircase_diagnostic(model, loader), name, "staircase_diagnostic")
        if staircase is not None:
            results.update(staircase)

        ablation, _ = safe_run(lambda: pgd_steps_ablation(model, loader, eps=eps), name, "pgd_steps_ablation")
        if ablation is not None:
            pd.DataFrame([{"model": name, "steps": k, "acc": v} for k, v in ablation.items()]) \
                .to_csv(ablation_csv_path(name), index=False)

        traj, _ = safe_run(
            lambda: pgd_trajectory_diagnostics(model, loader, eps=eps, max_batches=5),
            name, "pgd_trajectory_diagnostics")
        if traj is not None:
            with open(trajectory_json_path(name), "w") as f:
                json.dump(traj, f, indent=2)

        def _component_ablation():
            qat_instance = _get_qat_instance_for_model(model)
            if qat_instance is not None:
                return run_quant_component_ablation(qat_instance, loader, name, eps=eps)
            return None

        component_rows, _ = safe_run(_component_ablation, name, "run_quant_component_ablation")
        if component_rows is not None:
            pd.DataFrame(component_rows).to_csv(component_ablation_csv_path(name), index=False)

    return results
