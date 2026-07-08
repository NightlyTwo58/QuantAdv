"""Full model and defense evaluation suites."""
import json
import traceback

import pandas as pd
import torchattacks

from .config import *
from . import defense as dfn
from .paths import csv_path, json_path, defense_summary_csv_path
from .quantization import count_quant_layers
from .evaluation import sanity_check_accuracy
from .attacks import (
    make_torchattack,
    run_fgsm_pgd,
    run_autoattack,
    run_extra_whitebox_attacks,
    run_uap_attack,
    run_surrogate_attack,
    transfer_attack,
    transfer_attack_mim,
    transfer_uap_attack,
    run_random_noise_seeded,
    run_defense_adaptive_attacks,
    run_bpda,
    run_boundary_attack,
    run_nes_attack,
    unwrap_model,
)
from .diagnostics import (
    gradient_diagnostics,
    staircase_diagnostic,
    pgd_steps_ablation,
    pgd_trajectory_diagnostics,
    layerwise_grad_profile,
    run_quant_component_ablation,
    confidence_margin_diagnostic,
)


def run_suite(model, loader, name, fp32_ref=None, eps=DEFAULT_EPS):
    model.eval()
    results = {"model": name}

    def safe_set(key, fn, warning, default=None):
        try:
            results[key] = fn()
        except Exception as e:
            print(f"  [WARN] {warning} for {name}: {e}")
            results[key] = default

    def safe_update(fn, warning, defaults=None):
        try:
            results.update(fn())
        except Exception as e:
            print(f"  [WARN] {warning} for {name}: {e}")
            if defaults:
                for key, value in defaults.items():
                    results[key] = value() if callable(value) else value

    safe_set("clean_acc", lambda: sanity_check_accuracy(model, loader), "clean_acc failed")
    safe_update(
        lambda: run_fgsm_pgd(model, loader, eps=eps),
        "FGSM/PGD failed",
        {"FGSM": lambda: results.get("FGSM", None), "PGD": lambda: results.get("PGD", None)},
    )
    safe_set("AutoAttack", lambda: run_autoattack(model, loader, eps=eps), "AutoAttack failed")
    safe_update(lambda: run_extra_whitebox_attacks(model, loader, eps=eps), "CW/DeepFool/JSMA failed")
    safe_set("UAP", lambda: run_uap_attack(model, loader, eps=eps), "UAP attack failed")
    safe_set("Surrogate_Transfer", lambda: run_surrogate_attack(model, loader, eps=eps), "surrogate attack failed")

    if fp32_ref is not None:
        safe_set("Transfer_from_FP32", lambda: transfer_attack(fp32_ref, model, loader, eps=eps), "transfer_attack failed")
        safe_set("MIM_Transfer", lambda: transfer_attack_mim(fp32_ref, model, loader, eps=eps), "MIM transfer_attack failed")
        safe_set("UAP_Transfer", lambda: transfer_uap_attack(fp32_ref, model, loader, eps=eps), "UAP transfer_attack failed")

        if count_quant_layers(model) > 0:
            safe_set("Transfer_to_FP32", lambda: transfer_attack(model, fp32_ref, loader, eps=eps), "reverse transfer_attack failed")
            safe_set("MIM_Transfer_to_FP32", lambda: transfer_attack_mim(model, fp32_ref, loader, eps=eps), "reverse MIM transfer_attack failed")
            safe_set("UAP_Transfer_to_FP32", lambda: transfer_uap_attack(model, fp32_ref, loader, eps=eps), "reverse UAP transfer_attack failed")

    safe_update(lambda: run_random_noise_seeded(model, loader, eps=eps), "random_noise_attack failed", {"Random_Noise": None})

    try:
        results.update(run_defense_adaptive_attacks(model, loader, eps=eps))
    except Exception as e:
        print(f"  [WARN] adaptive defense attack failed for {name}: {e}")
        defense_model = unwrap_model(model)
        if isinstance(defense_model, dfn.SanitizedModel):
            results["BPDA_Adaptive"] = None
        elif isinstance(defense_model, dfn.SmoothedModel):
            results["EOT_PGD"] = None
        elif isinstance(defense_model, dfn.GuardrailModel):
            results["Adaptive_Guardrail"] = None
        elif isinstance(defense_model, dfn.DetectGuardModel):
            results["Adaptive_DetectGuard"] = None

    if count_quant_layers(model) > 0:
        safe_update(lambda: run_bpda(model, loader, eps=eps, n_restarts=BPDA_RESTARTS_SUITE), "BPDA failed", {"BPDA_PGD": None})
        safe_update(lambda: gradient_diagnostics(model, loader, fp32_ref=fp32_ref, max_batches=GRAD_DIAG_MAX_BATCHES), "gradient_diagnostics failed")
        safe_update(lambda: staircase_diagnostic(model, loader), "staircase_diagnostic failed")

        try:
            results.update(run_boundary_attack(model, loader, eps=eps, max_images=BOUNDARY_MAX_IMAGES_SUITE, steps=BOUNDARY_STEPS_SUITE, seed=BOUNDARY_SEED))
        except Exception as e:
            print(f"  [WARN] boundary_attack failed for {name}: {e}")
            results["Boundary_acc"] = None
            results["Boundary_mean_Linf"] = None
            results["Boundary_median_Linf"] = None
            results["Boundary_min_Linf"] = None
            results["Boundary_max_Linf"] = None
            results["Boundary_std_Linf"] = None
            results["Boundary_n"] = 0

        safe_update(
            lambda: run_nes_attack(model, loader, eps=eps, seeds=SEEDS, n_samples=NES_SAMPLES_SUITE, query_chunk=NES_QUERY_CHUNK),
            "NES attack failed",
            {"NES": None},
        )

        try:
            ablation = pgd_steps_ablation(model, loader, eps=eps)
            pd.DataFrame([{"model": name, "steps": k, "acc": v} for k, v in ablation.items()]) \
                .to_csv(csv_path(name, "ablation"), index=False)
        except Exception as e:
            print(f"  [WARN] pgd_steps_ablation failed for {name}: {e}")

        try:
            traj = pgd_trajectory_diagnostics(model, loader, eps=eps, max_batches=TRAJECTORY_MAX_BATCHES)
            with open(json_path(name, "trajectory"), "w") as f:
                json.dump(traj, f, indent=2)
        except Exception as e:
            print(f"  [WARN] pgd_trajectory_diagnostics failed for {name}: {e}")

        try:
            prof_hard = layerwise_grad_profile(model, loader, use_ste=False)
            prof_ste = layerwise_grad_profile(model, loader, use_ste=True)
            rows = [{"model": name, "layer": n, "grad_norm_hard": prof_hard.get(n),
                     "grad_norm_ste": prof_ste.get(n)} for n in prof_hard]
            pd.DataFrame(rows).to_csv(csv_path(name, "layerwise"), index=False)
        except Exception as e:
            print(f"  [WARN] layerwise_grad_profile failed for {name}: {e}")

        # weight-only / activation-only / both ablation
        try:
            rows = run_quant_component_ablation(model, loader, name, eps=eps)
            pd.DataFrame(rows).to_csv(csv_path(name, "component_ablation"), index=False)
        except Exception as e:
            print(f"  [WARN] run_quant_component_ablation failed for {name}: {e}")

        try:
            margins = confidence_margin_diagnostic(model, loader, eps=eps, max_batches=MARGIN_MAX_BATCHES)
            with open(json_path(name, "margin"), "w") as f:
                json.dump(margins, f)
        except Exception as e:
            print(f"  [WARN] confidence_margin_diagnostic failed for {name}: {e}")

    return results


def run_defense_suite(model_registry, finetune_loader, eval_loader):

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
