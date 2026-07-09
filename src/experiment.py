"""Experiment entrypoint orchestrating the active archive workflow."""

import importlib.util
import os
import traceback

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from .config import *
from .data import get_dataloaders, load_pretrained
from .quantization import (
    convert_to_quant,
    convert_to_chaotic_quant,
    prepare_qat,
    quantizable_layer_names,
    verify_quantization_layers,
)
from .attacks import with_image_compression
from .compute import parallelize
from .evaluation import sanity_check_accuracy, run_epsilon_sweep_for_model
from .diagnostics import run_chunk_quantization_attacks
from .suite import run_suite, run_defense_suite
from .paths import csv_path
from .plots import (
    plot_defense_comparison,
    plot_epsilon_sweep_curves,
    plot_pgd_steps_ablation,
    plot_pgd_trajectory,
    plot_layerwise_grad_profile,
    plot_component_ablation,
    plot_chunk_quantization_attacks,
    plot_gradient_masking_summary,
    plot_confidence_margin_diagnostic,
    plot_results_heatmap,
)


def check_environment():
    missing = [
        pkg
        for pkg in ("torchattacks", "autoattack", "pytorchcv")
        if importlib.util.find_spec(pkg) is None
    ]
    if missing:
        raise ImportError(
            f"Missing packages: {missing}.\nInstall via: pip install -r requirements.txt"
        )
    print("All required packages are available.")
    print("device:", device)
    if not os.path.isdir(CIFAR10_DIR):
        raise FileNotFoundError(f"Expected extracted CIFAR-10 at {CIFAR10_DIR!r}")


def main():
    check_environment()
    finetune_loader, eval_loader = get_dataloaders()
    model_registry = {}

    for arch_key in PRETRAINED_NAMES:
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
        except Exception as e:
            print(f"  [FAIL] could not load {arch_key}: {e}")
            traceback.print_exc()
            continue

        try:
            int8_ptq = convert_to_quant(
                fp32, bits=QAT_BITS, quant_weight=True, quant_act=True
            )
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
                bits=QAT_BITS,
                finetune_loader=finetune_loader,
                epochs=QAT_MAIN_EPOCHS,
            )
            verify_quantization_layers(
                arch_key, fp32, int8_qat, "int8 QAT", fp32_layer_names
            )
            model_registry[f"{arch_key}_int8_QAT"] = (int8_qat, fp32)
        except Exception as e:
            print(f"  [FAIL] int8 QAT for {arch_key}: {e}")
            traceback.print_exc()
            raise

        if QUANTIZATION_DEBUG_ONLY:
            print(
                "\nQUANTIZATION_DEBUG_ONLY=True; exiting before defenses, attacks, sweeps, and plots."
            )
            print("Registry built:", list(model_registry.keys()))
            return

        try:
            model_registry[f"{arch_key}_FP32_Compressed"] = (
                with_image_compression(fp32),
                fp32,
            )
        except Exception as e:
            print(f"  [FAIL] compressed FP32 for {arch_key}: {e}")

        try:
            chaotic_int8_ptq = convert_to_chaotic_quant(
                fp32, bits=QAT_BITS, quant_weight=True, quant_act=True
            )
            model_registry[f"{arch_key}_chaotic_int8_PTQ"] = (chaotic_int8_ptq, fp32)
        except Exception as e:
            print(f"  [FAIL] chaotic int8 PTQ for {arch_key}: {e}")

        try:
            chaotic_int4_ptq = convert_to_chaotic_quant(
                fp32, bits=4, quant_weight=True, quant_act=True
            )
            model_registry[f"{arch_key}_chaotic_int4_PTQ"] = (chaotic_int4_ptq, fp32)
        except Exception as e:
            print(f"  [FAIL] chaotic int4 PTQ for {arch_key}: {e}")

        try:
            compressed_chaotic_int8 = with_image_compression(
                convert_to_chaotic_quant(
                    fp32, bits=QAT_BITS, quant_weight=True, quant_act=True
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
                bits=QAT_BITS,
                finetune_loader=finetune_loader,
                epochs=QAT_MAIN_EPOCHS,
                chaotic=True,
            )
            model_registry[f"{arch_key}_chaotic_int8_QAT"] = (chaotic_int8_qat, fp32)
        except Exception as e:
            print(f"  [FAIL] chaotic int8 QAT for {arch_key}: {e}")
            traceback.print_exc()

    try:
        model_registry, df_defense_summary = run_defense_suite(
            model_registry, finetune_loader, eval_loader
        )
        if not df_defense_summary.empty:
            print(
                "\nDefense summary (guardrail/detector flag rates, certified accuracy):"
            )
            print(df_defense_summary.to_string(index=False))
    except Exception as e:
        print(f"  [FAIL] run_defense_suite failed: {e}")
        traceback.print_exc()

    chunk_model_names = []
    for arch_key in PRETRAINED_NAMES:
        entry = model_registry.get(f"{arch_key}_FP32")
        if entry is None:
            continue
        chunk_model_names.append(arch_key)
        out_path = csv_path(arch_key, "chunk_quant")
        if os.path.exists(out_path):
            print(f"Skipping chunk quantization for {arch_key} (already in {out_path})")
            continue
        print(f"\nChunk quantization sweep for {arch_key} ...")
        try:
            rows = run_chunk_quantization_attacks(
                entry[0],
                eval_loader,
                arch_key,
                bits=QAT_BITS,
                n_chunks=CHUNK_QUANT_NUM_CHUNKS,
                eps=DEFAULT_EPS,
            )
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"Chunk quantization results saved to {out_path}")
        except Exception as e:
            print(f"  [FAIL] chunk quantization sweep failed for {arch_key}: {e}")
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
        r"_(FP32|int8_PTQ|int4_PTQ|int8_QAT).*", "", regex=True
    )

    df_results["FP32_Worst_Robust_Acc"] = df_results["Architecture"].map(fp32_baseline)

    if {
        "Worst_Robust_Acc",
        "FP32_Worst_Robust_Acc",
    }.issubset(df_results.columns):
        df_results["True_Robustness_Gain"] = (
            df_results["Worst_Robust_Acc"] - df_results["FP32_Worst_Robust_Acc"]
        )

    df_results.to_csv(RESULTS_CSV, index=False)

    acc_cols = [
        c
        for c in [
            "clean_acc",
            "FGSM",
            "PGD",
            "CW",
            "DeepFool",
            "JSMA",
            "AutoAttack",
            "Transfer_from_FP32",
            "MIM_Transfer",
            "UAP_Transfer",
            "Transfer_to_FP32",
            "MIM_Transfer_to_FP32",
            "UAP_Transfer_to_FP32",
            "Surrogate_Transfer",
            "Random_Noise",
            "BPDA_PGD",
            "BPDA_Adaptive",
            "EOT_PGD",
            "Adaptive_Guardrail",
            "Adaptive_DetectGuard",
        ]
        if c in df_results.columns
    ]

    if len(acc_cols) > 0:
        df_plot = df_results.melt(
            id_vars="model",
            value_vars=acc_cols,
            var_name="Attack",
            value_name="Accuracy",
        )

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
        sweep_done = set(
            zip(df_sweep["model"].astype(str), df_sweep["epsilon"].round(6))
        )
    else:
        df_sweep = pd.DataFrame()
        sweep_done = set()

    for name, (model, ref) in model_registry.items():
        print(f"\nSweeping {name} ...")
        pending_eps = [
            eps for eps in SWEEP_EPSILONS if (name, round(eps, 6)) not in sweep_done
        ]
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
        plot_chunk_quantization_attacks(chunk_model_names)
    except Exception as e:
        print(f"  [WARN] plot_chunk_quantization_attacks failed: {e}")

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
