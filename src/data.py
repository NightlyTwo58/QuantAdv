#!/usr/bin/env python3
"""
Shared result collation, derivation, and visualization for QuantAdv experiments.

Reads individual per-model CSV/JSON files from DATA_DIR, writes combined CSVs,
and regenerates the summary PNG plots without loading models or running attacks.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
import stats as qstats


def _model_from_prefixed_path(path: Path, prefix: str, suffix: str) -> str:
    """Extract the model name encoded in a result filename."""
    name = path.name
    if name.startswith(prefix):
        name = name[len(prefix) :]
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def _name(path) -> str:
    """Return a display-safe stem for a path-like value."""
    return Path(path).name


def _read_csvs(paths: Iterable[Path], model_prefix: str | None = None) -> pd.DataFrame:
    """Read and concatenate matching CSV files with model metadata attached."""
    frames = []
    for path in sorted(paths):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] could not read {path}: {exc}")
            continue

        if model_prefix and "model" not in df.columns:
            df.insert(0, "model", _model_from_prefixed_path(path, model_prefix, ".csv"))

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    if "model" in out.columns:
        out = out.drop_duplicates(
            subset=[
                c
                for c in ["model", "epsilon", "steps", "layer", "config"]
                if c in out.columns
            ],
            keep="last",
        )
    return out


def _write(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Create parent directories and write a dataframe to CSV."""
    if df is not None and not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows)")
    return df


def _result_files(data_dir: Path) -> list[Path]:
    """Collect result-file paths by experiment family."""
    blocked = {
        _name(RESULTS_CSV),
        _name(SWEEP_CSV),
        _name(ABLATION_COMBINED_CSV),
        _name(LAYERWISE_COMBINED_CSV),
        _name(TRAJECTORY_COMBINED_CSV),
        _name(COMPONENT_ABLATION_COMBINED_CSV),
        _name(MARGIN_COMBINED_CSV),
        "defense_summary.csv",
    }
    files = []
    for path in data_dir.glob("results_*.csv"):
        if path.name in blocked:
            continue
        if path.name.startswith("results_sweep_"):
            continue
        files.append(path)
    return files


def combine_scalar_results(data_dir: Path, output: Path = RESULTS_CSV) -> pd.DataFrame:
    """Combine per-model scalar result CSV files into one table."""
    files = _result_files(data_dir)
    df = _read_csvs(files, model_prefix="results_")
    return _write(df, output)


def combine_sweeps(data_dir: Path, output: Path = SWEEP_CSV) -> pd.DataFrame:
    """Combine epsilon-sweep CSV files into one table."""
    files = [p for p in data_dir.glob("sweep_*.csv") if p.name != output.name]
    df = _read_csvs(files, model_prefix="sweep_")
    if not df.empty and {"model", "epsilon"}.issubset(df.columns):
        df = df.sort_values(["model", "epsilon"]).reset_index(drop=True)
    return _write(df, output)


def combine_ablation(
    data_dir: Path, output: Path = ABLATION_COMBINED_CSV
) -> pd.DataFrame:
    """Combine PGD step-ablation CSV files into one table."""
    files = [p for p in data_dir.glob("ablation_*.csv") if p.name != output.name]
    df = _read_csvs(files, model_prefix="ablation_")
    if not df.empty and {"model", "steps"}.issubset(df.columns):
        df = df.sort_values(["model", "steps"]).reset_index(drop=True)
    return _write(df, output)


def combine_layerwise(
    data_dir: Path, output: Path = LAYERWISE_COMBINED_CSV
) -> pd.DataFrame:
    """Combine layerwise gradient-profile CSV files into one table."""
    files = [p for p in data_dir.glob("layerwise_*.csv") if p.name != output.name]
    df = _read_csvs(files, model_prefix="layerwise_")
    return _write(df, output)


def combine_component_ablation(
    data_dir: Path, output: Path = COMPONENT_ABLATION_COMBINED_CSV
) -> pd.DataFrame:
    """Combine quantization component-ablation CSV files into one table."""
    files = [
        p for p in data_dir.glob("component_ablation_*.csv") if p.name != output.name
    ]
    df = _read_csvs(files, model_prefix="component_ablation_")
    return _write(df, output)


def combine_trajectories(
    data_dir: Path, output: Path = TRAJECTORY_COMBINED_CSV
) -> pd.DataFrame:
    """Combine PGD trajectory diagnostic CSV files into one table."""
    rows = []
    for path in sorted(data_dir.glob("trajectory_*.json")):
        model = _model_from_prefixed_path(path, "trajectory_", ".json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                traj = json.load(f)
        except Exception as exc:
            print(f"[WARN] could not read {path}: {exc}")
            continue

        grad = traj.get("grad_norm_per_step", []) or []
        move = traj.get("movement_from_random_start_per_step", []) or []
        n = max(len(grad), len(move))
        for i in range(n):
            rows.append(
                {
                    "model": model,
                    "step": i + 1,
                    "grad_norm_per_step": grad[i] if i < len(grad) else np.nan,
                    "movement_from_random_start_per_step": (
                        move[i] if i < len(move) else np.nan
                    ),
                }
            )

    return _write(pd.DataFrame(rows), output)


def combine_margins(data_dir: Path, output: Path = MARGIN_COMBINED_CSV) -> pd.DataFrame:
    """Combine confidence-margin diagnostic CSV files into one table."""
    rows = []
    for path in sorted(data_dir.glob("margin_*.json")):
        model = _model_from_prefixed_path(path, "margin_", ".json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                margins = json.load(f)
        except Exception as exc:
            print(f"[WARN] could not read {path}: {exc}")
            continue

        for kind, values in (
            ("clean", margins.get("clean_margins", [])),
            ("adv", margins.get("adv_margins", [])),
        ):
            for i, value in enumerate(values):
                rows.append({"model": model, "kind": kind, "index": i, "margin": value})

    return _write(pd.DataFrame(rows), output)


def _model_names(*dfs: pd.DataFrame) -> list[str]:
    """Return the preferred display order for model names."""
    names = []
    for df in dfs:
        if df is not None and not df.empty and "model" in df.columns:
            names.extend(df["model"].dropna().astype(str).tolist())
    return list(dict.fromkeys(names))


def plot_summary_results(df_results: pd.DataFrame, output: Path = PLOT_PNG) -> None:
    """Plot the headline accuracy and robustness summary metrics."""
    if df_results is None or df_results.empty or "model" not in df_results.columns:
        return

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
            "Surrogate_Transfer",
            "Random_Noise",
            "BPDA_PGD",
            "NES",
            "Boundary_acc",
        ]
        if c in df_results.columns and df_results[c].notna().any()
    ]

    if not acc_cols:
        return

    df_plot = df_results.melt(
        id_vars="model", value_vars=acc_cols, var_name="Attack", value_name="Accuracy"
    )
    df_plot = df_plot.dropna(subset=["Accuracy"])
    if df_plot.empty:
        return

    plt.figure(figsize=SUMMARY_PLOT_FIGSIZE)
    sns.barplot(data=df_plot, x="model", y="Accuracy", hue="Attack")
    plt.xticks(rotation=SUMMARY_XTICK_ROTATION, ha="right")
    plt.title("Model Accuracy under Various Adversarial Attacks")
    plt.ylim(0, PLOT_MAX_ACCURACY)
    plt.grid(axis="y", linestyle="--", alpha=SUMMARY_GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close()
    print(f"wrote {output}")


def plot_epsilon_sweep_curves(
    df_sweep: pd.DataFrame, output: Path = SWEEP_PLOT_PNG
) -> None:
    """Plot robustness metrics as a function of attack epsilon.
    """
    if df_sweep is None or df_sweep.empty:
        return
    value_cols = [
        c
        for c in ["PGD_acc", "PGD_ste_acc", "Random_Noise_acc", "BPDA_acc"]
        if c in df_sweep.columns
    ]
    if not value_cols:
        return

    df_long = df_sweep.melt(
        id_vars=["model", "epsilon"],
        value_vars=value_cols,
        var_name="Attack",
        value_name="Accuracy",
    )
    df_long = df_long.dropna(subset=["Accuracy"])
    if df_long.empty:
        return

    models = df_long["model"].dropna().astype(str).unique()
    cols = min(SWEEP_PLOT_COLS_MAX, len(models))
    rows = int(math.ceil(len(models) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(SWEEP_PLOT_WIDTH * cols, SWEEP_PLOT_HEIGHT * rows),
        squeeze=False,
    )

    for i, model in enumerate(models):
        ax = axes[i // cols][i % cols]
        sns.lineplot(
            data=df_long[df_long["model"] == model],
            x="epsilon",
            y="Accuracy",
            hue="Attack",
            marker="o",
            ax=ax,
        )
        ax.set_title(model)
        ax.set_ylim(0, PLOT_MAX_ACCURACY)
        ax.grid(linestyle="--", alpha=PLOT_GRID_ALPHA)

    for j in range(len(models), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Accuracy vs Perturbation Budget (Epsilon Sweep)")
    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close(fig)
    print(f"wrote {output}")


def plot_pgd_steps_ablation(
    df_ablation: pd.DataFrame, output: Path = ABLATION_PLOT_PNG
) -> None:
    """Plot accuracy as PGD step count changes."""
    if (
        df_ablation is None
        or df_ablation.empty
        or not {"steps", "acc", "model"}.issubset(df_ablation.columns)
    ):
        return

    plt.figure(figsize=ABLATION_FIGSIZE)
    sns.lineplot(data=df_ablation, x="steps", y="acc", hue="model", marker="o")
    plt.title("PGD Accuracy vs Number of Steps (Gradient Masking Check)")
    plt.xlabel("PGD steps")
    plt.ylabel("Accuracy")
    plt.ylim(0, PLOT_MAX_ACCURACY)
    plt.grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close()
    print(f"wrote {output}")


def plot_pgd_trajectory(
    df_traj: pd.DataFrame, output: Path = TRAJECTORY_PLOT_PNG
) -> None:
    """Plot gradient norms and movement along PGD trajectories."""
    required = {
        "model",
        "step",
        "grad_norm_per_step",
        "movement_from_random_start_per_step",
    }
    if df_traj is None or df_traj.empty or not required.issubset(df_traj.columns):
        return

    fig, axes = plt.subplots(1, 2, figsize=TRAJECTORY_FIGSIZE)
    for model, group in df_traj.groupby("model", sort=False):
        group = group.sort_values("step")
        axes[0].plot(
            group["step"], group["grad_norm_per_step"], marker="o", label=model
        )
        axes[1].plot(
            group["step"],
            group["movement_from_random_start_per_step"],
            marker="o",
            label=model,
        )

    axes[0].set_title("Gradient Norm per PGD Step")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Grad Norm")
    if (pd.to_numeric(df_traj["grad_norm_per_step"], errors="coerce") > 0).any():
        axes[0].set_yscale("log")
    axes[0].grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    axes[0].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    axes[1].set_title("Perturbation Movement per PGD Step")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Linf Movement from Random Start")
    axes[1].grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    axes[1].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close(fig)
    print(f"wrote {output}")


def plot_layerwise_grad_profile(
    df_layer: pd.DataFrame, output: Path = LAYERWISE_PLOT_PNG
) -> None:
    """Plot per-layer gradient magnitudes for diagnosed models."""
    required = {"model", "layer", "grad_norm_hard", "grad_norm_ste"}
    if df_layer is None or df_layer.empty or not required.issubset(df_layer.columns):
        return

    models = df_layer["model"].dropna().astype(str).unique()
    cols = min(LAYERWISE_PLOT_COLS_MAX, len(models))
    rows = int(math.ceil(len(models) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(LAYERWISE_PLOT_WIDTH * cols, LAYERWISE_PLOT_HEIGHT * rows),
        squeeze=False,
    )

    for i, model in enumerate(models):
        df = df_layer[df_layer["model"] == model].reset_index(drop=True)
        ax = axes[i // cols][i % cols]
        x = np.arange(len(df))
        ax.plot(x, df["grad_norm_hard"], marker="o", label="hard-round")
        ax.plot(x, df["grad_norm_ste"], marker="o", label="STE")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(
            df["layer"],
            rotation=LAYERWISE_XTICK_ROTATION,
            fontsize=LAYERWISE_XTICK_FONT_SIZE,
        )
        ax.set_title(model)
        ax.set_ylabel("Grad Norm (log)")
        ax.legend(fontsize=PLOT_LEGEND_FONT_SIZE)
        ax.grid(linestyle="--", alpha=PLOT_GRID_ALPHA)

    for j in range(len(models), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Layerwise Gradient Norms: Hard-Round vs STE")
    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close(fig)
    print(f"wrote {output}")


def plot_component_ablation(
    df_component: pd.DataFrame, output: Path = COMPONENT_ABLATION_PLOT_PNG
) -> None:
    """Plot clean, hard-PGD, STE-PGD, and BPDA component-ablation accuracies.

    ``PGD_ste_acc`` is a budget-matched STE companion to ``PGD_acc``
    (``PGD_hard_acc``) added alongside ``BPDA_acc``. Expect ``weight_only``
    to show little gap between ``PGD_acc`` and ``PGD_ste_acc`` (weight
    rounding doesn't sit on the gradient path back to the input) and
    ``act_only``/``both`` to show a large one (activation rounding does).
    """
    required = {"model", "config", "clean_acc", "PGD_acc"}
    if (
        df_component is None
        or df_component.empty
        or not required.issubset(df_component.columns)
    ):
        return

    value_vars = [
        c
        for c in ["clean_acc", "PGD_acc", "PGD_ste_acc", "BPDA_acc"]
        if c in df_component.columns
    ]
    df_long = df_component.melt(
        id_vars=["model", "config"],
        value_vars=value_vars,
        var_name="Metric",
        value_name="Accuracy",
    ).dropna(subset=["Accuracy"])

    if df_long.empty:
        return

    g = sns.catplot(
        data=df_long,
        x="config",
        y="Accuracy",
        hue="Metric",
        col="model",
        kind="bar",
        col_wrap=COMPONENT_ABLATION_COL_WRAP,
        height=COMPONENT_ABLATION_HEIGHT,
        sharey=True,
    )
    g.set_titles("{col_name}")
    g.set(ylim=(0, PLOT_MAX_ACCURACY))
    g.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close(g.fig)
    print(f"wrote {output}")


def plot_gradient_masking_summary(
    df_results: pd.DataFrame, output: Path = MASKING_SUMMARY_PLOT_PNG
) -> None:
    """Plot direct evidence of gradient masking under hard-round quantization.
    """
    if (
        df_results is None
        or df_results.empty
        or "model" not in df_results.columns
        or "frac_zero_grad_hard" not in df_results.columns
    ):
        return

    df = df_results.dropna(subset=["frac_zero_grad_hard"]).copy()
    if df.empty:
        return

    has_norms = {"grad_norm_hard", "grad_norm_ste"}.issubset(df.columns)
    fig, axes = plt.subplots(
        1, 2 if has_norms else 1, figsize=MASKING_SUMMARY_FIGSIZE, squeeze=False
    )
    axes = axes[0]

    sns.barplot(data=df, x="model", y="frac_zero_grad_hard", ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=MASKING_BASELINE_LINEWIDTH)
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", labelrotation=SUMMARY_XTICK_ROTATION)
    axes[0].set_title("Fraction of Zero Input Gradients (Hard Round)")
    axes[0].set_ylabel("frac_zero_grad_hard")
    axes[0].grid(axis="y", linestyle="--", alpha=PLOT_GRID_ALPHA)

    if has_norms:
        df_norm = df.dropna(subset=["grad_norm_hard", "grad_norm_ste"])
        if not df_norm.empty:
            long = df_norm.melt(
                id_vars="model",
                value_vars=["grad_norm_hard", "grad_norm_ste"],
                var_name="Regime",
                value_name="Gradient Norm",
            )
            sns.barplot(
                data=long, x="model", y="Gradient Norm", hue="Regime", ax=axes[1]
            )
            axes[1].set_yscale("log")
            axes[1].tick_params(axis="x", labelrotation=SUMMARY_XTICK_ROTATION)
            axes[1].set_title("Input Gradient Norm: Hard-Round vs STE")
            axes[1].grid(axis="y", linestyle="--", alpha=PLOT_GRID_ALPHA)
        else:
            axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close(fig)
    print(f"wrote {output}")


def plot_confidence_margin_diagnostic(
    df_margins: pd.DataFrame, output: Path = MARGIN_PLOT_PNG
) -> None:
    """Plot clean and adversarial confidence-margin distributions."""
    required = {"model", "kind", "margin"}
    if (
        df_margins is None
        or df_margins.empty
        or not required.issubset(df_margins.columns)
    ):
        return

    models = df_margins["model"].dropna().astype(str).unique()
    cols = min(MARGIN_PLOT_COLS_MAX, len(models))
    rows = int(math.ceil(len(models) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(MARGIN_PLOT_WIDTH * cols, MARGIN_PLOT_HEIGHT * rows),
        squeeze=False,
    )

    for i, model in enumerate(models):
        ax = axes[i // cols][i % cols]
        group = df_margins[df_margins["model"] == model]
        clean = group[group["kind"] == "clean"]["margin"].dropna()
        adv = group[group["kind"] == "adv"]["margin"].dropna()
        if not clean.empty:
            ax.hist(
                clean,
                bins=MARGIN_HIST_BINS,
                alpha=MARGIN_HIST_ALPHA,
                label="clean",
                density=True,
            )
        if not adv.empty:
            ax.hist(
                adv,
                bins=MARGIN_HIST_BINS,
                alpha=MARGIN_HIST_ALPHA,
                label="PGD-adv",
                density=True,
            )
        ax.set_title(model)
        ax.set_xlabel("Top1 - Top2 Softmax Margin")
        ax.legend(fontsize=PLOT_LEGEND_FONT_SIZE)
        ax.grid(linestyle="--", alpha=PLOT_GRID_ALPHA)

    for j in range(len(models), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle("Confidence Margin: Clean vs PGD-Adversarial")
    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close(fig)
    print(f"wrote {output}")


def plot_results_heatmap(
    df_results: pd.DataFrame, output: Path = HEATMAP_PLOT_PNG
) -> None:
    """Plot scalar metrics as a model-by-metric heatmap."""
    if df_results is None or df_results.empty or "model" not in df_results.columns:
        return

    candidate_cols = [
        "clean_acc",
        "FGSM",
        "PGD",
        "AutoAttack",
        "CW",
        "DeepFool",
        "JSMA",
        "Surrogate_Transfer",
        "Transfer_from_FP32",
        "MIM_Transfer",
        "UAP_Transfer",
        "Random_Noise",
        "BPDA_PGD",
        "NES",
        "Boundary_acc",
    ]
    cols = [
        c
        for c in candidate_cols
        if c in df_results.columns and df_results[c].notna().any()
    ]
    if not cols:
        return

    df_heat = df_results.set_index("model")[cols].astype(float)
    plt.figure(
        figsize=(
            max(HEATMAP_MIN_WIDTH, len(cols)),
            max(HEATMAP_MIN_HEIGHT, len(df_heat) * HEATMAP_ROW_HEIGHT),
        )
    )
    sns.heatmap(
        df_heat,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        linewidths=HEATMAP_LINEWIDTHS,
    )
    plt.title("Full Results Heatmap: Models vs Attacks")
    plt.tight_layout()
    plt.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close()
    print(f"wrote {output}")


def add_sweep_masking_metrics(df_sweep: pd.DataFrame) -> pd.DataFrame:
    """Add an explicit masking-gap column to the epsilon-sweep table.
    """
    if df_sweep is None or df_sweep.empty:
        return df_sweep
    if {"PGD_acc", "PGD_ste_acc"}.issubset(df_sweep.columns):
        df_sweep = df_sweep.copy()
        df_sweep["PGD_masking_gap"] = pd.to_numeric(
            df_sweep["PGD_acc"], errors="coerce"
        ) - pd.to_numeric(df_sweep["PGD_ste_acc"], errors="coerce")
    return df_sweep


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived robustness and masking metrics to a results table."""
    if df is None or df.empty or "model" not in df:
        return df
    df = df.copy()
    robust = [
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
        if c in df
    ]
    if robust:
        df["General_Robustness"] = (
            df[robust].apply(pd.to_numeric, errors="coerce").min(axis=1)
        )
        df["Worst_Robust_Acc"] = df["General_Robustness"]
    if {"clean_acc", "General_Robustness"}.issubset(df):
        clean = pd.to_numeric(df["clean_acc"], errors="coerce").replace(0, np.nan)
        df["General_Robustness_Retention"] = df["General_Robustness"] / clean
    if {"PGD", "General_Robustness"}.issubset(df):
        gap = (
            pd.to_numeric(df["PGD"], errors="coerce") - df["General_Robustness"]
        ).clip(lower=0)
        scale = (
            pd.to_numeric(df["clean_acc"], errors="coerce").replace(0, np.nan)
            if "clean_acc" in df
            else pd.Series(1.0, index=df.index)
        )
        df["General_Masking_Score"] = (gap / scale).clip(0, 1)

    architecture = (
        df["model"]
        .astype(str)
        .str.replace(
            r"_(FP32|int8_PTQ|int4_PTQ|int8_QAT|int4_QAT).*", "", regex=True
        )
    )
    df["Architecture"] = architecture
    if "Worst_Robust_Acc" in df:
        baseline = (
            df[df["model"].astype(str).str.endswith("_FP32")]
            .assign(Architecture=architecture)
            .drop_duplicates("Architecture", keep="last")
            .set_index("Architecture")["Worst_Robust_Acc"]
        )
        df["FP32_Worst_Robust_Acc"] = architecture.map(baseline)
        df["True_Robustness_Gain"] = (
            df["Worst_Robust_Acc"] - df["FP32_Worst_Robust_Acc"]
        )
    return df


def add_paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Add paired McNemar tests between each model variant and FP32 baseline."""
    return qstats.add_paired_mcnemar_tests(
        df, baseline_name=qstats.fp32_baseline_name
    )


def _read_if_present(path: Path) -> pd.DataFrame:
    """Read a CSV file when it exists, otherwise return an empty table."""
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"[WARN] could not read {path}: {exc}")
        return pd.DataFrame()


def read_table(path) -> pd.DataFrame:
    """Read an existing CSV table or return an empty table."""
    return _read_if_present(Path(path))


def upsert_table(path, rows: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    """Merge rows into a keyed CSV table and persist the result."""
    path = Path(path)
    combined = _merge_frames([_read_if_present(path), rows], list(keys))
    return _write(combined, path)


def _merge_frames(frames: Iterable[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    """Merge multiple dataframes on shared key columns."""
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    subset = [key for key in keys if key in result]
    return (
        result.drop_duplicates(subset=subset, keep="last")
        if subset
        else result.drop_duplicates()
    )


def _combine_csv_family(
    data_dir: Path, output: Path, patterns: Iterable[str], keys: list[str]
) -> pd.DataFrame:
    """Combine a family of CSV files using filename patterns and keys."""
    paths = {
        path.resolve()
        for pattern in patterns
        for path in data_dir.glob(pattern)
        if path.resolve() != output.resolve()
    }
    frames = [
        _read_if_present(output),
        *(_read_if_present(path) for path in sorted(paths)),
    ]
    return _write(_merge_frames(frames, keys), output)


def _plot_metric_grid(
    df: pd.DataFrame, title: str, output: Path, id_columns=("model",), page_size=12
) -> None:
    """Render a paginated grid of metric plots for a dataframe."""
    if df is None or df.empty:
        return
    numeric = [
        column
        for column in df.select_dtypes(include=np.number)
        if df[column].notna().any()
    ]
    if not numeric:
        return
    ids = [c for c in id_columns if c in df]
    if len(df) > 60:
        df = (
            df.groupby(ids, dropna=False, as_index=False)[numeric].mean()
            if ids
            else pd.DataFrame([df[numeric].mean()])
        )
    labels = df[ids].astype(str).agg(" | ".join, axis=1)
    if labels.empty:
        labels = pd.Series(df.index.astype(str), index=df.index)
    output.parent.mkdir(parents=True, exist_ok=True)
    for page, start in enumerate(range(0, len(numeric), page_size), 1):
        columns = numeric[start : start + page_size]
        values = df[columns].apply(pd.to_numeric, errors="coerce")
        scaled = (values - values.min()) / (values.max() - values.min()).replace(0, 1)
        height = max(3.0, min(18.0, 0.35 * len(values) + 2))
        fig, ax = plt.subplots(figsize=(max(7.0, 0.8 * len(columns)), height))
        sns.heatmap(
            scaled, cmap="viridis", mask=values.isna(), yticklabels=labels, ax=ax
        )
        ax.set_title(f"{title} ({start + 1}-{start + len(columns)} of {len(numeric)})")
        ax.set_xlabel("Metric (column-normalized)")
        ax.set_ylabel("")
        page_output = (
            output
            if len(numeric) <= page_size
            else output.with_name(f"{output.stem}_{page:02d}{output.suffix}")
        )
        fig.savefig(page_output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
        plt.close(fig)


def _plot_model_subgraphs(df: pd.DataFrame, table_name: str, output_dir: Path) -> None:
    """Plot model-level subgraphs for a combined result table."""
    if df is None or df.empty or "model" not in df or df["model"].nunique() < 2:
        return
    subgraph_dir = output_dir / "subgraphs" / table_name
    for model, group in df.groupby("model", sort=False):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model))
        _plot_metric_grid(group, str(model), subgraph_dir / f"{safe}.png")


def plot_performance(df: pd.DataFrame, output: Path = PERFORMANCE_PLOT_PNG) -> None:
    """Plot performance and resource metrics for evaluated models."""
    columns = [
        c
        for c in (
            "run_seconds",
            "rss_peak_mib",
            "cuda_allocated_peak_mib",
            "average_cpu_cores_used",
        )
        if c in df
    ]
    if df is None or df.empty or not columns:
        return
    long = df.melt(
        id_vars="model", value_vars=columns, var_name="metric", value_name="value"
    ).dropna()
    grid = sns.catplot(
        data=long,
        x="model",
        y="value",
        col="metric",
        kind="bar",
        col_wrap=2,
        sharey=False,
    )
    grid.set_xticklabels(rotation=45, ha="right")
    grid.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close(grid.fig)


def plot_dither_sweep(df: pd.DataFrame, output: Path = DITHER_PLOT_PNG) -> None:
    """Plot chaotic-dither sweep results."""
    values = [c for c in ("clean_acc", "PGD_acc") if c in df]
    if df is None or df.empty or "dither_amplitude" not in df or not values:
        return
    long = df.melt(
        id_vars=["model", "dither_amplitude"],
        value_vars=values,
        var_name="metric",
        value_name="accuracy",
    )
    grid = sns.relplot(
        data=long,
        x="dither_amplitude",
        y="accuracy",
        hue="metric",
        col="model",
        col_wrap=3,
        kind="line",
        marker="o",
    )
    grid.set(ylim=(0, 1))
    grid.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close(grid.fig)


def plot_chunk_quantization_attacks(
    df: pd.DataFrame, output: Path = CHUNK_QUANT_PLOT_PNG
) -> None:
    """Plot chunk-wise quantization attack results."""
    if df is None or df.empty:
        return
    x = next(
        (c for c in ("chunk_id", "chunk", "chunk_index", "layer_start") if c in df),
        None,
    )
    values = [c for c in ("clean_acc", "PGD_acc") if c in df]
    if x is None or not values:
        return
    ids = [c for c in ("model", x) if c in df]
    long = df.melt(
        id_vars=ids, value_vars=values, var_name="metric", value_name="accuracy"
    )
    grid = sns.relplot(
        data=long,
        x=x,
        y="accuracy",
        hue="metric",
        col="model",
        col_wrap=3,
        kind="line",
        marker="o",
    )
    grid.set(ylim=(0, 1))
    grid.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close(grid.fig)


def plot_defense_comparison(df: pd.DataFrame, output: Path = DEFENSE_PLOT_PNG) -> None:
    """Plot defense accuracy and attack-resistance comparisons."""
    if df is None or df.empty:
        return
    model_col = next((c for c in ("model", "defense", "name") if c in df), None)
    values = [
        c
        for c in df.select_dtypes(include=np.number)
        if "rate" in c.lower() or "acc" in c.lower()
    ]
    if model_col is None or not values:
        return
    long = df.melt(
        id_vars=model_col, value_vars=values, var_name="metric", value_name="value"
    ).dropna()
    grid = sns.catplot(
        data=long,
        x=model_col,
        y="value",
        hue="metric",
        kind="bar",
        height=5,
        aspect=1.8,
    )
    grid.set_xticklabels(rotation=45, ha="right")
    grid.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close(grid.fig)


def combine_all(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Combine all available experiment result families."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    result_path = data_dir / Path(RESULTS_CSV).name
    df_results = add_paired_tests(
        add_derived_metrics(
            _combine_csv_family(
                data_dir,
                result_path,
                ("results_*.csv", "accuracyresult*.csv"),
                ["model"],
            )
        )
    )
    _write(df_results, result_path)
    df_sweep = _combine_csv_family(
        data_dir,
        data_dir / Path(SWEEP_CSV).name,
        ("sweep_*.csv", "sweepresult*.csv"),
        ["model", "epsilon"],
    )
    df_sweep = add_sweep_masking_metrics(df_sweep)
    df_ablation = combine_ablation(
        data_dir, data_dir / Path(ABLATION_COMBINED_CSV).name
    )
    df_layer = combine_layerwise(data_dir, data_dir / Path(LAYERWISE_COMBINED_CSV).name)
    df_component = combine_component_ablation(
        data_dir, data_dir / Path(COMPONENT_ABLATION_COMBINED_CSV).name
    )
    df_traj = combine_trajectories(
        data_dir, data_dir / Path(TRAJECTORY_COMBINED_CSV).name
    )
    df_margins = combine_margins(data_dir, data_dir / Path(MARGIN_COMBINED_CSV).name)
    df_performance = _combine_csv_family(
        data_dir,
        data_dir / Path(PERFORMANCE_CSV).name,
        ("performance_metrics*.csv",),
        ["model"],
    )
    df_dither = _combine_csv_family(
        data_dir,
        data_dir / Path(CHAOTIC_DITHER_SWEEP_CSV).name,
        ("chaotic_dither_sweep*.csv",),
        ["model", "bits", "dither_amplitude"],
    )
    df_chunk = _combine_csv_family(
        data_dir,
        data_dir / Path(CHUNK_COMBINED_CSV).name,
        ("chunk_quant_*.csv",),
        ["model", "chunk_id"],
    )
    df_defense = _read_if_present(data_dir / Path(DEFENSE_SUMMARY_CSV).name)

    return {
        "results": df_results,
        "sweep": df_sweep,
        "ablation": df_ablation,
        "layerwise": df_layer,
        "component_ablation": df_component,
        "trajectory": df_traj,
        "margins": df_margins,
        "performance": df_performance,
        "dither": df_dither,
        "chunk_quant": df_chunk,
        "defense": df_defense,
    }


def plot_all(dfs: dict[str, pd.DataFrame], output_dir: Path = DATA_DIR) -> None:
    """Generate all summary plots from combined result tables."""
    output_dir = Path(output_dir)
    plot_summary_results(dfs["results"], output_dir / Path(PLOT_PNG).name)
    plot_epsilon_sweep_curves(dfs["sweep"], output_dir / _name(SWEEP_PLOT_PNG))
    plot_pgd_steps_ablation(dfs["ablation"], output_dir / _name(ABLATION_PLOT_PNG))
    plot_pgd_trajectory(dfs["trajectory"], output_dir / _name(TRAJECTORY_PLOT_PNG))
    plot_layerwise_grad_profile(
        dfs["layerwise"], output_dir / _name(LAYERWISE_PLOT_PNG)
    )
    plot_component_ablation(
        dfs["component_ablation"], output_dir / _name(COMPONENT_ABLATION_PLOT_PNG)
    )
    plot_gradient_masking_summary(
        dfs["results"], output_dir / _name(MASKING_SUMMARY_PLOT_PNG)
    )
    plot_confidence_margin_diagnostic(
        dfs["margins"], output_dir / _name(MARGIN_PLOT_PNG)
    )
    plot_results_heatmap(dfs["results"], output_dir / _name(HEATMAP_PLOT_PNG))
    plot_performance(dfs["performance"], output_dir / _name(PERFORMANCE_PLOT_PNG))
    plot_dither_sweep(dfs["dither"], output_dir / _name(DITHER_PLOT_PNG))
    plot_chunk_quantization_attacks(
        dfs["chunk_quant"], output_dir / _name(CHUNK_QUANT_PLOT_PNG)
    )
    plot_defense_comparison(dfs["defense"], output_dir / _name(DEFENSE_PLOT_PNG))

    overview_dir = output_dir / "visualizations"
    for name, frame in dfs.items():
        _plot_metric_grid(
            frame, name.replace("_", " ").title(), overview_dir / f"{name}.png"
        )
        _plot_model_subgraphs(frame, name, overview_dir)


def print_report(dfs: dict[str, pd.DataFrame]) -> None:
    """Print a concise console report from combined result tables."""
    print(
        pd.DataFrame(
            [
                {"table": name, "rows": len(frame), "columns": len(frame.columns)}
                for name, frame in dfs.items()
            ]
        ).to_string(index=False)
    )