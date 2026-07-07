#!/usr/bin/env python3
"""
Standalone result combiner for QuantAdv experiments.

Reads individual per-model CSV/JSON files from DATA_DIR, writes combined CSVs,
and regenerates the summary PNG plots without loading models or running attacks.
"""

from __future__ import annotations

import argparse
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


try:
    import config as _config
except Exception:
    _config = None


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(getattr(_config, "DATA_DIR", PROJECT_ROOT / "data"))

RESULTS_CSV = Path(getattr(_config, "RESULTS_CSV", DATA_DIR / "accuracyresult.csv"))
SWEEP_CSV = Path(getattr(_config, "SWEEP_CSV", DATA_DIR / "sweepresult.csv"))
PLOT_PNG = Path(getattr(_config, "PLOT_PNG", DATA_DIR / "accuracyplot.png"))

SWEEP_PLOT_PNG = Path(getattr(_config, "SWEEP_PLOT_PNG", DATA_DIR / "epsilon_sweep.png"))
ABLATION_PLOT_PNG = Path(getattr(_config, "ABLATION_PLOT_PNG", DATA_DIR / "pgd_steps_ablation.png"))
TRAJECTORY_PLOT_PNG = Path(getattr(_config, "TRAJECTORY_PLOT_PNG", DATA_DIR / "pgd_trajectory.png"))
LAYERWISE_PLOT_PNG = Path(getattr(_config, "LAYERWISE_PLOT_PNG", DATA_DIR / "layerwise_grad_profile.png"))
COMPONENT_ABLATION_PLOT_PNG = Path(getattr(_config, "COMPONENT_ABLATION_PLOT_PNG", DATA_DIR / "component_ablation.png"))
MASKING_SUMMARY_PLOT_PNG = Path(getattr(_config, "MASKING_SUMMARY_PLOT_PNG", DATA_DIR / "gradient_masking_summary.png"))
MARGIN_PLOT_PNG = Path(getattr(_config, "MARGIN_PLOT_PNG", DATA_DIR / "confidence_margin.png"))
HEATMAP_PLOT_PNG = Path(getattr(_config, "HEATMAP_PLOT_PNG", DATA_DIR / "results_heatmap.png"))

ABLATION_COMBINED_CSV = DATA_DIR / "ablation_combined.csv"
LAYERWISE_COMBINED_CSV = DATA_DIR / "layerwise_combined.csv"
TRAJECTORY_COMBINED_CSV = DATA_DIR / "trajectory_combined.csv"
COMPONENT_ABLATION_COMBINED_CSV = DATA_DIR / "component_ablation_combined.csv"
MARGIN_COMBINED_CSV = DATA_DIR / "margin_combined.csv"

PLOT_MAX_ACCURACY = getattr(_config, "PLOT_MAX_ACCURACY", 1.0)
PLOT_DPI = getattr(_config, "PLOT_DPI", 200)
PLOT_BBOX_INCHES = getattr(_config, "PLOT_BBOX_INCHES", "tight")
PLOT_GRID_ALPHA = getattr(_config, "PLOT_GRID_ALPHA", 0.35)
PLOT_LEGEND_FONT_SIZE = getattr(_config, "PLOT_LEGEND_FONT_SIZE", 8)

SUMMARY_PLOT_FIGSIZE = getattr(_config, "SUMMARY_PLOT_FIGSIZE", (14, 7))
SUMMARY_XTICK_ROTATION = getattr(_config, "SUMMARY_XTICK_ROTATION", 45)
SUMMARY_GRID_ALPHA = getattr(_config, "SUMMARY_GRID_ALPHA", 0.35)

SWEEP_PLOT_COLS_MAX = getattr(_config, "SWEEP_PLOT_COLS_MAX", 3)
SWEEP_PLOT_WIDTH = getattr(_config, "SWEEP_PLOT_WIDTH", 6)
SWEEP_PLOT_HEIGHT = getattr(_config, "SWEEP_PLOT_HEIGHT", 4)

ABLATION_FIGSIZE = getattr(_config, "ABLATION_FIGSIZE", (12, 7))
TRAJECTORY_FIGSIZE = getattr(_config, "TRAJECTORY_FIGSIZE", (14, 5))

LAYERWISE_PLOT_COLS_MAX = getattr(_config, "LAYERWISE_PLOT_COLS_MAX", 3)
LAYERWISE_PLOT_WIDTH = getattr(_config, "LAYERWISE_PLOT_WIDTH", 6)
LAYERWISE_PLOT_HEIGHT = getattr(_config, "LAYERWISE_PLOT_HEIGHT", 4)
LAYERWISE_XTICK_ROTATION = getattr(_config, "LAYERWISE_XTICK_ROTATION", 90)
LAYERWISE_XTICK_FONT_SIZE = getattr(_config, "LAYERWISE_XTICK_FONT_SIZE", 6)

COMPONENT_ABLATION_COL_WRAP = getattr(_config, "COMPONENT_ABLATION_COL_WRAP", 3)
COMPONENT_ABLATION_HEIGHT = getattr(_config, "COMPONENT_ABLATION_HEIGHT", 4)

MASKING_SUMMARY_FIGSIZE = getattr(_config, "MASKING_SUMMARY_FIGSIZE", (13, 5))
MASKING_BASELINE_LINEWIDTH = getattr(_config, "MASKING_BASELINE_LINEWIDTH", 1.0)
MASKING_SCATTER_SIZE = getattr(_config, "MASKING_SCATTER_SIZE", 60)

MARGIN_PLOT_COLS_MAX = getattr(_config, "MARGIN_PLOT_COLS_MAX", 3)
MARGIN_PLOT_WIDTH = getattr(_config, "MARGIN_PLOT_WIDTH", 5)
MARGIN_PLOT_HEIGHT = getattr(_config, "MARGIN_PLOT_HEIGHT", 4)
MARGIN_HIST_BINS = getattr(_config, "MARGIN_HIST_BINS", 30)
MARGIN_HIST_ALPHA = getattr(_config, "MARGIN_HIST_ALPHA", 0.55)

HEATMAP_MIN_WIDTH = getattr(_config, "HEATMAP_MIN_WIDTH", 12)
HEATMAP_MIN_HEIGHT = getattr(_config, "HEATMAP_MIN_HEIGHT", 6)
HEATMAP_ROW_HEIGHT = getattr(_config, "HEATMAP_ROW_HEIGHT", 0.45)
HEATMAP_VMIN = getattr(_config, "HEATMAP_VMIN", 0.0)
HEATMAP_VMAX = getattr(_config, "HEATMAP_VMAX", 1.0)
HEATMAP_LINEWIDTHS = getattr(_config, "HEATMAP_LINEWIDTHS", 0.5)


def _model_from_prefixed_path(path: Path, prefix: str, suffix: str) -> str:
    name = path.name
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    return name


def _read_csvs(paths: Iterable[Path], model_prefix: str | None = None) -> pd.DataFrame:
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
        out = out.drop_duplicates(subset=[c for c in ["model", "epsilon", "steps", "layer", "config"] if c in out.columns],
                                  keep="last")
    return out


def _write(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if df is not None and not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} rows)")
    return df


def _result_files(data_dir: Path) -> list[Path]:
    blocked = {
        RESULTS_CSV.name,
        SWEEP_CSV.name,
        ABLATION_COMBINED_CSV.name,
        LAYERWISE_COMBINED_CSV.name,
        TRAJECTORY_COMBINED_CSV.name,
        COMPONENT_ABLATION_COMBINED_CSV.name,
        MARGIN_COMBINED_CSV.name,
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
    files = _result_files(data_dir)
    df = _read_csvs(files, model_prefix="results_")
    return _write(df, output)


def combine_sweeps(data_dir: Path, output: Path = SWEEP_CSV) -> pd.DataFrame:
    files = [p for p in data_dir.glob("sweep_*.csv") if p.name != output.name]
    df = _read_csvs(files, model_prefix="sweep_")
    if not df.empty and {"model", "epsilon"}.issubset(df.columns):
        df = df.sort_values(["model", "epsilon"]).reset_index(drop=True)
    return _write(df, output)


def combine_ablation(data_dir: Path, output: Path = ABLATION_COMBINED_CSV) -> pd.DataFrame:
    files = [p for p in data_dir.glob("ablation_*.csv") if p.name != output.name]
    df = _read_csvs(files, model_prefix="ablation_")
    if not df.empty and {"model", "steps"}.issubset(df.columns):
        df = df.sort_values(["model", "steps"]).reset_index(drop=True)
    return _write(df, output)


def combine_layerwise(data_dir: Path, output: Path = LAYERWISE_COMBINED_CSV) -> pd.DataFrame:
    files = [p for p in data_dir.glob("layerwise_*.csv") if p.name != output.name]
    df = _read_csvs(files, model_prefix="layerwise_")
    return _write(df, output)


def combine_component_ablation(data_dir: Path, output: Path = COMPONENT_ABLATION_COMBINED_CSV) -> pd.DataFrame:
    files = [p for p in data_dir.glob("component_ablation_*.csv") if p.name != output.name]
    df = _read_csvs(files, model_prefix="component_ablation_")
    return _write(df, output)


def combine_trajectories(data_dir: Path, output: Path = TRAJECTORY_COMBINED_CSV) -> pd.DataFrame:
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
            rows.append({
                "model": model,
                "step": i + 1,
                "grad_norm_per_step": grad[i] if i < len(grad) else np.nan,
                "movement_from_random_start_per_step": move[i] if i < len(move) else np.nan,
            })

    return _write(pd.DataFrame(rows), output)


def combine_margins(data_dir: Path, output: Path = MARGIN_COMBINED_CSV) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.glob("margin_*.json")):
        model = _model_from_prefixed_path(path, "margin_", ".json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                margins = json.load(f)
        except Exception as exc:
            print(f"[WARN] could not read {path}: {exc}")
            continue

        for kind, values in (("clean", margins.get("clean_margins", [])), ("adv", margins.get("adv_margins", []))):
            for i, value in enumerate(values):
                rows.append({"model": model, "kind": kind, "index": i, "margin": value})

    return _write(pd.DataFrame(rows), output)


def _model_names(*dfs: pd.DataFrame) -> list[str]:
    names = []
    for df in dfs:
        if df is not None and not df.empty and "model" in df.columns:
            names.extend(df["model"].dropna().astype(str).tolist())
    return list(dict.fromkeys(names))


def plot_summary_results(df_results: pd.DataFrame, output: Path = PLOT_PNG) -> None:
    if df_results is None or df_results.empty or "model" not in df_results.columns:
        return

    acc_cols = [c for c in [
        "clean_acc", "FGSM", "PGD", "CW", "DeepFool", "JSMA", "AutoAttack",
        "Transfer_from_FP32", "MIM_Transfer", "UAP_Transfer",
        "Surrogate_Transfer", "Random_Noise", "BPDA_PGD", "NES", "Boundary_acc",
    ] if c in df_results.columns and df_results[c].notna().any()]

    if not acc_cols:
        return

    df_plot = df_results.melt(id_vars="model", value_vars=acc_cols, var_name="Attack", value_name="Accuracy")
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


def plot_epsilon_sweep_curves(df_sweep: pd.DataFrame, output: Path = SWEEP_PLOT_PNG) -> None:
    if df_sweep is None or df_sweep.empty:
        return
    value_cols = [c for c in ["PGD_acc", "Random_Noise_acc", "BPDA_acc"] if c in df_sweep.columns]
    if not value_cols:
        return

    df_long = df_sweep.melt(id_vars=["model", "epsilon"], value_vars=value_cols, var_name="Attack", value_name="Accuracy")
    df_long = df_long.dropna(subset=["Accuracy"])
    if df_long.empty:
        return

    models = df_long["model"].dropna().astype(str).unique()
    cols = min(SWEEP_PLOT_COLS_MAX, len(models))
    rows = int(math.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(SWEEP_PLOT_WIDTH * cols, SWEEP_PLOT_HEIGHT * rows), squeeze=False)

    for i, model in enumerate(models):
        ax = axes[i // cols][i % cols]
        sns.lineplot(data=df_long[df_long["model"] == model], x="epsilon", y="Accuracy", hue="Attack", marker="o", ax=ax)
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


def plot_pgd_steps_ablation(df_ablation: pd.DataFrame, output: Path = ABLATION_PLOT_PNG) -> None:
    if df_ablation is None or df_ablation.empty or not {"steps", "acc", "model"}.issubset(df_ablation.columns):
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


def plot_pgd_trajectory(df_traj: pd.DataFrame, output: Path = TRAJECTORY_PLOT_PNG) -> None:
    required = {"model", "step", "grad_norm_per_step", "movement_from_random_start_per_step"}
    if df_traj is None or df_traj.empty or not required.issubset(df_traj.columns):
        return

    fig, axes = plt.subplots(1, 2, figsize=TRAJECTORY_FIGSIZE)
    for model, group in df_traj.groupby("model", sort=False):
        group = group.sort_values("step")
        axes[0].plot(group["step"], group["grad_norm_per_step"], marker="o", label=model)
        axes[1].plot(group["step"], group["movement_from_random_start_per_step"], marker="o", label=model)

    axes[0].set_title("Gradient Norm per PGD Step")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Grad Norm")
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


def plot_layerwise_grad_profile(df_layer: pd.DataFrame, output: Path = LAYERWISE_PLOT_PNG) -> None:
    required = {"model", "layer", "grad_norm_hard", "grad_norm_ste"}
    if df_layer is None or df_layer.empty or not required.issubset(df_layer.columns):
        return

    models = df_layer["model"].dropna().astype(str).unique()
    cols = min(LAYERWISE_PLOT_COLS_MAX, len(models))
    rows = int(math.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(LAYERWISE_PLOT_WIDTH * cols, LAYERWISE_PLOT_HEIGHT * rows), squeeze=False)

    for i, model in enumerate(models):
        df = df_layer[df_layer["model"] == model].reset_index(drop=True)
        ax = axes[i // cols][i % cols]
        x = np.arange(len(df))
        ax.plot(x, df["grad_norm_hard"], marker="o", label="hard-round")
        ax.plot(x, df["grad_norm_ste"], marker="o", label="STE")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(df["layer"], rotation=LAYERWISE_XTICK_ROTATION, fontsize=LAYERWISE_XTICK_FONT_SIZE)
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


def plot_component_ablation(df_component: pd.DataFrame, output: Path = COMPONENT_ABLATION_PLOT_PNG) -> None:
    required = {"model", "config", "clean_acc", "PGD_acc"}
    if df_component is None or df_component.empty or not required.issubset(df_component.columns):
        return

    df_long = df_component.melt(
        id_vars=["model", "config"],
        value_vars=["clean_acc", "PGD_acc"],
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


def plot_gradient_masking_summary(df_results: pd.DataFrame, output: Path = MASKING_SUMMARY_PLOT_PNG) -> None:
    if df_results is None or df_results.empty or not {"model", "PGD", "AutoAttack"}.issubset(df_results.columns):
        return

    df = df_results.dropna(subset=["PGD", "AutoAttack"]).copy()
    if df.empty:
        return

    df["PGD_minus_AutoAttack"] = df["PGD"] - df["AutoAttack"]

    fig, axes = plt.subplots(1, 2, figsize=MASKING_SUMMARY_FIGSIZE)
    sns.barplot(data=df, x="model", y="PGD_minus_AutoAttack", ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=MASKING_BASELINE_LINEWIDTH)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=SUMMARY_XTICK_ROTATION, ha="right")
    axes[0].set_title("PGD - AutoAttack Accuracy Gap")
    axes[0].grid(axis="y", linestyle="--", alpha=PLOT_GRID_ALPHA)

    if "frac_zero_grad_hard" in df.columns and df["frac_zero_grad_hard"].notna().any():
        df2 = df.dropna(subset=["frac_zero_grad_hard"])
        sns.scatterplot(data=df2, x="frac_zero_grad_hard", y="PGD_minus_AutoAttack", hue="model", s=MASKING_SCATTER_SIZE, ax=axes[1])
        axes[1].set_title("Masking Gap vs Fraction of Zero Gradients")
        axes[1].grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close(fig)
    print(f"wrote {output}")


def plot_confidence_margin_diagnostic(df_margins: pd.DataFrame, output: Path = MARGIN_PLOT_PNG) -> None:
    required = {"model", "kind", "margin"}
    if df_margins is None or df_margins.empty or not required.issubset(df_margins.columns):
        return

    models = df_margins["model"].dropna().astype(str).unique()
    cols = min(MARGIN_PLOT_COLS_MAX, len(models))
    rows = int(math.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(MARGIN_PLOT_WIDTH * cols, MARGIN_PLOT_HEIGHT * rows), squeeze=False)

    for i, model in enumerate(models):
        ax = axes[i // cols][i % cols]
        group = df_margins[df_margins["model"] == model]
        clean = group[group["kind"] == "clean"]["margin"].dropna()
        adv = group[group["kind"] == "adv"]["margin"].dropna()
        if not clean.empty:
            ax.hist(clean, bins=MARGIN_HIST_BINS, alpha=MARGIN_HIST_ALPHA, label="clean", density=True)
        if not adv.empty:
            ax.hist(adv, bins=MARGIN_HIST_BINS, alpha=MARGIN_HIST_ALPHA, label="PGD-adv", density=True)
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


def plot_results_heatmap(df_results: pd.DataFrame, output: Path = HEATMAP_PLOT_PNG) -> None:
    if df_results is None or df_results.empty or "model" not in df_results.columns:
        return

    candidate_cols = [
        "clean_acc", "FGSM", "PGD", "AutoAttack", "CW", "DeepFool", "JSMA",
        "Surrogate_Transfer", "Transfer_from_FP32", "MIM_Transfer", "UAP_Transfer",
        "Random_Noise", "BPDA_PGD", "NES", "Boundary_acc",
    ]
    cols = [c for c in candidate_cols if c in df_results.columns and df_results[c].notna().any()]
    if not cols:
        return

    df_heat = df_results.set_index("model")[cols].astype(float)
    plt.figure(figsize=(max(HEATMAP_MIN_WIDTH, len(cols)), max(HEATMAP_MIN_HEIGHT, len(df_heat) * HEATMAP_ROW_HEIGHT)))
    sns.heatmap(df_heat, annot=True, fmt=".2f", cmap="RdYlGn", vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, linewidths=HEATMAP_LINEWIDTHS)
    plt.title("Full Results Heatmap: Models vs Attacks")
    plt.tight_layout()
    plt.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close()
    print(f"wrote {output}")


def combine_all(data_dir: Path) -> dict[str, pd.DataFrame]:
    data_dir.mkdir(parents=True, exist_ok=True)

    df_results = combine_scalar_results(data_dir, RESULTS_CSV)
    df_sweep = combine_sweeps(data_dir, SWEEP_CSV)
    df_ablation = combine_ablation(data_dir, ABLATION_COMBINED_CSV)
    df_layer = combine_layerwise(data_dir, LAYERWISE_COMBINED_CSV)
    df_component = combine_component_ablation(data_dir, COMPONENT_ABLATION_COMBINED_CSV)
    df_traj = combine_trajectories(data_dir, TRAJECTORY_COMBINED_CSV)
    df_margins = combine_margins(data_dir, MARGIN_COMBINED_CSV)

    if df_results.empty and RESULTS_CSV.exists():
        df_results = pd.read_csv(RESULTS_CSV)
    if df_sweep.empty and SWEEP_CSV.exists():
        df_sweep = pd.read_csv(SWEEP_CSV)

    return {
        "results": df_results,
        "sweep": df_sweep,
        "ablation": df_ablation,
        "layerwise": df_layer,
        "component_ablation": df_component,
        "trajectory": df_traj,
        "margins": df_margins,
    }


def plot_all(dfs: dict[str, pd.DataFrame]) -> None:
    plot_summary_results(dfs["results"], PLOT_PNG)
    plot_epsilon_sweep_curves(dfs["sweep"], SWEEP_PLOT_PNG)
    plot_pgd_steps_ablation(dfs["ablation"], ABLATION_PLOT_PNG)
    plot_pgd_trajectory(dfs["trajectory"], TRAJECTORY_PLOT_PNG)
    plot_layerwise_grad_profile(dfs["layerwise"], LAYERWISE_PLOT_PNG)
    plot_component_ablation(dfs["component_ablation"], COMPONENT_ABLATION_PLOT_PNG)
    plot_gradient_masking_summary(dfs["results"], MASKING_SUMMARY_PLOT_PNG)
    plot_confidence_margin_diagnostic(dfs["margins"], MARGIN_PLOT_PNG)
    plot_results_heatmap(dfs["results"], HEATMAP_PLOT_PNG)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine QuantAdv per-model CSV/JSON outputs and regenerate plots.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing result files.")
    parser.add_argument("--no-plots", action="store_true", help="Only write combined CSVs; do not regenerate PNG plots.")
    args = parser.parse_args()

    data_dir = args.data_dir

    print(f"combining files in {data_dir}")
    dfs = combine_all(data_dir)

    if not args.no_plots:
        plot_all(dfs)

    print("done")


if __name__ == "__main__":
    main()
