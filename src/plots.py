"""Plotting helpers ported from the active archive implementation."""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import *
from .paths import csv_path, json_path


def plot_defense_comparison(df_results):
    if df_results is None or df_results.empty:
        return
    defense_tags = ("_AT", "_Sanitized", "_Smoothed", "_Guardrail", "_DetectGuard")
    df_def = df_results[
        df_results["model"].astype(str).str.contains("|".join(defense_tags))
    ]
    if df_def.empty:
        return
    cols = [
        c
        for c in [
            "clean_acc",
            "PGD",
            "AutoAttack",
            "BPDA_Adaptive",
            "EOT_PGD",
            "Adaptive_Guardrail",
            "Adaptive_DetectGuard",
        ]
        if c in df_def.columns and df_def[c].notna().any()
    ]
    if not cols:
        return
    df_long = df_def.melt(
        id_vars="model", value_vars=cols, var_name="Attack", value_name="Accuracy"
    )

    plt.figure(figsize=SUMMARY_PLOT_FIGSIZE)
    sns.barplot(data=df_long, x="model", y="Accuracy", hue="Attack")
    plt.xticks(rotation=SUMMARY_XTICK_ROTATION, ha="right")
    plt.title("Defense Variants: Accuracy under Attack")
    plt.ylim(0, PLOT_MAX_ACCURACY)
    plt.grid(axis="y", linestyle="--", alpha=SUMMARY_GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(
        os.path.join(DATA_DIR, "defense_comparison.png"),
        dpi=PLOT_DPI,
        bbox_inches=PLOT_BBOX_INCHES,
    )
    plt.show()


def plot_epsilon_sweep_curves(df_sweep):
    if df_sweep is None or df_sweep.empty:
        return
    value_cols = [
        c for c in ["PGD_acc", "Random_Noise_acc", "BPDA_acc"] if c in df_sweep.columns
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

    models = df_long["model"].unique()
    cols = min(SWEEP_PLOT_COLS_MAX, len(models))
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(SWEEP_PLOT_WIDTH * cols, SWEEP_PLOT_HEIGHT * rows),
        squeeze=False,
    )
    for i, m in enumerate(models):
        ax = axes[i // cols][i % cols]
        sns.lineplot(
            data=df_long[df_long["model"] == m],
            x="epsilon",
            y="Accuracy",
            hue="Attack",
            marker="o",
            ax=ax,
        )
        ax.set_title(m)
        ax.set_ylim(0, PLOT_MAX_ACCURACY)
        ax.grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    for j in range(len(models), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Accuracy vs Perturbation Budget (Epsilon Sweep)")
    fig.tight_layout()
    fig.savefig(SWEEP_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_pgd_steps_ablation(model_names):
    frames = [
        pd.read_csv(csv_path(n, "ablation"))
        for n in model_names
        if os.path.exists(csv_path(n, "ablation"))
    ]
    if not frames:
        return
    df_all = pd.concat(frames, ignore_index=True)

    plt.figure(figsize=ABLATION_FIGSIZE)
    sns.lineplot(data=df_all, x="steps", y="acc", hue="model", marker="o")
    plt.title("PGD Accuracy vs Number of Steps (Gradient Masking Check)")
    plt.xlabel("PGD steps")
    plt.ylabel("Accuracy")
    plt.ylim(0, PLOT_MAX_ACCURACY)
    plt.grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(ABLATION_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_pgd_trajectory(model_names):
    trajs = {}
    for name in model_names:
        p = json_path(name, "trajectory")
        if os.path.exists(p):
            with open(p) as f:
                trajs[name] = json.load(f)
    if not trajs:
        return

    fig, axes = plt.subplots(1, 2, figsize=TRAJECTORY_FIGSIZE)
    for name, traj in trajs.items():
        steps = range(1, len(traj["grad_norm_per_step"]) + 1)
        axes[0].plot(steps, traj["grad_norm_per_step"], marker="o", label=name)
        axes[1].plot(
            steps, traj["movement_from_random_start_per_step"], marker="o", label=name
        )

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
    fig.savefig(TRAJECTORY_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_layerwise_grad_profile(model_names):
    quant_names = [n for n in model_names if os.path.exists(csv_path(n, "layerwise"))]
    if not quant_names:
        return

    cols = min(LAYERWISE_PLOT_COLS_MAX, len(quant_names))
    rows = int(np.ceil(len(quant_names) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(LAYERWISE_PLOT_WIDTH * cols, LAYERWISE_PLOT_HEIGHT * rows),
        squeeze=False,
    )
    for i, name in enumerate(quant_names):
        df = pd.read_csv(csv_path(name, "layerwise"))
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
        ax.set_title(name)
        ax.set_ylabel("Grad Norm (log)")
        ax.legend(fontsize=PLOT_LEGEND_FONT_SIZE)
        ax.grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    for j in range(len(quant_names), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Layerwise Gradient Norms: Hard-Round vs STE")
    fig.tight_layout()
    fig.savefig(LAYERWISE_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_component_ablation(model_names):
    frames = [
        pd.read_csv(csv_path(n, "component_ablation"))
        for n in model_names
        if os.path.exists(csv_path(n, "component_ablation"))
    ]
    if not frames:
        return
    df_all = pd.concat(frames, ignore_index=True)
    df_long = df_all.melt(
        id_vars=["model", "config"],
        value_vars=["clean_acc", "PGD_acc"],
        var_name="Metric",
        value_name="Accuracy",
    )

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
    g.savefig(COMPONENT_ABLATION_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_chunk_quantization_attacks(model_names):
    frames = [
        pd.read_csv(csv_path(n, "chunk_quant"))
        for n in model_names
        if os.path.exists(csv_path(n, "chunk_quant"))
    ]
    if not frames:
        return
    df_all = pd.concat(frames, ignore_index=True)
    df_long = df_all.melt(
        id_vars=["model", "chunk_label", "first_layer", "last_layer"],
        value_vars=["clean_acc", "PGD_acc"],
        var_name="Metric",
        value_name="Accuracy",
    )
    df_long = df_long.dropna(subset=["Accuracy"])
    if df_long.empty:
        return

    g = sns.catplot(
        data=df_long,
        x="chunk_label",
        y="Accuracy",
        hue="Metric",
        col="model",
        kind="bar",
        col_wrap=CHUNK_QUANT_COL_WRAP,
        height=CHUNK_QUANT_HEIGHT,
        sharey=True,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Quantized layer chunk", "Accuracy")
    g.set(ylim=(0, PLOT_MAX_ACCURACY))
    for ax in g.axes.flatten():
        ax.grid(axis="y", linestyle="--", alpha=PLOT_GRID_ALPHA)
    g.savefig(CHUNK_QUANT_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_gradient_masking_summary(df_results):
    if (
        df_results is None
        or df_results.empty
        or not {"model", "PGD", "AutoAttack"}.issubset(df_results.columns)
    ):
        return
    df = df_results.dropna(subset=["PGD", "AutoAttack"]).copy()
    if df.empty:
        return
    df["PGD_minus_AutoAttack"] = df["PGD"] - df["AutoAttack"]

    fig, axes = plt.subplots(1, 2, figsize=MASKING_SUMMARY_FIGSIZE)
    sns.barplot(data=df, x="model", y="PGD_minus_AutoAttack", ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=MASKING_BASELINE_LINEWIDTH)
    axes[0].set_xticklabels(
        axes[0].get_xticklabels(), rotation=SUMMARY_XTICK_ROTATION, ha="right"
    )
    axes[0].set_title("PGD - AutoAttack Accuracy Gap")
    axes[0].grid(axis="y", linestyle="--", alpha=PLOT_GRID_ALPHA)

    if "frac_zero_grad_hard" in df.columns:
        df2 = df.dropna(subset=["frac_zero_grad_hard"])
        sns.scatterplot(
            data=df2,
            x="frac_zero_grad_hard",
            y="PGD_minus_AutoAttack",
            hue="model",
            s=MASKING_SCATTER_SIZE,
            ax=axes[1],
        )
        axes[1].set_title("Masking Gap vs Fraction of Zero Gradients")
        axes[1].grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(MASKING_SUMMARY_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_confidence_margin_diagnostic(model_names):
    data = {}
    for name in model_names:
        p = json_path(name, "margin")
        if os.path.exists(p):
            with open(p) as f:
                data[name] = json.load(f)
    if not data:
        return

    cols = min(MARGIN_PLOT_COLS_MAX, len(data))
    rows = int(np.ceil(len(data) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(MARGIN_PLOT_WIDTH * cols, MARGIN_PLOT_HEIGHT * rows),
        squeeze=False,
    )
    for i, (name, margins) in enumerate(data.items()):
        ax = axes[i // cols][i % cols]
        ax.hist(
            margins["clean_margins"],
            bins=MARGIN_HIST_BINS,
            alpha=MARGIN_HIST_ALPHA,
            label="clean",
            density=True,
        )
        ax.hist(
            margins["adv_margins"],
            bins=MARGIN_HIST_BINS,
            alpha=MARGIN_HIST_ALPHA,
            label="PGD-adv",
            density=True,
        )
        ax.set_title(name)
        ax.set_xlabel("Top1 - Top2 Softmax Margin")
        ax.legend(fontsize=PLOT_LEGEND_FONT_SIZE)
        ax.grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    for j in range(len(data), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Confidence Margin: Clean vs PGD-Adversarial")
    fig.tight_layout()
    fig.savefig(MARGIN_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_results_heatmap(df_results):
    if df_results is None or df_results.empty:
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
        "Transfer_to_FP32",
        "MIM_Transfer_to_FP32",
        "UAP_Transfer_to_FP32",
        "Random_Noise",
        "BPDA_PGD",
        "BPDA_Adaptive",
        "EOT_PGD",
        "Adaptive_Guardrail",
        "Adaptive_DetectGuard",
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
    plt.savefig(HEATMAP_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()
