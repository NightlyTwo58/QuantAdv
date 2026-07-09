"""Plotting helpers ported from the active archive implementation."""

import json
import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import *
from .paths import csv_path, json_path, SCORECARD_CSV, SCORECARD_PLOT_PNG, defense_summary_csv_path


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


def plot_results_heatmap_by_category(df_results):
    """Split the full results heatmap into smaller, readable category heatmaps
    instead of one giant heatmap with 20+ columns. Categories with only 1-2
    columns are rendered as horizontal bar charts instead. Rows with no data
    for any column in the category (e.g. FP32 models for transfer attacks)
    are dropped entirely rather than shown as empty/blank rows."""
    if df_results is None or df_results.empty:
        return
    categories = {
        "core_attacks": ["clean_acc", "FGSM", "PGD", "AutoAttack"],
        "extended_whitebox": ["CW", "DeepFool", "JSMA"],
        "transfer_attacks": ["Surrogate_Transfer", "Transfer_from_FP32", "MIM_Transfer",
                              "UAP_Transfer", "Transfer_to_FP32", "MIM_Transfer_to_FP32",
                              "UAP_Transfer_to_FP32"],
        "defense": ["Adaptive_Guardrail", "Adaptive_DetectGuard"],
    }
    for title, candidate in categories.items():
        cols = [c for c in candidate if c in df_results.columns and df_results[c].notna().any()]
        if not cols:
            continue
        df_heat = df_results.set_index("model")[cols].astype(float)
        df_heat = df_heat.dropna(how="all")
        if df_heat.empty:
            continue
        n_rows = len(df_heat)
        pretty_title = title.replace(chr(95), chr(32)).title()

        if len(cols) <= 2:
            sort_col = cols[-1] if len(cols) == 2 else cols[0]
            df_heat_sorted = df_heat.sort_values(by=sort_col, ascending=True, na_position="first")
            df_long = df_heat_sorted.reset_index().melt(id_vars="model", var_name="Attack", value_name="Accuracy")
            df_long = df_long.dropna(subset=["Accuracy"])
            model_order = df_heat_sorted.index.tolist()
            plt.figure(figsize=(12, min(14, max(6, n_rows * 0.4))))
            ax = sns.barplot(data=df_long, y="model", x="Accuracy", hue="Attack", orient="h",
                              order=model_order, edgecolor="black", linewidth=0.6)
            data_max = float(df_heat.values[pd.notna(df_heat.values)].max()) if df_heat.notna().any().any() else PLOT_MAX_ACCURACY
            ax.set_xlim(0, min(PLOT_MAX_ACCURACY, data_max * 1.2))
            ax.set_title(f"{pretty_title}: Accuracy by Model, sorted by {sort_col}")
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("")
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
            ax.grid(axis="x", linestyle="--", alpha=PLOT_GRID_ALPHA)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.savefig(os.path.join(DATA_DIR, f"heatmap_{title}.png"), dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
            plt.show()
            continue

        plt.figure(figsize=(12, min(14, max(6, n_rows * HEATMAP_ROW_HEIGHT))))
        ax = sns.heatmap(df_heat, annot=True, fmt=".2f", cmap="RdYlGn", vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, linewidths=HEATMAP_LINEWIDTHS)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=9)
        plt.title(f"{pretty_title}: Accuracy by Model")
        plt.ylabel("")
        plt.subplots_adjust(left=0.28, bottom=0.15)
        plt.savefig(os.path.join(DATA_DIR, f"heatmap_{title}.png"), dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
        plt.show()


def plot_masking_scorecard(df_scorecard=None):
    """Read the masking scorecard (or accept it directly) and plot the two
    masking-gap metrics as separate subplots, since they have different
    interpretations and mixing them on one axis obscures the meaning:
      - pgd_minus_transfer_gap: near 0 is expected/healthy (a transfer
        attack doesn't need this model's gradients, so PGD and transfer
        should be similar for a non-masked model).
      - pgd_minus_random_gap: near 0 = PGD is doing no better than blind
        random noise, i.e. a strong masking signal. This one should be
        clearly positive for a healthy model."""
    if df_scorecard is None:
        if not os.path.exists(SCORECARD_CSV):
            return
        df_scorecard = pd.read_csv(SCORECARD_CSV)
    if df_scorecard is None or df_scorecard.empty:
        return
    plot_cols = [c for c in ["pgd_minus_transfer_gap", "pgd_minus_random_gap"] if c in df_scorecard.columns]
    if not plot_cols:
        return

    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(SUMMARY_PLOT_FIGSIZE[0], SUMMARY_PLOT_FIGSIZE[1] * len(plot_cols)), squeeze=False)
    subtitles = {
        "pgd_minus_transfer_gap": "PGD - Transfer Gap (near 0 is expected/healthy)",
        "pgd_minus_random_gap": "PGD - Random Noise Gap (near 0 = masking signal)",
    }
    for i, col in enumerate(plot_cols):
        ax = axes[i][0]
        df_col = df_scorecard.dropna(subset=[col])
        sns.barplot(data=df_col, x="model", y=col, ax=ax, edgecolor="black", linewidth=0.6)
        ax.axhline(0, color="black", linewidth=MASKING_BASELINE_LINEWIDTH)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=SUMMARY_XTICK_ROTATION, ha="right")
        ax.set_title(subtitles.get(col, col))
        ax.set_ylabel("Accuracy Gap")
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=SUMMARY_GRID_ALPHA)
        ax.set_axisbelow(True)
    fig.suptitle("Masking Scorecard")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.4)
    fig.savefig(SCORECARD_PLOT_PNG, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_defense_summary(df_defense=None):
    """Plot the defense_summary.csv output (generated by suite.py's
    run_defense_suite) which was previously being written but never plotted."""
    if df_defense is None:
        p = defense_summary_csv_path()
        if not os.path.exists(p):
            return
        df_defense = pd.read_csv(p)
    if df_defense is None or df_defense.empty:
        return
    id_col = "model" if "model" in df_defense.columns else df_defense.columns[0]
    value_cols = [c for c in df_defense.columns if c != id_col and df_defense[c].dtype != object]
    if not value_cols:
        return
    df_long = df_defense.melt(id_vars=id_col, value_vars=value_cols, var_name="Defense", value_name="Accuracy")
    plt.figure(figsize=SUMMARY_PLOT_FIGSIZE)
    sns.barplot(data=df_long, x=id_col, y="Accuracy", hue="Defense")
    plt.xticks(rotation=SUMMARY_XTICK_ROTATION, ha="right")
    plt.title("Defense Suite Summary")
    plt.ylim(0, PLOT_MAX_ACCURACY)
    plt.grid(axis="y", linestyle="--", alpha=SUMMARY_GRID_ALPHA)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "defense_summary_plot.png"), dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


STD_DISPLAY_THRESHOLD = 0.005


def plot_results_heatmap_with_error(df_results):
    """Like plot_results_heatmap_by_category, but annotates each cell with
    its standard deviation (mean +/- std) wherever a `{col}_std` column
    exists, addressing the lack of error margins noted in issue #17.
    Categories with only 1-2 columns are rendered as horizontal bar charts
    with error bars instead of a cramped 1-column heatmap."""
    if df_results is None or df_results.empty:
        return
    categories = {
        "core_attacks": ["clean_acc", "FGSM", "PGD", "AutoAttack"],
        "extended_whitebox": ["CW", "DeepFool", "JSMA"],
        "transfer_attacks": ["Surrogate_Transfer", "Transfer_from_FP32", "MIM_Transfer",
                              "UAP_Transfer", "Transfer_to_FP32", "MIM_Transfer_to_FP32",
                              "UAP_Transfer_to_FP32"],
        "defense": ["Adaptive_Guardrail", "Adaptive_DetectGuard"],
    }
    df_indexed = df_results.set_index("model")
    for title, candidate in categories.items():
        cols = [c for c in candidate if c in df_results.columns and df_results[c].notna().any()]
        if not cols:
            continue
        df_heat = df_indexed[cols].astype(float)
        n_rows = len(df_heat)
        pretty_title = title.replace(chr(95), chr(32)).title()

        if len(cols) <= 2:
            fig, ax = plt.subplots(figsize=(12, min(14, max(6, n_rows * 0.4))))
            y_pos = np.arange(n_rows)
            bar_h = 0.8 / len(cols)
            for i, col in enumerate(cols):
                std_col = f"{col}_std"
                errs = df_indexed[std_col].reindex(df_heat.index).values if std_col in df_indexed.columns else None
                ax.barh(y_pos + i * bar_h, df_heat[col].values, height=bar_h,
                        xerr=errs, capsize=3, label=col)
            ax.set_yticks(y_pos + bar_h * (len(cols) - 1) / 2)
            ax.set_yticklabels(df_heat.index, fontsize=9)
            ax.set_xlim(0, PLOT_MAX_ACCURACY)
            ax.set_xlabel("Accuracy")
            ax.set_title(f"{pretty_title}: Accuracy by Model (mean {chr(177)} std)")
            ax.legend()
            ax.grid(axis="x", linestyle="--", alpha=PLOT_GRID_ALPHA)
            plt.tight_layout()
            plt.savefig(os.path.join(DATA_DIR, f"heatmap_{title}_witherror.png"), dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
            plt.show()
            continue

        annot = df_heat.copy().astype(object)
        for col in cols:
            std_col = f"{col}_std"
            for idx in df_heat.index:
                mean_val = df_heat.loc[idx, col]
                if std_col in df_indexed.columns:
                    std_val = df_indexed.loc[idx, std_col] if idx in df_indexed.index else None
                    if pd.notna(std_val):
                        annot.loc[idx, col] = f"{mean_val:.2f}\n\u00b1{std_val:.2f}"
                        continue
                annot.loc[idx, col] = f"{mean_val:.2f}"

        plt.figure(figsize=(12, min(14, max(6, n_rows * HEATMAP_ROW_HEIGHT * 1.5))))
        ax = sns.heatmap(df_heat, annot=annot, fmt="", cmap="RdYlGn", vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, linewidths=HEATMAP_LINEWIDTHS, annot_kws={"fontsize": 8, "linespacing": 1.6})
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        plt.title(f"{pretty_title}: Accuracy by Model (mean {chr(177)} std)")
        plt.ylabel("")
        plt.subplots_adjust(left=0.28, bottom=0.22)
        plt.savefig(os.path.join(DATA_DIR, f"heatmap_{title}_witherror.png"), dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
        plt.show()


# --- Overrides applied after readability feedback ---
STD_DISPLAY_THRESHOLD = 0.005


def plot_results_heatmap_with_error(df_results):
    """Overridden version: omits trivial (near-zero) std annotations, drops
    rows with no data for the category (e.g. FP32 for transfer attacks),
    and adds bar edges for visibility."""
    if df_results is None or df_results.empty:
        return
    categories = {
        "core_attacks": ["clean_acc", "FGSM", "PGD", "AutoAttack"],
        "extended_whitebox": ["CW", "DeepFool", "JSMA"],
        "transfer_attacks": ["Surrogate_Transfer", "Transfer_from_FP32", "MIM_Transfer",
                              "UAP_Transfer", "Transfer_to_FP32", "MIM_Transfer_to_FP32",
                              "UAP_Transfer_to_FP32"],
        "defense": ["Adaptive_Guardrail", "Adaptive_DetectGuard"],
    }
    df_indexed = df_results.set_index("model")
    for title, candidate in categories.items():
        cols = [c for c in candidate if c in df_results.columns and df_results[c].notna().any()]
        if not cols:
            continue
        df_heat = df_indexed[cols].astype(float)
        df_heat = df_heat.dropna(how="all")
        if df_heat.empty:
            continue
        n_rows = len(df_heat)
        pretty_title = title.replace(chr(95), chr(32)).title()

        if len(cols) <= 2:
            sort_col = cols[-1] if len(cols) == 2 else cols[0]
            df_heat = df_heat.sort_values(by=sort_col, ascending=True, na_position="first")
            n_rows = len(df_heat)
            fig, ax = plt.subplots(figsize=(12, min(14, max(6, n_rows * 0.4))))
            y_pos = np.arange(n_rows)
            bar_h = 0.8 / len(cols)
            for i, col in enumerate(cols):
                std_col = f"{col}_std"
                errs = None
                if std_col in df_indexed.columns:
                    errs = df_indexed[std_col].reindex(df_heat.index).values
                    if not (pd.notna(errs) & (errs >= STD_DISPLAY_THRESHOLD)).any():
                        errs = None
                ax.barh(y_pos + i * bar_h, df_heat[col].values, height=bar_h,
                        xerr=errs, capsize=3, label=col, edgecolor="black", linewidth=0.6)
            ax.set_yticks(y_pos + bar_h * (len(cols) - 1) / 2)
            ax.set_yticklabels(df_heat.index, fontsize=9)
            data_max = float(df_heat.values[pd.notna(df_heat.values)].max()) if df_heat.notna().any().any() else PLOT_MAX_ACCURACY
            ax.set_xlim(0, min(PLOT_MAX_ACCURACY, data_max * 1.2))
            ax.set_xlabel("Accuracy")
            ax.set_title(f"{pretty_title}: Accuracy by Model (mean {chr(177)} std), sorted by {sort_col}")
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
            ax.grid(axis="x", linestyle="--", alpha=PLOT_GRID_ALPHA)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.savefig(os.path.join(DATA_DIR, f"heatmap_{title}_witherror.png"), dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
            plt.show()
            continue

        annot = df_heat.copy().astype(object)
        for col in cols:
            std_col = f"{col}_std"
            for idx in df_heat.index:
                mean_val = df_heat.loc[idx, col]
                if pd.isna(mean_val):
                    annot.loc[idx, col] = ""
                    continue
                if std_col in df_indexed.columns:
                    std_val = df_indexed.loc[idx, std_col] if idx in df_indexed.index else None
                    if pd.notna(std_val) and std_val >= STD_DISPLAY_THRESHOLD:
                        annot.loc[idx, col] = f"{mean_val:.2f}\n\u00b1{std_val:.2f}"
                        continue
                annot.loc[idx, col] = f"{mean_val:.2f}"

        plt.figure(figsize=(12, min(14, max(6, n_rows * HEATMAP_ROW_HEIGHT * 1.5))))
        ax = sns.heatmap(df_heat, annot=annot, fmt="", cmap="RdYlGn", vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, linewidths=HEATMAP_LINEWIDTHS, annot_kws={"fontsize": 8, "linespacing": 1.6})
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        plt.title(f"{pretty_title}: Accuracy by Model (mean {chr(177)} std)")
        plt.ylabel("")
        plt.subplots_adjust(left=0.28, bottom=0.22)
        plt.savefig(os.path.join(DATA_DIR, f"heatmap_{title}_witherror.png"), dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
        plt.show()


def plot_masking_diagnostics_gap(df_results):
    """Simplified version of the masking_diagnostics category: instead of
    showing Random_Noise (which is ~0.9 for almost every model and adds no
    information) next to BPDA_PGD, just show BPDA_PGD alone, sorted, with a
    single color. This is the metric that actually varies and matters."""
    if df_results is None or df_results.empty:
        return
    if "BPDA_PGD" not in df_results.columns:
        return
    df = df_results[["model", "BPDA_PGD"]].dropna(subset=["BPDA_PGD"]).copy()
    if df.empty:
        return
    df = df.sort_values(by="BPDA_PGD", ascending=True)
    n_rows = len(df)

    fig, ax = plt.subplots(figsize=(12, min(14, max(6, n_rows * 0.4))))
    ax.barh(df["model"], df["BPDA_PGD"], color="#4C72B0", edgecolor="black", linewidth=0.6)
    data_max = float(df["BPDA_PGD"].max())
    ax.set_xlim(0, min(PLOT_MAX_ACCURACY, data_max * 1.2))
    ax.set_xlabel("BPDA_PGD Accuracy")
    ax.set_title("Masking Diagnostic: BPDA_PGD Accuracy by Model (lower = stronger masking signal)")
    ax.grid(axis="x", linestyle="--", alpha=PLOT_GRID_ALPHA)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "heatmap_masking_diagnostics_simplified.png"), dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_pgd_vs_bpda_scatter(df_results):
    """Scatter plot: x=PGD accuracy, y=BPDA_PGD accuracy, one point per model.
    Points near the y=x diagonal indicate genuine robustness (PGD and BPDA
    agree). Points far below the diagonal indicate gradient masking (PGD
    looks strong but BPDA, which routes around the masking, reveals the
    model is actually weak). Color-coded by quantization variant."""
    if df_results is None or df_results.empty:
        return
    if "PGD" not in df_results.columns or "BPDA_PGD" not in df_results.columns:
        return
    df = df_results[["model", "PGD", "BPDA_PGD"]].dropna()
    if df.empty:
        return

    def _variant(name):
        for suf in ["_FP32", "_int8_PTQ", "_int4_PTQ", "_int8_QAT"]:
            if name.endswith(suf):
                return suf[1:]
        return "other"
    df = df.copy()
    df["variant"] = df["model"].apply(_variant)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="y = x (no masking)")
    sns.scatterplot(data=df, x="PGD", y="BPDA_PGD", hue="variant", s=90, edgecolor="black", linewidth=0.5, ax=ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("PGD Accuracy")
    ax.set_ylabel("BPDA_PGD Accuracy")
    ax.set_title("Genuine Robustness vs Gradient Masking\n(points below the diagonal = masking signal)")
    ax.grid(linestyle="--", alpha=PLOT_GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "scatter_pgd_vs_bpda.png"), dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_pgd_bpda_slope(df_results):
    """Slope chart: each model is a line connecting its PGD accuracy (left)
    to its BPDA_PGD accuracy (right). Steep downward slopes indicate
    gradient masking; near-flat lines indicate genuine robustness."""
    if df_results is None or df_results.empty:
        return
    if "PGD" not in df_results.columns or "BPDA_PGD" not in df_results.columns:
        return
    df = df_results[["model", "PGD", "BPDA_PGD"]].dropna()
    if df.empty:
        return
    df = df.sort_values(by="BPDA_PGD", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.35)))
    colors = plt.cm.RdYlGn(df["BPDA_PGD"].values / max(df["PGD"].max(), 0.01))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([0, 1], [row["PGD"], row["BPDA_PGD"]], marker="o", color=colors[i], linewidth=1.5)
        ax.text(-0.05, row["PGD"], row["model"], ha="right", va="center", fontsize=8)
    ax.set_xlim(-0.35, 1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["PGD", "BPDA_PGD"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("PGD to BPDA_PGD Accuracy Drop per Model\n(steep drop = masking signal)")
    ax.grid(axis="y", linestyle="--", alpha=PLOT_GRID_ALPHA)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "slope_pgd_to_bpda.png"), dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()


def plot_masking_by_quant_variant(df_results):
    """Bubble/strip chart summarizing masking severity by quantization
    variant category (int8 PTQ, int4 PTQ, int8 QAT): x=variant category,
    y=masking gap (PGD - BPDA_PGD), one point per model, sized by clean
    accuracy. Shows which quantization strategy is most prone to masking
    as a category-level trend rather than model-by-model."""
    if df_results is None or df_results.empty:
        return
    needed = ["PGD", "BPDA_PGD"]
    if not all(c in df_results.columns for c in needed):
        return
    df = df_results.dropna(subset=needed).copy()
    if df.empty:
        return

    def _variant(name):
        for suf in ["_FP32", "_int8_PTQ", "_int4_PTQ", "_int8_QAT"]:
            if name.endswith(suf):
                return suf[1:]
        return "other"
    df["variant"] = df["model"].apply(_variant)
    df = df[df["variant"] != "FP32"]
    if df.empty:
        return
    df["masking_gap"] = df["PGD"] - df["BPDA_PGD"]
    size = (df["clean_acc"] * 300) if "clean_acc" in df.columns else 100

    fig, ax = plt.subplots(figsize=(9, 6))
    variant_order = [v for v in ["int8_PTQ", "int4_PTQ", "int8_QAT"] if v in df["variant"].unique()]
    sns.stripplot(data=df, x="variant", y="masking_gap", order=variant_order, size=10,
                  jitter=0.15, edgecolor="black", linewidth=0.6, ax=ax)
    means = df.groupby("variant")["masking_gap"].mean().reindex(variant_order)
    for i, v in enumerate(variant_order):
        ax.hlines(means[v], i - 0.25, i + 0.25, color="black", linewidth=2.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Quantization Variant")
    ax.set_ylabel("Masking Gap (PGD - BPDA_PGD)")
    ax.set_title("Masking Severity by Quantization Strategy\n(black bar = mean; higher = more masking)")
    ax.grid(axis="y", linestyle="--", alpha=PLOT_GRID_ALPHA)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "masking_by_variant.png"), dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.show()

...
