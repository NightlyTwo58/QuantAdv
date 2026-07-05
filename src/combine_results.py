#!/usr/bin/env python
# coding: utf-8
"""
Combines the per-model CSVs/JSON written by QuantAdvCC.py into master files
and produces summary plots.

Per-model files (all in DATA_DIR), one set per model:
  results_<model>.csv     -- scalar metrics (clean_acc, FGSM, PGD, ..., now
                              also Random_Noise, grad_cosine_sim_with_FP32,
                              plateau_fraction)
  sweep_<model>.csv        -- epsilon sweep (PGD_acc, BPDA_acc, now also
                              Random_Noise_acc)
  ablation_<model>.csv     -- PGD accuracy at step counts 0/1/2/5/10/20/50
  layerwise_<model>.csv    -- per-layer input-gradient norm, hard vs STE
  trajectory_<model>.json  -- per-step grad norm / movement during PGD

Run this after launcher.py finishes all subprocesses:

    python combine_results.py

Safe to re-run at any time since it only reads whatever per-model files
currently exist and never mutates them.
"""
import glob
import json
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RESULTS_GLOB = os.path.join(DATA_DIR, "results_*.csv")
SWEEP_GLOB = os.path.join(DATA_DIR, "sweep_*.csv")
ABLATION_GLOB = os.path.join(DATA_DIR, "ablation_*.csv")
LAYERWISE_GLOB = os.path.join(DATA_DIR, "layerwise_*.csv")
TRAJECTORY_GLOB = os.path.join(DATA_DIR, "trajectory_*.json")

RESULTS_COMBINED_CSV = os.path.join(DATA_DIR, "results_combined.csv")
SWEEP_COMBINED_CSV = os.path.join(DATA_DIR, "sweep_combined.csv")
ABLATION_COMBINED_CSV = os.path.join(DATA_DIR, "ablation_combined.csv")
LAYERWISE_COMBINED_CSV = os.path.join(DATA_DIR, "layerwise_combined.csv")
TRAJECTORY_COMBINED_CSV = os.path.join(DATA_DIR, "trajectory_combined.csv")
SCORECARD_CSV = os.path.join(DATA_DIR, "masking_scorecard.csv")

ACCURACY_PLOT_PNG = os.path.join(DATA_DIR, "accuracy_plot.png")
SWEEP_PLOT_PNG = os.path.join(DATA_DIR, "sweep_plot.png")
ABLATION_PLOT_PNG = os.path.join(DATA_DIR, "ablation_plot.png")
TRAJECTORY_PLOT_PNG = os.path.join(DATA_DIR, "trajectory_plot.png")
LAYERWISE_PLOT_PNG = os.path.join(DATA_DIR, "layerwise_plot.png")
SCORECARD_PLOT_PNG = os.path.join(DATA_DIR, "masking_scorecard_plot.png")

"""
Combines resuts from launcher.py's multiple QuantAdvCC.py threads and creates graphs.
"""

def _load_and_concat(pattern, exclude_pattern=None):
    paths = sorted(glob.glob(pattern))
    if exclude_pattern is not None:
        excluded = set(glob.glob(exclude_pattern))
        paths = [p for p in paths if p not in excluded]

    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")

    if not frames:
        return pd.DataFrame(), paths

    df = pd.concat(frames, ignore_index=True)
    return df, paths


def _model_name_from_path(path, prefix, suffix):
    base = os.path.basename(path)
    if base.startswith(prefix) and base.endswith(suffix):
        return base[len(prefix):-len(suffix)]
    return base


def _load_trajectories(pattern):
    """
    trajectory_<model>.json files don't carry the model name inside them
    (they're just {"grad_norm_per_step": [...], "movement...": [...]}), so
    the model name is recovered from the filename. Flattened into a long
    dataframe: model, step (1-indexed), grad_norm, movement.
    """
    paths = sorted(glob.glob(pattern))
    rows = []
    for p in paths:
        model = _model_name_from_path(p, "trajectory_", ".json")
        try:
            with open(p) as f:
                traj = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
            continue
        grad_norms = traj.get("grad_norm_per_step", [])
        movements = traj.get("movement_from_random_start_per_step", [])
        n = max(len(grad_norms), len(movements))
        for i in range(n):
            rows.append({
                "model": model,
                "step": i + 1,
                "grad_norm": grad_norms[i] if i < len(grad_norms) else None,
                "movement": movements[i] if i < len(movements) else None,
            })
    if not rows:
        return pd.DataFrame(), paths
    return pd.DataFrame(rows), paths


def _parse_model_variant(model_name):
    for suffix in ("_int8_PTQ", "_int8_QAT", "_FP32"):
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)], suffix[1:]
    return model_name, "unknown"


def combine_results():
    df_results, results_paths = _load_and_concat(RESULTS_GLOB)
    df_sweep, sweep_paths = _load_and_concat(SWEEP_GLOB)
    df_ablation, ablation_paths = _load_and_concat(ABLATION_GLOB)
    df_layerwise, layerwise_paths = _load_and_concat(LAYERWISE_GLOB)
    df_trajectory, trajectory_paths = _load_trajectories(TRAJECTORY_GLOB)

    if df_results.empty:
        print(f"No files matched {RESULTS_GLOB}; nothing to combine for results.")
    else:
        df_results.to_csv(RESULTS_COMBINED_CSV, index=False)
        print(f"Combined {len(results_paths)} results file(s) -> {RESULTS_COMBINED_CSV}")
        print(df_results)

    if df_sweep.empty:
        print(f"No files matched {SWEEP_GLOB}; nothing to combine for sweep.")
    else:
        df_sweep.to_csv(SWEEP_COMBINED_CSV, index=False)
        print(f"Combined {len(sweep_paths)} sweep file(s) -> {SWEEP_COMBINED_CSV}")

    if df_ablation.empty:
        print(f"No files matched {ABLATION_GLOB}; nothing to combine for ablation.")
    else:
        df_ablation.to_csv(ABLATION_COMBINED_CSV, index=False)
        print(f"Combined {len(ablation_paths)} ablation file(s) -> {ABLATION_COMBINED_CSV}")

    if df_layerwise.empty:
        print(f"No files matched {LAYERWISE_GLOB}; nothing to combine for layerwise.")
    else:
        df_layerwise.to_csv(LAYERWISE_COMBINED_CSV, index=False)
        print(f"Combined {len(layerwise_paths)} layerwise file(s) -> {LAYERWISE_COMBINED_CSV}")

    if df_trajectory.empty:
        print(f"No files matched {TRAJECTORY_GLOB}; nothing to combine for trajectory.")
    else:
        df_trajectory.to_csv(TRAJECTORY_COMBINED_CSV, index=False)
        print(f"Combined {len(trajectory_paths)} trajectory file(s) -> {TRAJECTORY_COMBINED_CSV}")

    return df_results, df_sweep, df_ablation, df_layerwise, df_trajectory


def plot_accuracy(df_results):
    if df_results.empty:
        return

    acc_cols = [
        c
        for c in [
            "No Attack",
            "Random_Noise",
            "FGSM (W)",
            "PGD (W)",
            "AutoAttack (W/B)",
            "Transfer_from_FP32 (B)",
            "BPDA_PGD (W)",
        ]
        if c in df_results.columns
    ]

    if not acc_cols:
        print("No accuracy columns found in combined results; skipping accuracy plot.")
        return

    df_plot = df_results.melt(
        id_vars="model",
        value_vars=acc_cols,
        var_name="Attack",
        value_name="Accuracy",
    )

    plt.figure(figsize=(14, 6))
    sns.barplot(data=df_plot, x="model", y="Accuracy", hue="Attack")
    plt.xticks(rotation=45, ha="right")
    plt.title("Model Accuracy under Adversarial Attacks")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {ACCURACY_PLOT_PNG}")


def plot_sweep(df_sweep):
    if df_sweep.empty:
        return

    # NEW: Random_Noise_acc alongside PGD_acc/BPDA_acc lets you see, at every
    # epsilon, whether PGD is actually beating pure random search.
    value_cols = [c for c in ["PGD_acc", "BPDA_acc", "Random_Noise_acc"] if c in df_sweep.columns]
    if not value_cols:
        print("No sweep accuracy columns found; skipping sweep plot.")
        return

    df_plot = df_sweep.melt(
        id_vars=["model", "epsilon"],
        value_vars=value_cols,
        var_name="Attack",
        value_name="Accuracy",
    ).dropna(subset=["Accuracy"])

    if df_plot.empty:
        print("Sweep data has no non-null accuracy values; skipping sweep plot.")
        return

    df_plot["series"] = df_plot["model"] + " / " + df_plot["Attack"]

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_plot, x="epsilon", y="Accuracy", hue="series", marker="o")
    plt.title("Accuracy vs. Perturbation Budget (Epsilon Sweep)")
    plt.xlabel("Epsilon (L-infinity budget)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(SWEEP_PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {SWEEP_PLOT_PNG}")


def plot_ablation(df_ablation):
    """
    NEW. accuracy vs. PGD step count (0 = random start only) per model.
    A flat line across the whole 0-50 step range is a masking signature:
    more optimization steps aren't buying the attacker anything.
    symlog x-axis so step=0 is visible next to step=50.
    """
    if df_ablation.empty:
        print("No ablation data found; skipping ablation plot.")
        return

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_ablation, x="steps", y="acc", hue="model", marker="o")
    plt.xscale("symlog")
    plt.title("PGD Accuracy vs. Attack Step Budget\n(flat line = optimization steps are not finding real adversarial directions)")
    plt.xlabel("PGD steps (0 = random start only, i.e. Random_Noise)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(ABLATION_PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {ABLATION_PLOT_PNG}")


def plot_trajectory(df_trajectory):
    """
    NEW. Two panels: mean input-gradient norm per PGD step, and mean L_inf
    movement from the random start per PGD step, one line per model. Shows
    WHEN in the trajectory (not just whether) the gradient collapses.
    """
    if df_trajectory.empty:
        print("No trajectory data found; skipping trajectory plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.lineplot(data=df_trajectory, x="step", y="grad_norm", hue="model", ax=axes[0], legend=False)
    axes[0].set_yscale("symlog", linthresh=1e-6)
    axes[0].set_title("Input-gradient norm per PGD step")
    axes[0].set_xlabel("PGD step")
    axes[0].set_ylabel("mean grad norm")

    sns.lineplot(data=df_trajectory, x="step", y="movement", hue="model", ax=axes[1])
    axes[1].set_title("Cumulative movement from random start per PGD step")
    axes[1].set_xlabel("PGD step")
    axes[1].set_ylabel("mean L_inf distance from x_start")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")

    plt.tight_layout()
    plt.savefig(TRAJECTORY_PLOT_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {TRAJECTORY_PLOT_PNG}")


def plot_layerwise(df_layerwise):
    """
    NEW. Small-multiples grid, one subplot per model, showing grad_norm_hard
    vs grad_norm_ste across layer depth (x-axis = layer order as recorded,
    a proxy for network depth). Reveals whether the gradient dies at a single
    early bottleneck or decays gradually across depth.
    """
    if df_layerwise.empty:
        print("No layerwise data found; skipping layerwise plot.")
        return

    models = sorted(df_layerwise["model"].unique())
    n = len(models)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = df_layerwise[df_layerwise["model"] == model].reset_index(drop=True)
        layer_idx = range(len(sub))
        ax.plot(layer_idx, sub["grad_norm_hard"], marker="o", label="hard", color="tab:red")
        ax.plot(layer_idx, sub["grad_norm_ste"], marker="o", label="STE", color="tab:blue")
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("layer depth (quantized layers only)", fontsize=7)
        ax.set_ylabel("grad norm", fontsize=7)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    plt.suptitle("Gradient norm by layer depth: hard rounding vs. STE", y=1.0)
    plt.tight_layout()
    plt.savefig(LAYERWISE_PLOT_PNG, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Wrote {LAYERWISE_PLOT_PNG}")


def build_masking_scorecard(df_results):
    """
    NEW. One row per quantized model, pulling together every masking signal
    computed across the pipeline into a single sortable table + bar chart:
      - pgd_minus_transfer_gap : PGD acc - Transfer acc (large positive = masking)
      - pgd_minus_random_gap   : PGD acc - Random_Noise acc (near 0 = masking,
                                  since PGD is doing no better than blind noise)
      - frac_zero_grad_hard    : fraction of exactly-zero input gradients
      - plateau_fraction       : fraction of bitwise-identical logits under
                                  fine steps (autograd-free geometric signature)
    This is the single table to put in a paper's appendix / to sort models by
    "how masked is this."
    """
    if df_results.empty:
        return pd.DataFrame()

    needed = ["model", "PGD", "Transfer_from_FP32", "Random_Noise",
              "frac_zero_grad_hard", "plateau_fraction"]
    available = [c for c in needed if c in df_results.columns]
    if "model" not in available or "PGD" not in available:
        print("Missing core columns for masking scorecard; skipping.")
        return pd.DataFrame()

    df = df_results[available].copy()
    df[["arch", "variant"]] = df["model"].apply(lambda m: pd.Series(_parse_model_variant(m)))

    if "Transfer_from_FP32" in df.columns:
        df["pgd_minus_transfer_gap"] = df["PGD"] - df["Transfer_from_FP32"]
    if "Random_Noise" in df.columns:
        df["pgd_minus_random_gap"] = df["PGD"] - df["Random_Noise"]

    # only quantized models have a meaningful masking story
    df = df[df["variant"] != "FP32"].sort_values(
        by=[c for c in ["pgd_minus_transfer_gap"] if c in df.columns] or ["model"],
        ascending=False,
    )

    df.to_csv(SCORECARD_CSV, index=False)
    print(f"Wrote {SCORECARD_CSV}")

    plot_cols = [c for c in ["pgd_minus_transfer_gap", "pgd_minus_random_gap"] if c in df.columns]
    if plot_cols:
        df_plot = df.melt(id_vars="model", value_vars=plot_cols, var_name="metric", value_name="value")
        plt.figure(figsize=(14, 6))
        sns.barplot(data=df_plot, x="model", y="value", hue="metric")
        plt.axhline(0, color="k", lw=0.8)
        plt.xticks(rotation=45, ha="right")
        plt.title("Masking scorecard: gaps that should be ~0 for genuinely robust models")
        plt.tight_layout()
        plt.savefig(SCORECARD_PLOT_PNG, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Wrote {SCORECARD_PLOT_PNG}")

    return df


def main():
    df_results, df_sweep, df_ablation, df_layerwise, df_trajectory = combine_results()
    plot_accuracy(df_results)
    plot_sweep(df_sweep)
    plot_ablation(df_ablation)
    plot_trajectory(df_trajectory)
    plot_layerwise(df_layerwise)
    build_masking_scorecard(df_results)
    print("\nAll done.")


if __name__ == "__main__":
    main()