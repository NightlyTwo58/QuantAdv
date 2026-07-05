"""
Filesystem path helpers for per-model result files.

All per-run artifacts (results, sweeps, ablations, layerwise profiles,
trajectories, and their combined/aggregated counterparts) are namespaced by
model name and centralized here so that both the experiment runner and the
result-combining/plotting code agree on where files live.
"""
import os

from .config import DATA_DIR


def results_csv_path(model_name):
    return os.path.join(DATA_DIR, f"results_{model_name}.csv")


def sweep_csv_path(model_name):
    return os.path.join(DATA_DIR, f"sweep_{model_name}.csv")


def ablation_csv_path(model_name):
    return os.path.join(DATA_DIR, f"ablation_{model_name}.csv")


def layerwise_csv_path(model_name):
    return os.path.join(DATA_DIR, f"layerwise_{model_name}.csv")


def trajectory_json_path(model_name):
    return os.path.join(DATA_DIR, f"trajectory_{model_name}.json")


def component_ablation_csv_path(model_name):
    """weight-only vs activation-only vs both quantization ablation."""
    return os.path.join(DATA_DIR, f"component_ablation_{model_name}.csv")


# --- Combined / aggregated result paths (used by quantadv.combine) --------

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
