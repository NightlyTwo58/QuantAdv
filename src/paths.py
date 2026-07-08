"""Filesystem path helpers for QuantAdv artifacts."""
import os

from .config import DATA_DIR


def csv_path(model_name, type):
    return os.path.join(DATA_DIR, f"{type}_{model_name}.csv")


def json_path(model_name, type):
    return os.path.join(DATA_DIR, f"{type}_{model_name}.json")


def defense_summary_csv_path():
    return os.path.join(DATA_DIR, "defense_summary.csv")


# Backward-compatible names used by older callers.
def results_csv_path(model_name):
    return csv_path(model_name, "results")


def sweep_csv_path(model_name):
    return csv_path(model_name, "sweep")


def ablation_csv_path(model_name):
    return csv_path(model_name, "ablation")


def layerwise_csv_path(model_name):
    return csv_path(model_name, "layerwise")


def trajectory_json_path(model_name):
    return json_path(model_name, "trajectory")


def component_ablation_csv_path(model_name):
    return csv_path(model_name, "component_ablation")


# Combined / aggregated result paths used by the older combine module.
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



