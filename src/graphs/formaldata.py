#!/usr/bin/env python3
"""Build consolidated formal figures from every QuantAdv result table.

Combination and derived-stat logic stays in :mod:`data`.  This module groups
related metrics and creates one figure per group, always comparing every model
and quantization for which that group is available.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import data as report_data
from ..config import DATA_DIR, PLOT_BBOX_INCHES, PLOT_DPI


MetricPredicate = Callable[[str, pd.DataFrame], bool]


@dataclass(frozen=True)
class MetricGroup:
    """A semantic group rendered as one consolidated figure."""

    name: str
    title: str
    predicate: MetricPredicate


@dataclass(frozen=True)
class TableSpec:
    """Plot structure and grouping rules for one combined artifact table."""

    groups: tuple[MetricGroup, ...]
    x: str | None = None
    dimensions: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    distribution: bool = False


def _contains(*tokens: str) -> MetricPredicate:
    return lambda column, _frame: any(token in column.lower() for token in tokens)


def _starts(*prefixes: str) -> MetricPredicate:
    return lambda column, _frame: column.lower().startswith(prefixes)


def _ends(*suffixes: str) -> MetricPredicate:
    return lambda column, _frame: column.lower().endswith(suffixes)


def _bounded(column: str, frame: pd.DataFrame) -> bool:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return bool(len(values) and values.between(0, 1).all())


def _accuracy_metric(column: str, frame: pd.DataFrame) -> bool:
    """Recognize attack/clean accuracies without a brittle attack-name list."""
    lower = column.lower()
    if lower.endswith(("_std", "_pm", "_low", "_high", "_n", "_correct")):
        return False
    if any(
        token in lower
        for token in ("mcnemar", "grad", "plateau", "masking", "retention")
    ):
        return False
    bases_with_vectors = {
        name[: -len("_n")].lower()
        for name in frame.columns
        if name.lower().endswith("_n")
    }
    base = re.sub(r"_(mean|acc)$", "", lower)
    return (
        "acc" in lower
        or base in bases_with_vectors
        or ("boundary" not in lower and _bounded(column, frame))
    )


def _uncertainty_metric(column: str, _frame: pd.DataFrame) -> bool:
    return bool(
        re.search(r"_(n|correct|std|wilson_(low|high|pm)|cp_(low|high|pm))$", column)
    )


RESULT_GROUPS = (
    MetricGroup("accuracy", "Clean and Adversarial Accuracy", _accuracy_metric),
    MetricGroup("uncertainty", "Accuracy Uncertainty and Sample Counts", _uncertainty_metric),
    MetricGroup("robustness", "Derived Robustness and Masking Scores", _contains("robustness", "robust_acc", "masking_score", "masking_gap")),
    MetricGroup("gradients", "Gradient and Staircase Diagnostics", _contains("grad", "plateau", "staircase")),
    MetricGroup("boundary", "Decision-Boundary Diagnostics", _starts("boundary_")),
    MetricGroup("mcnemar", "Paired McNemar Comparisons", _starts("mcnemar_")),
)


TABLE_SPECS: dict[str, TableSpec] = {
    "results": TableSpec(RESULT_GROUPS),
    "sweep": TableSpec(
        (
            MetricGroup("accuracy", "Accuracy Across Perturbation Budgets", _contains("acc", "mean")),
            MetricGroup("variability", "Sweep Variability", _ends("_std")),
            MetricGroup("masking_gap", "PGD–BPDA Gap Across Budgets", _contains("minus", "gap")),
        ),
        x="epsilon",
    ),
    "ablation": TableSpec(
        (MetricGroup("accuracy", "Accuracy Across Attack Steps", _accuracy_metric),),
        x="steps",
        dimensions=("attack",),
    ),
    "trajectory": TableSpec(
        (MetricGroup("trajectory", "PGD Trajectory Diagnostics", lambda _c, _f: True),),
        x="step",
    ),
    "layerwise": TableSpec(
        (MetricGroup("gradients", "Layerwise Gradient Propagation", lambda _c, _f: True),),
        x="layer",
    ),
    "component_ablation": TableSpec(
        (MetricGroup("accuracy", "Quantization Component Ablation", _accuracy_metric),),
        dimensions=("config",),
    ),
    "performance": TableSpec(
        (
            MetricGroup("time", "Runtime and CPU Cost", _contains("second", "cpu_core")),
            MetricGroup("memory", "CPU and CUDA Memory", _contains("mib")),
            MetricGroup("model_size", "Model Size and Precision", _contains("parameter", "weight_bits")),
        )
    ),
    "dither": TableSpec(
        (MetricGroup("accuracy", "Chaotic-Dither Accuracy", _accuracy_metric),),
        x="dither_amplitude",
        dimensions=("bits",),
    ),
    "chunk_quant": TableSpec(
        (MetricGroup("accuracy", "Chunk-Quantization Accuracy", _accuracy_metric),),
        x="chunk_id",
        dimensions=("layers",),
    ),
    "margins": TableSpec(
        (MetricGroup("margins", "Clean and Adversarial Confidence Margins", _contains("margin")),),
        dimensions=("kind",),
        excluded=("index",),
        distribution=True,
    ),
    "defense": TableSpec(
        (MetricGroup("accuracy_rates", "Defense Accuracy and Detection Rates", _accuracy_metric),),
        dimensions=("defense",),
    ),
}


def _numeric_metrics(frame: pd.DataFrame, spec: TableSpec) -> list[str]:
    excluded = {spec.x, *spec.excluded}
    return [
        column
        for column in frame.columns
        if column not in excluded
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
        and not pd.api.types.is_bool_dtype(frame[column])
    ]


def group_metrics(frame: pd.DataFrame, spec: TableSpec) -> dict[str, list[str]]:
    """Assign every numeric metric to its first semantic group or ``other``."""
    available = _numeric_metrics(frame, spec)
    remaining = set(available)
    grouped: dict[str, list[str]] = {}
    for group in spec.groups:
        matches = [c for c in available if c in remaining and group.predicate(c, frame)]
        if matches:
            grouped[group.name] = matches
            remaining.difference_update(matches)
    if remaining:
        grouped["other"] = [c for c in available if c in remaining]
    return grouped


def _labels(frame: pd.DataFrame, dimensions: Iterable[str]) -> pd.Series:
    columns = [c for c in ("model", *dimensions) if c in frame.columns]
    if not columns:
        return pd.Series("All", index=frame.index)
    return frame[columns].fillna("NA").astype(str).agg(" | ".join, axis=1)


def _long(frame: pd.DataFrame, metrics: list[str], spec: TableSpec) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["Series"] = _labels(prepared, spec.dimensions)
    ids = ["Series", *([spec.x] if spec.x and spec.x in prepared else [])]
    long = prepared.melt(ids, metrics, "Metric", "Value")
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce")
    return long.dropna(subset=["Value"])


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _plot_grouped_bars(ax: plt.Axes, long: pd.DataFrame) -> None:
    sns.barplot(data=long, x="Series", y="Value", hue="Metric", errorbar=None, ax=ax)
    _style_axes(ax)


def _plot_lines(ax: plt.Axes, long: pd.DataFrame, x: str) -> None:
    long = long.copy()
    long["Series / Metric"] = long["Series"] + " | " + long["Metric"]
    sns.lineplot(
        data=long,
        x=x,
        y="Value",
        hue="Series / Metric",
        marker="o",
        estimator=None,
        ax=ax,
    )
    ax.grid(linestyle="--", alpha=0.3)


def _plot_distribution(ax: plt.Axes, long: pd.DataFrame) -> None:
    sns.boxplot(data=long, x="Series", y="Value", hue="Metric", ax=ax)
    _style_axes(ax)


def _plot_heatmap(long: pd.DataFrame, title: str) -> plt.Figure:
    """Render a dense metric group without losing models or metric columns."""
    matrix = long.pivot_table(
        index="Series", columns="Metric", values="Value", aggfunc="mean", sort=False
    )
    width = max(12, min(30, 0.34 * len(matrix.columns) + 6))
    height = max(5, min(18, 0.45 * len(matrix.index) + 3))
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(matrix, cmap="viridis", mask=matrix.isna(), ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Statistic")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=60)
    return fig


def _needs_subplots(long: pd.DataFrame) -> bool:
    """Use small multiples when metric magnitudes would make bars unreadable."""
    # Accuracy, rates, proportions, and signed gains share a meaningful scale
    # even when some attacks drive accuracy very close to zero.
    if long["Value"].between(-1, 1).all():
        return False
    scales = long.groupby("Metric")["Value"].apply(lambda s: s.abs().max()).replace(0, np.nan)
    return len(scales) > 1 and scales.max() / scales.min() > 100


def _plot_small_multiples(long: pd.DataFrame, title: str) -> plt.Figure:
    metrics = list(long["Metric"].drop_duplicates())
    columns = min(3, len(metrics))
    rows = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        subset = long[long["Metric"] == metric]
        sns.barplot(data=subset, x="Series", y="Value", errorbar=None, ax=ax, color="#4C72B0")
        ax.set_title(metric.replace("_", " "))
        _style_axes(ax)
    for ax in axes.flat[len(metrics) :]:
        ax.axis("off")
    fig.suptitle(title)
    return fig


def plot_metric_group(
    frame: pd.DataFrame,
    metrics: list[str],
    spec: TableSpec,
    title: str,
    output: Path,
) -> None:
    """Render all metrics and all models for one semantic group in one figure."""
    long = _long(frame, metrics, spec)
    if long.empty:
        return
    if not spec.x and not spec.distribution and len(metrics) > 20:
        fig = _plot_heatmap(long, title)
    elif not spec.x and not spec.distribution and _needs_subplots(long):
        fig = _plot_small_multiples(long, title)
    else:
        width = max(10, min(24, 5 + 0.45 * long["Series"].nunique()))
        fig, ax = plt.subplots(figsize=(width, 6.5))
        if spec.distribution:
            _plot_distribution(ax, long)
        elif spec.x and spec.x in long:
            _plot_lines(ax, long, spec.x)
        else:
            _plot_grouped_bars(ax, long)
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=PLOT_DPI, bbox_inches=PLOT_BBOX_INCHES)
    plt.close(fig)


def plot_table_groups(frame: pd.DataFrame, table: str, root: Path) -> list[Path]:
    if frame is None or frame.empty:
        return []
    spec = TABLE_SPECS.get(
        table,
        TableSpec((MetricGroup("all", table.replace("_", " ").title(), lambda _c, _f: True),)),
    )
    groups = group_metrics(frame, spec)
    titles = {group.name: group.title for group in spec.groups}
    outputs = []
    for name, metrics in groups.items():
        output = root / table / f"{name}.png"
        plot_metric_group(
            frame,
            metrics,
            spec,
            titles.get(name, f"{table.replace('_', ' ').title()}: Other Diagnostics"),
            output,
        )
        outputs.append(output)
    return outputs


def plot_all(tables: dict[str, pd.DataFrame], output_dir: Path = DATA_DIR) -> list[Path]:
    """Create one consolidated figure per semantic group."""
    root = Path(output_dir) / "formal_figures"
    outputs = [
        output
        for table, frame in tables.items()
        for output in plot_table_groups(frame, table, root)
    ]
    # Remove obsolete per-stat figures created by older versions of this module.
    expected = {path.resolve() for path in outputs}
    for path in root.rglob("*.png") if root.exists() else ():
        if path.resolve() not in expected:
            path.unlink()
    return outputs


def print_report(tables: dict[str, pd.DataFrame], plots: Iterable[Path] = ()) -> None:
    rows = []
    for table, frame in tables.items():
        spec = TABLE_SPECS.get(table)
        rows.append(
            {
                "table": table,
                "rows": len(frame),
                "figure_groups": len(group_metrics(frame, spec)) if spec and not frame.empty else 0,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"generated {len(list(plots))} consolidated formal figures")


def combine_all(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    return report_data.combine_all(Path(data_dir))


def generate_reports(
    data_dir: Path = DATA_DIR, *, plots: bool = True, summary: bool = True
) -> dict[str, pd.DataFrame]:
    tables = combine_all(Path(data_dir))
    outputs = plot_all(tables, Path(data_dir)) if plots else []
    if summary:
        print_report(tables, outputs)
    return tables


if __name__ == "__main__":
    generate_reports()
