"""Statistical helpers for QuantAdv result tables.

The experiment runner stores per-example correctness vectors so robustness
comparisons can use paired tests rather than independent-sample summaries.
SciPy provides the binomial confidence intervals and exact binomial tests used
here; this module keeps those details out of the model and plotting code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def _as_bool_vector(vector: Iterable[bool]) -> np.ndarray:
    """Return a one-dimensional boolean NumPy array."""
    return np.asarray(vector, dtype=bool).reshape(-1)


def binomial_interval(correct: int, total: int, confidence: float, method: str):
    """Return a SciPy binomial proportion interval as plain Python floats."""
    if total <= 0:
        return None, None
    interval = binomtest(correct, total).proportion_ci(
        confidence_level=confidence, method=method
    )
    return float(interval.low), float(interval.high)


def binomial_statistics(correct_vector, confidence: float = 0.95) -> dict[str, float]:
    """Summarize a correctness vector with exact and Wilson intervals."""
    vector = _as_bool_vector(correct_vector)
    total = int(vector.size)
    correct = int(vector.sum())
    wilson_low, wilson_high = binomial_interval(
        correct, total, confidence, method="wilson"
    )
    exact_low, exact_high = binomial_interval(
        correct, total, confidence, method="exact"
    )
    return {
        "n": total,
        "correct": correct,
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "wilson_pm": (
            (wilson_high - wilson_low) / 2 if wilson_low is not None else None
        ),
        "cp_low": exact_low,
        "cp_high": exact_high,
        "cp_pm": ((exact_high - exact_low) / 2 if exact_low is not None else None),
    }


def add_binomial_statistics(
    results: dict, metric: str, correct_vector, confidence: float = 0.95
) -> None:
    """Attach prefixed binomial statistics for ``metric`` to ``results``."""
    summary = binomial_statistics(correct_vector, confidence=confidence)
    results.update({f"{metric}_{key}": value for key, value in summary.items()})


def mcnemar_exact(vector_a, vector_b) -> dict[str, float]:
    """Run the exact two-sided McNemar test over paired correctness vectors."""
    a = _as_bool_vector(vector_a)
    b = _as_bool_vector(vector_b)
    if a.shape != b.shape:
        raise ValueError(
            f"McNemar vectors must have equal shape, got {a.shape} and {b.shape}"
        )

    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    discordant = a_only + b_only
    p_value = (
        float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def save_correctness_vectors(model_name: str, vectors: dict, output_dir) -> str:
    """Persist per-example correctness vectors for later paired tests."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in model_name)
    path = output_dir / f"{safe_name}.npz"
    np.savez_compressed(
        path, **{key: _as_bool_vector(value) for key, value in vectors.items()}
    )
    return str(path)


def add_paired_mcnemar_tests(
    df: pd.DataFrame,
    baseline_name: Callable[[str], str],
    path_column: str = "correctness_vectors_path",
    prefix_template: str = "McNemar_vs_FP32_{metric}",
) -> pd.DataFrame:
    """Add paired McNemar test columns comparing each row with its baseline row."""
    if df is None or df.empty or path_column not in df:
        return df

    df = df.copy()
    rows = df.set_index("model", drop=False)
    for index, row in df.iterrows():
        model = str(row["model"])
        baseline = baseline_name(model)
        if baseline not in rows.index:
            continue

        baseline_path = rows.loc[baseline, path_column]
        variant_path = row[path_column]
        if not all(
            isinstance(path, str) and Path(path).exists()
            for path in (baseline_path, variant_path)
        ):
            continue

        with np.load(baseline_path) as baseline_vectors, np.load(
            variant_path
        ) as variant_vectors:
            for metric in sorted(
                set(baseline_vectors.files) & set(variant_vectors.files)
            ):
                a = np.asarray(baseline_vectors[metric], dtype=bool)
                b = np.asarray(variant_vectors[metric], dtype=bool)
                if a.shape != b.shape:
                    continue
                test = mcnemar_exact(a, b)
                prefix = prefix_template.format(metric=metric)
                for key, value in test.items():
                    df.loc[index, f"{prefix}_{key}"] = value
    return df


def fp32_baseline_name(model_name: str) -> str:
    """Return the FP32 baseline row name for a QuantAdv model variant."""
    architecture = str(model_name).split("_", 1)[0]
    return f"{architecture}_FP32"
