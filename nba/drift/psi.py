"""Population stability index over decile bins, with an explicit missing-value bin.

Bins come from the reference: nine interior decile edges over every training row
(duplicates dropped, so a feature with few distinct values gets fewer bins) or, for a
feature with at most `bins` distinct values, one bin per value. The edges are fixed per
feature; the expected proportions vary by season day (reference.py). NaN is its own bin on
both sides. Proportions are smoothed by EPS before the log so an empty bin does not blow up;
PSI = sum((a - e) * ln(a / e)).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from nba import config

EPS = 1e-4


def make_bins(reference: pd.Series, bins: int = config.DRIFT_BINS) -> dict[str, Any]:
    """Bin definition for one feature from its reference values: decile edges, or one bin per
    value when there are at most `bins` distinct values. The expected proportions are
    computed separately (per season-day window) with `proportions`."""
    values = reference.dropna().to_numpy(dtype="float64")
    distinct = np.unique(values)
    if len(distinct) <= bins:
        return {"kind": "categorical", "values": [float(v) for v in distinct]}
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
    return {"kind": "quantile", "edges": [float(e) for e in edges]}


def n_bins(spec: dict[str, Any]) -> int:
    return len(spec["values"]) if spec["kind"] == "categorical" else len(spec["edges"]) + 1


def proportions(series: pd.Series, spec: dict[str, Any]) -> list[float]:
    """Share of rows per bin, the last entry being the missing share (NaN, plus values a
    categorical bin set has never seen). Sums to 1; all zeros for an empty series."""
    values = series.dropna().to_numpy(dtype="float64")
    n = int(len(series))
    if n == 0:
        return [0.0] * (n_bins(spec) + 1)
    missing = float((n - len(values)) / n)
    if spec["kind"] == "categorical":
        counts = pd.Series(values).value_counts()
        known = [float(counts.get(v, 0) / n) for v in spec["values"]]
        unseen = float(len(values) / n - sum(known))
        return known + [missing + unseen]
    idx = np.searchsorted(np.asarray(spec["edges"]), values, side="right")
    counts = np.bincount(idx, minlength=len(spec["edges"]) + 1)
    return [float(c / n) for c in counts] + [missing]


def psi_from(expected: list[float], actual: list[float]) -> float:
    total = 0.0
    for e, a in zip(expected, actual, strict=True):
        e2, a2 = max(e, EPS), max(a, EPS)
        total += (a2 - e2) * math.log(a2 / e2)
    return float(total)


def psi(window: pd.Series, spec: dict[str, Any], expected: list[float]) -> float:
    return psi_from(expected, proportions(window, spec))
