"""The decision policy: line-free directional calls and their evaluation (ADR-0001, ADR-0006).

A call for one (player, game, target) compares the model's prediction with the last-10 mean
the same slate row carries (`edge = prediction - baseline_last10`):

    over      edge >  threshold
    under     edge < -threshold
    no_call   otherwise

Once the box score exists the call resolves against the same last-10 mean: `hit` when the
actual lands on the called side, `miss` on the other side, `push` when actual equals the
mean (no side). Two causal baselines are scored on the rows the model calls: a coin flip
(0.5 by definition) and the sign of `season_mean - last10`, the direction a mean-reverting
forecaster would call from the player's own season-to-date mean, which is also available
before tip-off.

The threshold per target and population is chosen on the holdout-season replay rows: the
largest grid value whose coverage (share of rows called) is still at least
`config.POLICY_MIN_COVERAGE`. Bands are residual quantiles (`actual - prediction`) on the
same rows. Both are in-sample on that season; the artifact says so, publishes the whole
coverage curve so a reader can pick another point, and never asserts that the model wins.

This module is pure pandas over one frame; `evaluate.py` feeds it the gold marts and
writes the artifact, `decide.py` applies a committed artifact to a slate, and the dbt marts
(`fct_decision_policy`, `mart_policy_metrics`, `mart_policy_sweep`) reimplement the call and
outcome rules in SQL and reconcile to the artifact in a singular test.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from nba import config

CALLS: tuple[str, ...] = ("over", "under", "no_call")
OUTCOMES: tuple[str, ...] = ("hit", "miss", "push")
POPULATIONS: tuple[str, ...] = ("min10", "all")
POPULATION_TEXT: dict[str, str] = {
    "min10": "the training population: replayed rows with a box score, at least 10 minutes "
    "and both baselines defined",
    "all": "every replayed row with a box score and a last-10 baseline (all rows)",
}
BAND_LEVELS: dict[str, tuple[float, float]] = {"50": (0.25, 0.75), "80": (0.10, 0.90)}


def threshold_grid(target: str) -> list[float]:
    """The grid the threshold is chosen from, rounded to two decimals so SQL and pandas agree."""
    start, stop, step = config.POLICY_THRESHOLD_GRID[target]
    n = int(round((stop - start) / step))
    return [round(start + k * step, 2) for k in range(n + 1)]


def quantile_key(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def calls(edge: pd.Series, threshold: float) -> pd.Series:
    out = pd.Series("no_call", index=edge.index, dtype="object")
    out[edge > threshold] = "over"
    out[edge < -threshold] = "under"
    out[edge.isna()] = "no_call"
    return out


def sign_calls(diff: pd.Series) -> pd.Series:
    """A baseline's call from the sign of its own edge (no threshold; NaN or zero = no_call)."""
    return calls(diff, 0.0)


def outcomes(call: pd.Series, actual: pd.Series, baseline: pd.Series) -> pd.Series:
    """hit / miss / push per called row; None for no_call or a missing actual."""
    side = np.sign(actual - baseline)
    out = pd.Series([None] * len(call), index=call.index, dtype="object")
    resolved = call.isin(("over", "under")) & actual.notna() & baseline.notna()
    called_side = call.map({"over": 1.0, "under": -1.0})
    out[resolved & (side == 0)] = "push"
    out[resolved & (side != 0) & (side == called_side)] = "hit"
    out[resolved & (side != 0) & (side != called_side)] = "miss"
    return out


def _rate(hits: int, n: int) -> float | None:
    return hits / n if n else None


def curve_point(
    edge: pd.Series,
    actual: pd.Series,
    baseline: pd.Series,
    season_diff: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    """One row of the coverage curve: the model policy at `threshold` and both baselines."""
    n = int(len(edge))
    call = calls(edge, threshold)
    out = outcomes(call, actual, baseline)
    n_called = int(call.isin(("over", "under")).sum())
    n_hit, n_miss, n_push = (int((out == k).sum()) for k in ("hit", "miss", "push"))
    n_resolved = n_hit + n_miss
    # Season-mean sign on exactly the rows the model resolved: where the baseline has no side
    # (a tie, season mean == last-10 mean, or no season mean on a season debut) it is scored
    # as a coin flip, 0.5, so the comparison has one n. Both abstention kinds are counted.
    same = out.isin(("hit", "miss"))
    sm_call = sign_calls(season_diff)
    sm_out = outcomes(sm_call.where(same, "no_call"), actual, baseline)
    sm_hit, sm_miss = int((sm_out == "hit").sum()), int((sm_out == "miss").sum())
    sm_missing = int((same & season_diff.isna()).sum())
    sm_tie = int((same & (season_diff == 0)).sum())
    assert sm_hit + sm_miss + sm_missing + sm_tie == n_resolved
    # Season-mean sign as a policy with its own threshold on the same population.
    own_call = calls(season_diff, threshold)
    own_out = outcomes(own_call, actual, baseline)
    own_hit, own_miss = int((own_out == "hit").sum()), int((own_out == "miss").sum())
    return {
        "threshold": threshold,
        "n_called": n_called,
        "coverage": n_called / n if n else 0.0,
        "n_resolved": n_resolved,
        "n_push": n_push,
        "n_hit": n_hit,
        "hit_rate": _rate(n_hit, n_resolved),
        "net_correct": n_hit - n_miss,
        "season_mean_same_rows_n": n_resolved,
        "season_mean_same_rows_n_hit": sm_hit,
        "season_mean_same_rows_n_miss": sm_miss,
        "season_mean_same_rows_n_tie": sm_tie,
        "season_mean_same_rows_n_missing": sm_missing,
        "season_mean_same_rows_hit_rate": _rate(sm_hit + 0.5 * (sm_tie + sm_missing), n_resolved),
        "season_mean_own_n_called": int(own_call.isin(("over", "under")).sum()),
        "season_mean_own_hit_rate": _rate(own_hit, own_hit + own_miss),
    }


def coverage_curve(frame: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    edge = frame[f"pred_{target}"] - frame[f"baseline_last10_{target}"]
    season_diff = frame[f"baseline_season_{target}"] - frame[f"baseline_last10_{target}"]
    return [
        curve_point(
            edge,
            frame[f"actual_{target}"],
            frame[f"baseline_last10_{target}"],
            season_diff,
            t,
        )
        for t in threshold_grid(target)
    ]


def select_threshold(
    curve: list[dict[str, Any]], min_coverage: float = config.POLICY_MIN_COVERAGE
) -> float:
    """The largest threshold whose coverage is still at least min_coverage (else the smallest)."""
    eligible = [p["threshold"] for p in curve if p["coverage"] >= min_coverage]
    return max(eligible) if eligible else min(p["threshold"] for p in curve)


def bands(residual: pd.Series, quantiles: tuple[float, ...] = config.POLICY_BAND_QUANTILES) -> dict:
    """Residual quantiles (actual - prediction), linear interpolation, keyed q10 / q25 / ..."""
    r = residual.dropna().to_numpy(dtype="float64")
    return {quantile_key(q): float(np.quantile(r, q)) for q in quantiles}


def band_coverage(residual: pd.Series, quantiles: dict[str, float]) -> dict[str, float | None]:
    """Share of rows whose residual falls inside each band (in-sample: nominal by construction)."""
    r = residual.dropna()
    out: dict[str, float | None] = {}
    for level, (lo, hi) in BAND_LEVELS.items():
        low, high = quantiles[quantile_key(lo)], quantiles[quantile_key(hi)]
        out[f"coverage_{level}"] = float(((r >= low) & (r <= high)).mean()) if len(r) else None
    return out


def coin_flip_half_width(n: int) -> float | None:
    """Half of the 95% interval a fair coin gives at n resolved calls."""
    return 1.96 * math.sqrt(0.25 / n) if n else None


def verdict(target: str, population: str, chosen: dict[str, Any]) -> tuple[bool, str]:
    """Does the model policy beat both baselines at the chosen threshold, and one sentence why."""
    hit = chosen["hit_rate"]
    sm = chosen["season_mean_same_rows_hit_rate"]
    half = coin_flip_half_width(chosen["n_resolved"])
    if hit is None or sm is None or half is None:
        return False, f"No resolved calls for {target} on {population}; no verdict."
    beats_coin = hit - 0.5 > half
    beats_season = hit > sm
    where = "the training population" if population == "min10" else "all rows"
    if beats_coin and beats_season:
        text = (
            f"On {where} the model's {target} calls hit {hit:.1%} of {chosen['n_resolved']:,} "
            f"resolved calls, above the season-mean sign ({sm:.1%} on the same rows) and outside "
            f"the coin flip's 95% band (0.5 ± {half:.3f})."
        )
        return True, text
    reasons = []
    if not beats_season:
        reasons.append(f"the season-mean sign hits {sm:.1%} on the same rows")
    if not beats_coin:
        reasons.append(f"the rate is inside the coin flip's 95% band (0.5 ± {half:.3f})")
    text = (
        f"On {where} the model's {target} calls hit {hit:.1%} of {chosen['n_resolved']:,} "
        f"resolved calls, but {' and '.join(reasons)}; the model policy is not recommended there."
    )
    return False, text


def evaluate_target(frame: pd.DataFrame, target: str, population: str) -> dict[str, Any]:
    curve = coverage_curve(frame, target)
    threshold = select_threshold(curve)
    chosen = next(p for p in curve if p["threshold"] == threshold)
    residual = frame[f"actual_{target}"] - frame[f"pred_{target}"]
    quantiles = bands(residual)
    beats, text = verdict(target, population, chosen)
    return {
        "n": int(len(frame)),
        "threshold": threshold,
        "n_called": chosen["n_called"],
        "coverage": chosen["coverage"],
        "n_resolved": chosen["n_resolved"],
        "n_push": chosen["n_push"],
        "n_hit": chosen["n_hit"],
        "hit_rate": chosen["hit_rate"],
        "net_correct": chosen["net_correct"],
        "baselines": {
            "coin_flip": {
                "hit_rate": 0.5,
                "n": chosen["n_resolved"],
                "half_width_95": coin_flip_half_width(chosen["n_resolved"]),
            },
            "season_mean_sign": {
                "n": chosen["season_mean_same_rows_n"],
                "n_hit": chosen["season_mean_same_rows_n_hit"],
                "n_miss": chosen["season_mean_same_rows_n_miss"],
                "n_tie": chosen["season_mean_same_rows_n_tie"],
                "n_missing": chosen["season_mean_same_rows_n_missing"],
                "hit_rate": chosen["season_mean_same_rows_hit_rate"],
                "same_rows": True,
                "abstentions_scored_as": 0.5,
                "own_threshold_n_called": chosen["season_mean_own_n_called"],
                "own_threshold_hit_rate": chosen["season_mean_own_hit_rate"],
            },
        },
        "model_beats_both": beats,
        "verdict": text,
        "bands": {"quantiles": quantiles, **band_coverage(residual, quantiles)},
        "coverage_curve": curve,
    }


def evaluate_population(frame: pd.DataFrame, population: str) -> dict[str, Any]:
    return {
        "population": population,
        "description": POPULATION_TEXT[population],
        "n": int(len(frame)),
        "targets": {t: evaluate_target(frame, t, population) for t in config.TARGETS},
    }


def population_frames(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """`all` = every resolvable row; `min10` = the rows flagged in the training population."""
    resolvable = rows[rows["has_actual"] & rows["baseline_defined"]]
    return {
        "min10": resolvable[resolvable["in_metrics_population"]].reset_index(drop=True),
        "all": resolvable.reset_index(drop=True),
    }


DEFINITIONS: dict[str, str] = {
    "edge": "prediction - baseline_last10, both from the slate row scored before tip-off",
    "call": "over if edge > threshold; under if edge < -threshold; else no_call",
    "outcome": "hit if actual is on the called side of baseline_last10; miss if on the other; "
    "push if actual == baseline_last10 (excluded from hit_rate)",
    "hit_rate": "n_hit / (n_hit + n_miss) over the rows the model called",
    "coverage": "n_called / n, the share of the population's rows the policy calls",
    "net_correct": "n_hit - n_miss",
    "coin_flip": "0.5 by definition; half_width_95 = 1.96 * sqrt(0.25 / n_resolved)",
    "season_mean_sign": "over if season_mean - baseline_last10 > 0, under if < 0; scored on "
    "exactly the n_resolved rows the model resolved (n == n_resolved): a tie (season_mean == "
    "baseline_last10, n_tie) or a missing season mean (season debut, n_missing) has no side and "
    "is scored as a coin flip, 0.5; hit_rate = (n_hit + 0.5 * (n_tie + n_missing)) / n. "
    "own_threshold_*: the same sign as a policy with the same threshold applied to its own "
    "edge, on its own rows (its own n_called)",
    "season_mean": "the player's mean of the stat over earlier games of the same season "
    "(the feature module's <stat>_mean_season, recomputed in gold.fct_player_game)",
    "threshold_selection": "largest grid threshold with coverage >= min_coverage, chosen on "
    "these rows (in-sample)",
    "bands": "quantiles of actual - prediction on these rows (in-sample); band_50 = q25..q75, "
    "band_80 = q10..q90 around the prediction",
}


def build_artifact(
    rows: pd.DataFrame,
    season: str,
    *,
    generated_at: str,
    git_sha: str,
    input_description: dict[str, Any],
) -> dict[str, Any]:
    frames = population_frames(rows)
    identity = rows.iloc[0] if len(rows) else None
    return {
        "season": season,
        "kind": "policy_replay_evaluation",
        "generated_at": generated_at,
        "git_sha": git_sha,
        "model_revision": str(identity["model_revision"]) if identity is not None else None,
        "model_commit": config.MODEL_COMMIT,
        "feature_version": str(identity["feature_version"]) if identity is not None else None,
        "dataset_revision": str(identity["dataset_revision"]) if identity is not None else None,
        "input": input_description,
        "in_sample": True,
        "in_sample_note": (
            f"Thresholds and bands were chosen on the {season} replay rows they are evaluated "
            "on; the hit rates are in-sample. The coverage curve is published so any other "
            "threshold can be read off; the next untouched season is the out-of-sample test."
        ),
        "threshold_grid": {t: threshold_grid(t) for t in config.TARGETS},
        "min_coverage": config.POLICY_MIN_COVERAGE,
        "band_quantiles": list(config.POLICY_BAND_QUANTILES),
        "definitions": DEFINITIONS,
        "populations": {p: evaluate_population(frames[p], p) for p in POPULATIONS},
    }


def validate_artifact(artifact: dict[str, Any]) -> list[str]:
    """Structural and arithmetic checks; returns the list of problems (empty = valid)."""
    problems: list[str] = []
    for key in (
        "season",
        "generated_at",
        "git_sha",
        "model_revision",
        "in_sample",
        "threshold_grid",
        "min_coverage",
        "definitions",
        "populations",
    ):
        if key not in artifact:
            problems.append(f"missing key {key}")
    if artifact.get("in_sample") is not True:
        problems.append("in_sample must be true")
    pops = artifact.get("populations", {})
    for p in POPULATIONS:
        if p not in pops:
            problems.append(f"missing population {p}")
            continue
        block = pops[p]
        for t in config.TARGETS:
            tb = block.get("targets", {}).get(t)
            if tb is None:
                problems.append(f"{p}.{t}: missing")
                continue
            if tb["n"] != block["n"]:
                problems.append(f"{p}.{t}: n {tb['n']} != population n {block['n']}")
            if tb["threshold"] not in artifact["threshold_grid"][t]:
                problems.append(f"{p}.{t}: threshold {tb['threshold']} not on the grid")
            chosen = [c for c in tb["coverage_curve"] if c["threshold"] == tb["threshold"]]
            if len(chosen) != 1 or chosen[0]["hit_rate"] != tb["hit_rate"]:
                problems.append(f"{p}.{t}: chosen threshold does not match its curve point")
            if tb["n_resolved"] != tb["n_hit"] + (tb["n_hit"] - tb["net_correct"]):
                problems.append(f"{p}.{t}: n_resolved != n_hit + n_miss")
            if tb["n_called"] != tb["n_resolved"] + tb["n_push"]:
                problems.append(f"{p}.{t}: n_called != n_resolved + n_push")
            sm = tb["baselines"]["season_mean_sign"]
            if sm["n"] != tb["n_resolved"] or sm["n"] != (
                sm["n_hit"] + sm["n_miss"] + sm["n_tie"] + sm["n_missing"]
            ):
                problems.append(f"{p}.{t}: season-mean sign is not scored on n_resolved rows")
            if tb["coverage"] < artifact["min_coverage"] and tb["threshold"] != min(
                artifact["threshold_grid"][t]
            ):
                problems.append(f"{p}.{t}: coverage below min_coverage at a non-minimal threshold")
            q = tb["bands"]["quantiles"]
            if not (q["q10"] <= q["q25"] <= q["q75"] <= q["q90"]):
                problems.append(f"{p}.{t}: band quantiles not ordered")
    return problems


def site_summary(artifact: dict[str, Any], source_file: str, source_sha256: str) -> dict:
    """What the /decisions page imports: per population and target, the chosen policy, both
    baselines, the bands and the coverage curve; no row-level data."""
    pops = {}
    for p, block in artifact["populations"].items():
        targets = {}
        for t, tb in block["targets"].items():
            targets[t] = {
                k: tb[k]
                for k in (
                    "n",
                    "threshold",
                    "n_called",
                    "coverage",
                    "n_resolved",
                    "n_push",
                    "n_hit",
                    "hit_rate",
                    "baselines",
                    "model_beats_both",
                    "verdict",
                    "bands",
                )
            }
            targets[t]["coverage_curve"] = [
                {
                    k: c[k]
                    for k in (
                        "threshold",
                        "coverage",
                        "n_resolved",
                        "hit_rate",
                        "season_mean_same_rows_hit_rate",
                        "season_mean_own_hit_rate",
                    )
                }
                for c in tb["coverage_curve"]
            ]
        pops[p] = {
            "description": block["description"],
            "n": block["n"],
            "targets": targets,
            "model_beats_both_everywhere": all(
                tb["model_beats_both"] for tb in block["targets"].values()
            ),
        }
    return {
        "season": artifact["season"],
        "generated_at": artifact["generated_at"],
        "git_sha": artifact["git_sha"],
        "model_revision": artifact["model_revision"],
        "model_commit": artifact["model_commit"],
        "in_sample": artifact["in_sample"],
        "in_sample_note": artifact["in_sample_note"],
        "min_coverage": artifact["min_coverage"],
        "source_file": source_file,
        "source_sha256": source_sha256,
        "definitions": artifact["definitions"],
        "populations": pops,
    }


def apply_policy(
    predictions: pd.DataFrame, artifact: dict[str, Any], population: str, target: str
) -> pd.DataFrame:
    """Edge, call and bands for slate rows under one population's policy (no actuals needed)."""
    tb = artifact["populations"][population]["targets"][target]
    q = tb["bands"]["quantiles"]
    pred = predictions[f"pred_{target}"]
    edge = pred - predictions[f"{target}_mean_last10"]
    out = pd.DataFrame(index=predictions.index)
    out["edge"] = edge
    out["call"] = calls(edge, tb["threshold"])
    for level, (lo, hi) in BAND_LEVELS.items():
        out[f"low_{level}"] = pred + q[quantile_key(lo)]
        out[f"high_{level}"] = pred + q[quantile_key(hi)]
    return out
