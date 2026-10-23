"""Pure screening/ranking logic, independently testable without Streamlit."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


LOW_IS_CONCERNING = frozenset({"bm", "eey", "gpa", "roc"})
HIGH_IS_CONCERNING = frozenset({"sta", "snoa", "pman"})
ALL_METRICS = LOW_IS_CONCERNING | HIGH_IS_CONCERNING


def stage_screen(
    entering: pd.DataFrame,
    metrics: list[str],
    keep_pct: int,
    sector_relative: bool,
    stage_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Retain the most concerning firms; audit each reason for exclusion.

    Ranking is relative to the firms *entering* each selected stage. Missing
    metrics never become zero. A company with insufficient valid sector peers
    is excluded only if sector-relative scoring was requested.
    """
    if not metrics or set(metrics) - ALL_METRICS:
        raise ValueError("Select one or more recognized accounting metrics")
    if not 1 <= keep_pct <= 100:
        raise ValueError("Retain percentage must be between 1 and 100")

    base = entering.copy()
    missing = base[metrics].isna().any(axis=1)
    eligible = base.loc[~missing].copy()

    if sector_relative:
        for metric in metrics:
            group = eligible.groupby("sector")[metric]
            n = group.transform("count")
            avg = group.transform("mean")
            eligible[f"_rank_input_{metric}"] = (eligible[metric] - avg).where(
                n.ge(3) & eligible["sector"].notna()
            )
        unavailable_peers = eligible[
            [f"_rank_input_{metric}" for metric in metrics]
        ].isna().any(axis=1)
        peer_count = int(unavailable_peers.sum())
        eligible = eligible.loc[~unavailable_peers].copy()
    else:
        peer_count = 0
        for metric in metrics:
            eligible[f"_rank_input_{metric}"] = eligible[metric]

    for metric in metrics:
        eligible[f"{stage_name}_{metric}_rank"] = eligible[
            f"_rank_input_{metric}"
        ].rank(ascending=metric in LOW_IS_CONCERNING, method="average")

    ranks = [f"{stage_name}_{metric}_rank" for metric in metrics]
    eligible[f"{stage_name}_rank_sum"] = eligible[ranks].sum(
        axis=1, min_count=len(metrics)
    )
    eligible = eligible.sort_values([f"{stage_name}_rank_sum", "ticker"])

    # The same rank sum gets the same strength; ticker order only resolves
    # actual cutoffs, not scores. All selected firms have positive scores.
    if not eligible.empty:
        eligible[f"{stage_name}_strength"] = (
            eligible[f"{stage_name}_rank_sum"].rank(
                ascending=False, method="average"
            ) / len(eligible)
        )
    else:
        eligible[f"{stage_name}_strength"] = pd.Series(dtype=float)

    n_keep = max(1, math.floor(len(eligible) * keep_pct / 100)) if len(eligible) else 0
    kept = eligible.iloc[:n_keep].copy()
    rejected = eligible.iloc[n_keep:].copy()
    audit = {
        "stage": stage_name,
        "entering": len(entering),
        "missing_metric": int(missing.sum()),
        "insufficient_sector": peer_count,
        "excluded_by_cutoff": len(rejected),
        "retained": len(kept),
        "metrics": metrics,
    }
    assert sum(audit[key] for key in (
        "missing_metric", "insufficient_sector", "excluded_by_cutoff", "retained"
    )) == audit["entering"]
    return kept, rejected, audit


def concern_percentile(values: pd.Series, metric: str, value: float) -> float:
    """Return 0-100 relative concern rank; higher means more concerning."""
    if metric not in ALL_METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty or not np.isfinite(value):
        return float("nan")
    # Include the selected firm's value even when it is not present in a
    # comparison sample; this avoids making an invalid inference from a gap.
    ties = 0.5 * valid.eq(value).sum()
    if metric in LOW_IS_CONCERNING:
        return 100.0 * (valid.gt(value).sum() + ties) / len(valid)
    return 100.0 * (valid.lt(value).sum() + ties) / len(valid)
