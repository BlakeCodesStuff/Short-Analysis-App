"""Prepare a previously collected Yahoo snapshot for SHORT app deployment.

This is an offline step: no API calls, no new financial calculations and no
changes to the original Colab downloads. It implements the same objective
Beneish index exclusions as build_universe.py on future data collections.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from metrics import identify_beneish_index_exclusions


METRIC_COLUMNS = ("bm", "eey", "sta", "snoa", "pman", "gpa", "roc")


def prepare_snapshot(
    universe_path: Path,
    snapshot_path: Path,
    output_dir: Path,
    history_path: Path | None = None,
    issues_path: Path | None = None,
) -> dict:
    universe = pd.read_csv(universe_path)
    if universe.empty or universe["ticker"].duplicated().any():
        raise ValueError("Source universe must contain unique, nonempty tickers")
    original_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    exclusions = identify_beneish_index_exclusions(universe)
    cleaned = universe.loc[~universe["ticker"].isin(exclusions["ticker"])].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_dir / "short_universe.csv", index=False)
    exclusions.to_csv(output_dir / "quality_exclusions.csv", index=False)

    updated_snapshot = dict(original_snapshot)
    updated_snapshot.update({
        "nonfinancial_companies_before_quality_review": len(universe),
        "beneish_index_exclusions": len(exclusions),
        "nonfinancial_companies": len(cleaned),
        "screening_exclusion_rule": "Exclude nonpositive available Beneish GMI or SGAI",
        "coverage_by_metric": {column: int(cleaned[column].notna().sum())
                               for column in METRIC_COLUMNS},
    })
    if "valuation_currency_valid" in cleaned:
        updated_snapshot["currency_mismatches_or_unverified"] = int(
            cleaned["valuation_currency_valid"].eq(False).sum()
        )
    (output_dir / "snapshot.json").write_text(
        json.dumps(updated_snapshot, indent=2) + "\n", encoding="utf-8"
    )

    # Preserve all annual observations in the historical audit, including
    # firms deliberately excluded from the final screening universe.
    if history_path is not None:
        shutil.copyfile(history_path, output_dir / "financial_history.csv")
    if issues_path is not None:
        issues = pd.read_csv(issues_path)
        updated_issues = pd.concat([
            issues,
            exclusions[["ticker", "reason"]].rename(columns={"reason": "issue"}),
        ], ignore_index=True)
        updated_issues.to_csv(output_dir / "collection_issues.csv", index=False)
    return updated_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--history", type=Path)
    parser.add_argument("--issues", type=Path)
    args = parser.parse_args()
    result = prepare_snapshot(
        args.universe, args.snapshot, args.output_dir, args.history, args.issues
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
