"""Deterministic tests of filter effects and pre-screen data exclusions."""

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import identify_beneish_index_exclusions  # noqa: E402
from screening import concern_percentile, magic_formula_rankings, stage_screen  # noqa: E402


def toy_universe():
    return pd.DataFrame({
        "ticker": list("ABCDEFGH"),
        "sector": ["Tech"] * 4 + ["Retail"] * 4,
        "bm": [0.10, 0.20, 0.30, 0.40, 0.5, 0.6, 0.7, 0.8],
        "eey": [0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        "sta": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16],
        "gpa": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    })


class ScreenTests(unittest.TestCase):
    def test_percentages_and_metric_selection_change_survivors(self):
        data = toy_universe()
        all_firms, _, _ = stage_screen(data, ["bm"], 100, False, "v")
        shortlisted, _, audit = stage_screen(data, ["bm"], 25, False, "v")
        other_metric, _, _ = stage_screen(data, ["eey"], 25, False, "v")
        self.assertEqual(len(all_firms), 8)
        self.assertEqual(shortlisted["ticker"].tolist(), ["A", "B"])
        self.assertEqual(other_metric["ticker"].tolist(), ["H", "G"])
        self.assertEqual(audit["excluded_by_cutoff"], 6)

    def test_missing_values_have_separate_audit_not_zero(self):
        data = toy_universe()
        data.loc[0, "sta"] = np.nan
        kept, _, audit = stage_screen(data, ["sta"], 50, False, "q")
        self.assertEqual(len(kept), 3)
        self.assertEqual(audit["missing_metric"], 1)
        self.assertNotIn("A", kept["ticker"].tolist())
        self.assertEqual(
            audit["entering"], audit["missing_metric"]
            + audit["insufficient_sector"] + audit["excluded_by_cutoff"]
            + audit["retained"]
        )

    def test_sector_peer_rule_excludes_undersized_groups(self):
        data = toy_universe()
        data.loc[data["ticker"].eq("D"), "bm"] = np.nan
        data.loc[data["ticker"].eq("C"), "bm"] = np.nan
        kept, _, audit = stage_screen(data, ["bm"], 100, True, "v")
        self.assertEqual(audit["missing_metric"], 2)
        self.assertEqual(audit["insufficient_sector"], 2)
        self.assertEqual(len(kept), 4)

    def test_equal_scores_do_not_receive_arbitrary_strength(self):
        data = toy_universe()
        data["bm"] = 0.2
        kept, _, _ = stage_screen(data, ["bm"], 100, False, "v")
        self.assertEqual(kept["v_strength"].nunique(), 1)

    def test_relative_concern_percentiles_reverse_for_profitability(self):
        values = pd.Series([0.1, 0.2, 0.3])
        self.assertGreater(concern_percentile(values, "bm", 0.1), 80)
        self.assertGreater(concern_percentile(values, "sta", 0.3), 80)
        self.assertEqual(concern_percentile(values, "bm", 0.2), 50)

    def test_magic_formula_sums_two_independent_high_is_best_rankings(self):
        data = pd.DataFrame({
            "ticker": ["A", "B", "C", "D", "X"],
            "roc": [0.50, 0.40, 0.30, 0.20, np.nan],
            "eey": [0.02, 0.05, 0.04, 0.03, 0.09],
        })
        ranked = magic_formula_rankings(data)
        self.assertEqual(ranked["ticker"].tolist(), ["B", "A", "C", "D"])
        self.assertEqual(ranked["magic_rank_sum"].tolist(), [3, 5, 5, 7])
        self.assertEqual(ranked["magic_overall_rank"].tolist(), [1, 2, 2, 4])
        self.assertEqual(int(ranked.iloc[0]["magic_roc_rank"]), 2)
        self.assertEqual(int(ranked.iloc[0]["magic_eey_rank"]), 1)

    def test_magic_formula_handles_ties_and_nonfinite_values(self):
        data = pd.DataFrame({
            "ticker": ["B", "A", "INVALID", "MISSING"],
            "roc": [0.5, 0.5, float("inf"), 0.2],
            "eey": [0.1, 0.1, 0.8, np.nan],
        })
        ranked = magic_formula_rankings(data)
        self.assertEqual(ranked["ticker"].tolist(), ["A", "B"])
        self.assertEqual(ranked["magic_roc_rank"].tolist(), [1, 1])
        self.assertEqual(ranked["magic_eey_rank"].tolist(), [1, 1])
        self.assertEqual(ranked["magic_rank_sum"].tolist(), [2, 2])
        self.assertTrue(magic_formula_rankings(data.iloc[2:]).empty)

    def test_identify_exclusions_only_when_nonpositive_index_is_reported(self):
        data = pd.DataFrame({
            "ticker": ["KEEP", "NEG_GMI", "NEG_SGAI", "UNKNOWN"],
            "company_name": ["Keep", "GMI", "SGAI", "Unknown"],
            "gmi": [1.0, -0.1, 1.0, np.nan],
            "sgai": [0.8, 0.9, 0.0, np.nan],
        })
        excluded = identify_beneish_index_exclusions(data)
        self.assertEqual(set(excluded["ticker"]), {"NEG_GMI", "NEG_SGAI"})

    def test_deployed_static_snapshot_is_clean(self):
        dataset_path = Path(__file__).resolve().parents[1] / "data" / "short_universe.csv"
        if not dataset_path.exists():
            self.skipTest("No static snapshot has been prepared")
        data = pd.read_csv(dataset_path)
        self.assertEqual(len(data), 949)
        self.assertFalse(data["ticker"].duplicated().any())
        self.assertTrue(identify_beneish_index_exclusions(data).empty)
        self.assertEqual(data["pman"].notna().sum(), 457)


if __name__ == "__main__":
    unittest.main()
