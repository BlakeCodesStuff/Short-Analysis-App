"""Small independent calculations for the class-specific ratio definitions."""

import math
import sys
from pathlib import Path
from statistics import NormalDist
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import calculate_metrics  # noqa: E402


def example(year: int, ticker="TEST") -> dict:
    current = year == 2025
    return {
        "ticker": ticker, "sector": "Industrials", "fiscal_period_end": f"{year}-12-31",
        "company_name": "Synthetic test company",
        "total_revenue": 110 if current else 100,
        "cost_of_revenue": 64 if current else 60,
        "ebit": 22 if current else 20,
        "net_income": 18 if current else 15,
        "operating_cash_flow": 11 if current else 10,
        "total_assets": 220 if current else 200,
        "current_assets": 90 if current else 80,
        "current_liabilities": 50 if current else 40,
        "cash_and_cash_equivalents": 12 if current else 10,
        "net_property_plant_equipment": 55 if current else 50,
        "common_stock_equity": 140 if current else 120,
        "stockholders_equity": 144 if current else 123,
        "total_equity_gross_minority_interest": 150 if current else 128,
        "total_liabilities_net_minority_interest": 70 if current else 72,
        "ordinary_shares_number": 10,
        "fiscal_period_end_close": 20,
        "quote_currency": "USD", "financial_currency": "USD",
        "total_debt": 32 if current else 30,
        "preferred_stock": 4 if current else 3,
        "minority_interest": 6 if current else 5,
        "current_debt_and_capital_lease_obligation": 11 if current else 10,
        "long_term_debt_and_capital_lease_obligation": 21 if current else 20,
        "accounts_receivable": 20 if current else 15,
        "depreciation_and_amortization": 6 if current else 5,
        "selling_general_and_administration": 12 if current else 10,
        "current_debt": 11 if current else 10,
        "long_term_debt": 21 if current else 20,
        "taxes_payable": 3 if current else 2,
    }


class RatioTests(unittest.TestCase):
    def test_course_ratios_and_beneish_balance_sheet_tata(self):
        current = calculate_metrics(pd.DataFrame([example(2024), example(2025)])).iloc[0]
        self.assertAlmostEqual(current["gpa"], (110 - 64) / 220)
        self.assertAlmostEqual(current["roc"], 22 / (55 + 90 - 50))
        self.assertAlmostEqual(current["bm"], 140 / (10 * 20))
        self.assertAlmostEqual(current["excess_cash_course"], 12 + 90 - 50)
        self.assertAlmostEqual(current["tev_course"], 200 + 32 + 4 + 6 - 52)
        self.assertAlmostEqual(current["eey"], 22 / 190)
        self.assertAlmostEqual(current["sta"], (18 - 11) / 220)
        self.assertAlmostEqual(current["snoa"], ((220 - 12) - (220 - 11 - 21 - 6 - 4 - 140)) / 220)
        self.assertAlmostEqual(current["tata"], -6 / 220)
        self.assertNotAlmostEqual(current["sta"], current["tata"])
        expected_probm = (
            -4.84 + .920 * current["dsri"] + .528 * current["gmi"]
            + .404 * current["aqi"] + .892 * current["sgi"]
            + .115 * current["depi"] - .172 * current["sgai"]
            + 4.679 * current["tata"] - .327 * current["lvgi"]
        )
        self.assertAlmostEqual(current["pman"], NormalDist().cdf(expected_probm))

    def test_missing_preferred_is_reconstructed_from_equity_identity(self):
        older, latest = example(2024), example(2025)
        latest["preferred_stock"] = None
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertEqual(result["preferred_stock_used"], 4)
        self.assertIn("minus Common", result["preferred_stock_source"])
        self.assertTrue(math.isfinite(result["eey"]))
        self.assertTrue(math.isfinite(result["snoa"]))

    def test_missing_preferred_without_equity_proof_stays_missing(self):
        older, latest = example(2024), example(2025)
        latest["preferred_stock"] = None
        latest["stockholders_equity"] = None
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertTrue(math.isnan(result["eey"]))
        self.assertTrue(math.isnan(result["snoa"]))
        self.assertTrue(math.isnan(result["preferred_stock_used"]))

    def test_minority_and_current_lease_debt_come_from_tied_balances(self):
        older, latest = example(2024), example(2025)
        latest["minority_interest"] = None
        latest["current_debt_and_capital_lease_obligation"] = None
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertEqual(result["minority_interest_used"], 6)
        self.assertEqual(result["current_debt_and_capital_lease_obligation_used"], 11)
        self.assertTrue(math.isfinite(result["eey"]))
        self.assertTrue(math.isfinite(result["snoa"]))

    def test_conflicting_minority_identities_cannot_be_used(self):
        older, latest = example(2024), example(2025)
        latest["minority_interest"] = None
        latest["total_liabilities_net_minority_interest"] = 80  # implies -4, not 6
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertTrue(math.isnan(result["minority_interest_used"]))
        self.assertTrue(math.isnan(result["eey"]))

    def test_currency_mismatch_and_multi_class_block_valuation_only(self):
        older, latest = example(2024), example(2025)
        latest["financial_currency"] = "TWD"
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertTrue(math.isnan(result["bm"]))
        self.assertTrue(math.isnan(result["eey"]))
        self.assertIn("currency", result["bm_reason"])
        self.assertTrue(math.isfinite(result["gpa"]))
        latest["financial_currency"] = "USD"
        latest["multiple_traded_share_classes"] = True
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertTrue(math.isnan(result["bm"]))
        self.assertIn("Multiple share classes", result["bm_reason"])

    def test_missing_prior_year_blocks_pman(self):
        latest = example(2025)
        older = example(2023)
        latest["ticker"] = older["ticker"] = "GAP"
        result = calculate_metrics(pd.DataFrame([latest, older])).iloc[0]
        self.assertTrue(math.isnan(result["pman"]))
        self.assertIn("Missing consecutive annual", result["pman_reason"])

    def test_total_tax_payable_fallback_unlocks_tata_without_inventing_zero(self):
        older, latest = example(2024), example(2025)
        older["total_tax_payable"] = older.pop("taxes_payable")
        latest["total_tax_payable"] = latest.pop("taxes_payable")
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertEqual(result["taxes_payable_used"], 3)
        self.assertIn("Total Tax Payable", result["taxes_payable_source"])
        self.assertAlmostEqual(result["tata"], -6 / 220)
        self.assertTrue(math.isfinite(result["pman"]))

    def test_direct_taxes_payable_takes_priority_over_aggregate(self):
        older, latest = example(2024), example(2025)
        latest["total_tax_payable"] = 80  # a broader number, not the desired direct 3
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertEqual(result["taxes_payable_used"], 3)
        self.assertIn("direct", result["taxes_payable_source"])

    def test_current_debt_reconstructed_only_after_subtracting_lease(self):
        older, latest = example(2024), example(2025)
        for row, current_lease in ((older, 3), (latest, 4)):
            row["current_debt"] = None
            row["current_capital_lease_obligation"] = current_lease
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertEqual(result["current_debt_used"], 7)
        self.assertIn("minus Current Capital Lease", result["current_debt_source"])
        self.assertTrue(math.isfinite(result["pman"]))

    def test_current_lease_may_be_reconstructed_from_total_and_long_term(self):
        older, latest = example(2024), example(2025)
        for row, total_lease, long_lease in ((older, 5, 2), (latest, 6, 2)):
            row["current_debt"] = None
            row["capital_lease_obligations"] = total_lease
            row["long_term_capital_lease_obligation"] = long_lease
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertEqual(result["current_capital_lease_obligation_used"], 4)
        self.assertEqual(result["current_debt_used"], 7)
        self.assertTrue(math.isfinite(result["pman"]))

    def test_current_debt_missing_without_separate_lease_must_stay_missing(self):
        older, latest = example(2024), example(2025)
        for row in (older, latest):
            row["current_debt"] = None
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertTrue(math.isnan(result["current_debt_used"]))
        self.assertTrue(math.isnan(result["pman"]))
        self.assertIn("current_debt_used", result["pman_reason"])

    def test_snoa_debt_residual_requires_independent_matching_year(self):
        older, latest = example(2024), example(2025)
        older["current_debt_and_capital_lease_obligation"] = None
        latest["current_debt_and_capital_lease_obligation"] = None
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertTrue(math.isnan(result["current_debt_and_capital_lease_obligation_used"]))
        self.assertTrue(math.isnan(result["snoa"]))

    def test_snoa_debt_residual_blocked_by_conflicting_reported_year(self):
        older, latest = example(2024), example(2025)
        older["total_debt"] = 35  # reported current=10, long-term=20, mismatch 5
        latest["current_debt_and_capital_lease_obligation"] = None
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertTrue(math.isnan(result["current_debt_and_capital_lease_obligation_used"]))
        self.assertTrue(math.isnan(result["snoa"]))

    def test_direct_zero_current_debt_and_lease_justifies_zero_current_debt(self):
        older, latest = example(2024), example(2025)
        for row in (older, latest):
            row["current_debt"] = None
            row["current_debt_and_capital_lease_obligation"] = 0
        result = calculate_metrics(pd.DataFrame([older, latest])).iloc[0]
        self.assertEqual(result["current_debt_used"], 0)
        self.assertIn("directly reported", result["current_debt_source"])
        self.assertTrue(math.isfinite(result["pman"]))

    def test_duplicate_company_period_rejected(self):
        row = example(2025)
        with self.assertRaisesRegex(ValueError, "Duplicate ticker"):
            calculate_metrics(pd.DataFrame([row, row]))


if __name__ == "__main__":
    unittest.main()
