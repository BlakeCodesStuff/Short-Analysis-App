"""Keep the independently authored investment cases stable and easy to audit."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from investment_cases import CASES, CASE_INTRO, case_html


class InvestmentCaseTests(unittest.TestCase):
    def test_selected_companies_and_intro(self):
        self.assertEqual([case.ticker for case in CASES], ["FTAI", "COHR", "RKLB"])
        self.assertIn("not the top three ranked companies", CASE_INTRO)
        for case in CASES:
            self.assertEqual(len(case.metrics), 3)
            self.assertGreaterEqual(len(case.body), 3)
            self.assertGreaterEqual(len(case.evidence), 5)
            self.assertIn("sell/short", case.body[0])

    def test_fact_correction_and_no_source_lists(self):
        case = CASES[0]
        narrative = " ".join((*case.body, case.counterargument, case.conclusion))
        self.assertIn("$3.45 billion in debt", narrative)
        self.assertNotIn("$2.45 billion", narrative)
        for case in CASES:
            html = case_html(case)
            self.assertIn("Supporting Evidence", html)
            self.assertIn("Counterargument", html)
            self.assertIn("My conclusion", html)
            self.assertNotIn("Verified Supporting Evidence", html)
            self.assertNotIn("<a ", html)

    def test_user_supplied_text_is_html_escaped(self):
        changed = replace(CASES[0], counterargument="Risk <script>alert(1)</script>")
        html = case_html(changed)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>", html)


if __name__ == "__main__":
    unittest.main()
