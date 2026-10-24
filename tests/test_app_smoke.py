"""Exercise the whole app flow without needing a Streamlit server in CI."""

from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.sidebar = self
        self.reported_metrics = []
        self.reported_tables = []
        self.reported_charts = []
        self.reported_chart_configs = []
        self.reported_captions = []
        self.reported_warnings = []
        self.reported_markdown = []
        self.reported_tab_labels = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cache_data(self, **_):
        return lambda fn: fn

    def __getattr__(self, name):
        if name in {
            "set_page_config", "title", "info", "header", "subheader",
            "error", "download_button",
            "link_button",
        }:
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def markdown(self, text, **kwargs):
        self.reported_markdown.append((text, kwargs))

    def tabs(self, labels, **_):
        self.reported_tab_labels = list(labels)
        return [self] * len(labels)

    def multiselect(self, label, options, default=None, **_):
        return list(default) if default is not None else list(options)

    def toggle(self, label, value=False, **_):
        return value

    def slider(self, label, minimum, maximum, value, step, **_):
        return value

    def select_slider(self, label, options, value, **_):
        self.zoom_levels = list(options)
        return value

    def columns(self, count):
        return [self] * count

    def metric(self, label, value, **_):
        self.reported_metrics.append((label, value))

    def dataframe(self, table, **_):
        self.reported_tables.append(table)

    def plotly_chart(self, figure, **kwargs):
        self.reported_charts.append(figure)
        self.reported_chart_configs.append(kwargs.get("config", {}))

    def caption(self, message, **_):
        self.reported_captions.append(message)

    def warning(self, message, **_):
        self.reported_warnings.append(message)

    def selectbox(self, label, options, **_):
        return list(options)[0]

    def expander(self, label, **_):
        return self

    def stop(self):
        raise AssertionError("The sample dataset unexpectedly prevented app startup")


class AppSmokeTests(unittest.TestCase):
    def test_all_interface_sections_execute_on_real_snapshot(self):
        fake = FakeStreamlit()
        with patch.dict(sys.modules, {"streamlit": fake}):
            namespace = runpy.run_path(str(PROJECT / "app.py"))
        final = namespace["p"]
        self.assertEqual(len(final), 146)
        self.assertEqual(len(namespace["df"]), 949)
        self.assertTrue(final["short_warning_score"].between(0, 100).all())
        self.assertEqual(len(namespace["magic"]), 723)
        self.assertGreater(len(fake.reported_metrics), 4)
        self.assertTrue(any(label == "Valuation concern" for label, _ in fake.reported_metrics))
        self.assertTrue(any(label == "Earnings-quality concern" for label, _ in fake.reported_metrics))
        self.assertTrue(any(label == "Profitability concern" for label, _ in fake.reported_metrics))
        self.assertEqual(fake.reported_metrics[:4], [
            ("Starting companies", "949"),
            ("After valuation screen", "383"),
            ("After earnings-quality screen", "227"),
            ("After profitability screen", "146"),
        ])
        displayed_short = [
            t for t in fake.reported_tables
            if "Relative SHORT concern ranking (/100)" in t.columns
        ]
        self.assertEqual(len(displayed_short), 1)
        self.assertEqual(len(displayed_short[0]), 10)
        self.assertEqual(
            displayed_short[0]["Ticker"].tolist(), final.head(10)["ticker"].tolist()
        )
        self.assertEqual(fake.reported_tab_labels, [
            "FTAI Aviation · #3", "Coherent Corp. · #8", "Rocket Lab · #2"
        ])
        case_html = [
            content for content, kwargs in fake.reported_markdown
            if kwargs.get("unsafe_allow_html") and '<article class="investment-case">' in content
        ]
        self.assertEqual(len(case_html), 3)
        self.assertTrue(all("Supporting Evidence" in card for card in case_html))
        self.assertTrue(any("$3.45 billion in debt" in card for card in case_html))
        self.assertTrue(any("I chose three companies" in content
                            for content, _ in fake.reported_markdown))
        for case in namespace["CASES"]:
            self.assertEqual(final["ticker"].tolist().index(case.ticker) + 1,
                             case.default_rank)
            score = final.loc[final["ticker"].eq(case.ticker), "short_warning_score"].iloc[0]
            self.assertAlmostEqual(score, case.default_score)
        displayed_magic = [
            t for t in fake.reported_tables if "Combined rank sum" in t.columns
        ]
        self.assertEqual(len(displayed_magic), 1)
        self.assertEqual(len(displayed_magic[0]), 10)
        self.assertIn("ROC capital / assets (%)", displayed_magic[0].columns)
        self.assertIn("ROC interpretation", displayed_magic[0].columns)
        self.assertEqual(
            displayed_magic[0]["Ticker"].tolist(),
            namespace["magic"].head(10)["ticker"].tolist(),
        )
        flagged = displayed_magic[0].loc[
            displayed_magic[0]["ROC interpretation"].ne(""), "Ticker"
        ].tolist()
        self.assertEqual(set(flagged), {"OTEX", "DELL", "CPAY"})
        self.assertTrue(any(all(ticker in w for ticker in flagged)
                            and "2%" in w for w in fake.reported_warnings))
        self.assertEqual(len(fake.reported_charts), 3)
        chart = fake.reported_charts[0]
        self.assertEqual(sum(len(trace.x) for trace in chart.data), len(final))
        self.assertEqual([trace.name for trace in chart.data], [
            "Other surviving companies", "Top 10 SHORT candidates"
        ])
        self.assertEqual(len(chart.data[1].x), 10)
        self.assertEqual(chart.data[1].marker.symbol, "diamond")
        self.assertTrue(all(0 <= n <= 100 for trace in chart.data for n in trace.x))
        self.assertTrue(all(0 <= n <= 100 for trace in chart.data for n in trace.y))
        self.assertTrue(any("GE Aerospace" in text for text in chart.data[1].text))
        self.assertTrue(any("BM:" in text for text in chart.data[1].text))
        self.assertTrue(any("STA:" in text for text in chart.data[1].text))
        self.assertEqual(fake.zoom_levels, [1.0, 1.5, 2.0, 3.0, 4.0])
        self.assertTrue(fake.reported_chart_configs[0]["scrollZoom"])
        self.assertGreaterEqual(chart.layout.margin.b, 70)
        self.assertTrue(chart.layout.xaxis.automargin)
        self.assertEqual(list(chart.layout.xaxis.range), [0, 101])
        finer_zoom = namespace["short_candidate_scatter"](final, ["bm", "sta", "gpa"], 1.5)
        self.assertAlmostEqual(finer_zoom.layout.xaxis.range[0], 100 - 100 / 1.5)
        self.assertEqual(finer_zoom.layout.yaxis.range[1], 101)
        revenue_chart, cash_chart = fake.reported_charts[1:]
        self.assertEqual(revenue_chart.data[0].name, "Revenue")
        self.assertEqual([trace.name for trace in cash_chart.data], [
            "Net income", "Operating cash flow"
        ])
        self.assertGreater(len(revenue_chart.data[0].x), 1)
        self.assertEqual(
            list(revenue_chart.data[0].x), sorted(revenue_chart.data[0].x)
        )
        peer_tables = [t for t in fake.reported_tables if "Relative to peers" in t.columns]
        self.assertEqual(len(peer_tables), 1)
        self.assertEqual(len(peer_tables[0]), 7)
        self.assertIn("Difference vs median", peer_tables[0].columns)
        self.assertIn("Peer observations", peer_tables[0].columns)
        self.assertTrue(any(
            "Analysis reference date: October 17, 2026" in c
            and "Data collected: October 23, 2026 (UTC)" in c
            and "Fiscal year-ends: 2025-07-31 to 2026-07-31" in c
            for c in fake.reported_captions
        ))

    def test_history_keeps_dates_and_missing_values_without_mix_of_currencies(self):
        fake = FakeStreamlit()
        with patch.dict(sys.modules, {"streamlit": fake}):
            namespace = runpy.run_path(str(PROJECT / "app.py"))
        history = pd.DataFrame({
            "ticker": ["TEST", "TEST", "TEST", "TEST", "OTHER"],
            "financial_currency": ["USD", "USD", "USD", "CAD", "USD"],
            "fiscal_period_end": pd.to_datetime([
                "2025-12-31", "2023-12-31", "2024-12-31", "2022-12-31", "2024-12-31"
            ]),
            "fiscal_year": [2025, 2023, 2024, 2022, 2024],
            "total_revenue": [3.0, 1.0, float("nan"), 999.0, 88.0],
            "net_income": [0.4, 0.2, 0.3, 999.0, 1.0],
            "operating_cash_flow": [0.5, 0.1, 0.35, 999.0, 1.0],
        })
        series = namespace["company_financial_history"](history, "TEST", "USD")
        self.assertEqual(series["fiscal_year"].tolist(), [2023, 2024, 2025])
        self.assertTrue(pd.isna(series["total_revenue"].iloc[1]))
        revenue = namespace["annual_revenue_chart"](series, "USD")
        self.assertEqual(list(revenue.data[0].y), [1.0, 3.0])
        cash = namespace["annual_earnings_cash_chart"](series, "USD")
        self.assertEqual(list(cash.data[0].x), ["FY 2023", "FY 2024", "FY 2025"])

    def test_peer_assessment_direction_and_missing_data(self):
        fake = FakeStreamlit()
        with patch.dict(sys.modules, {"streamlit": fake}):
            namespace = runpy.run_path(str(PROJECT / "app.py"))
        assess = namespace["relative_peer_assessment"]
        self.assertEqual(assess("bm", 0.3, 0.6), "🔴 Higher SHORT concern")
        self.assertEqual(assess("roc", 0.2, 0.1), "🟢 Lower SHORT concern")
        self.assertEqual(assess("sta", 0.2, 0.1), "🔴 Higher SHORT concern")
        self.assertEqual(assess("sta", 0.1, 0.1), "⚪ Near peer median")
        self.assertEqual(assess("sta", float("nan"), 0.1), "⚪ Data unavailable")
        self.assertEqual(namespace["peer_gap_display"]("sta", 0.3, 0.2), "+10.00 pp")


if __name__ == "__main__":
    unittest.main()
