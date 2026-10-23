"""ACCT 3343 SHORT Analysis: static-data, sequential interactive screening."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st

from screening import LOW_IS_CONCERNING, concern_percentile, magic_formula_rankings, stage_screen


st.set_page_config(page_title="ACCT 3343 | SHORT Screening", page_icon="📉", layout="wide")
DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "short_universe.csv"
META_PATH = DATA_DIR / "snapshot.json"

LABELS = {
    "bm": "Book-to-Market (low = expensive relative to book)",
    "eey": "Enterprise Earnings Yield (low = weak earnings relative to TEV)",
    "sta": "Simplified Total Accruals (high = earnings/cash gap)",
    "snoa": "Scaled Net Operating Assets (high = accrual buildup)",
    "pman": "Beneish PMAN (high = greater model warning)",
    "gpa": "Gross Profitability / Assets (low = weak gross profitability)",
    "roc": "Return on Capital (low = weak operating return)",
}
PERCENT_METRICS = {"eey", "sta", "snoa", "pman", "gpa", "roc"}

SHORT_MEANING = {
    "bm": "Less book equity relative to the market price; this can reflect a rich valuation or unrecorded intangible value.",
    "eey": "Lower EBIT relative to enterprise value; the price may be high relative to operating earnings.",
    "sta": "Earnings run ahead of operating cash flow, making reported profit less cash-supported.",
    "snoa": "A larger share of assets consists of net operating assets, warranting closer accrual review.",
    "pman": "A higher Beneish model warning suggests examining revenue recognition and other accounting estimates.",
    "gpa": "Lower gross profit per dollar of assets can make a high valuation harder to support.",
    "roc": "A lower operating return on invested capital can make an optimistic market valuation vulnerable.",
}


@st.cache_data(show_spinner=False)
def read_dataset(path: str, modified_ns: int):
    frame = pd.read_csv(path)
    if frame["ticker"].duplicated().any():
        raise ValueError("Snapshot contains duplicate ticker rows")
    for name in LABELS:
        if name not in frame.columns:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if "sector" not in frame.columns:
        frame["sector"] = np.nan
    return frame


def shown_columns(frame: pd.DataFrame, include_risk=False) -> pd.DataFrame:
    cols = ["ticker", "company_name", "sector", "fiscal_period_end"]
    if include_risk:
        cols += ["short_warning_score"]
    cols += ["bm", "eey", "sta", "snoa", "pman", "gpa", "roc"]
    cols = [c for c in cols if c in frame]
    formatted = frame[cols].copy()
    for metric in PERCENT_METRICS.intersection(formatted.columns):
        formatted[metric] = (100 * formatted[metric]).round(2)
    if "bm" in formatted:
        formatted["bm"] = formatted["bm"].round(3)
    return formatted.rename(columns={
        "ticker": "Ticker", "company_name": "Company", "sector": "Sector",
        "fiscal_period_end": "Fiscal year-end",
        "short_warning_score": "Relative SHORT concern ranking (/100)",
        "bm": "B/M", "eey": "EEY (%)", "sta": "STA (%)", "snoa": "SNOA (%)",
        "pman": "PMAN (%)", "gpa": "GPA (%)", "roc": "ROC (%)",
    })


def metric_display(name: str, value: float) -> str:
    if pd.isna(value):
        return "Unavailable"
    return f"{value:.3f}" if name == "bm" else f"{value * 100:.2f}%"


def shown_magic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Display original metrics alongside transparent textbook rank additions."""
    cols = [
        "magic_overall_rank", "ticker", "company_name", "sector",
        "roc", "magic_roc_rank", "eey", "magic_eey_rank", "magic_rank_sum",
    ]
    shown = frame[cols].copy()
    shown["roc"] = (100 * shown["roc"]).round(2)
    shown["eey"] = (100 * shown["eey"]).round(2)
    return shown.rename(columns={
        "magic_overall_rank": "Overall rank", "ticker": "Ticker",
        "company_name": "Company", "sector": "Sector", "roc": "ROC (%)",
        "magic_roc_rank": "ROC rank", "eey": "EEY (%)",
        "magic_eey_rank": "EEY rank", "magic_rank_sum": "Combined rank sum",
    })


def peer_statistics(universe: pd.DataFrame, sector: str, metric: str, value: float) -> dict:
    peers = universe.loc[universe["sector"].eq(sector), metric].dropna()
    description = f"{sector} sector"
    if len(peers) < 5:
        peers = universe[metric].dropna()
        description = "screening universe"
    return {
        "median": peers.median() if len(peers) else float("nan"),
        "percentile": concern_percentile(peers, metric, float(value)) if pd.notna(value) else float("nan"),
        "sample": len(peers),
        "group": description,
    }


st.title("SHORT Analysis | Financial Statement Screening")
st.caption("ACCT 3343 · Screening for potential overvaluation and earnings-quality concerns")
st.info(
    "This is an educational, cross-sectional research screen. Its **Custom SHORT warning score** "
    "is an app-created relative ranking, not an established textbook or predictive financial "
    "model. A high score is not proof of overvaluation, earnings manipulation, or a profitable short trade."
)

if not CSV_PATH.exists():
    st.warning(
        "The static Yahoo Finance dataset hasn't been added yet. Run `build_universe.ipynb` "
        "in Colab, then place its `short_universe.csv` inside the project's `data/` folder. "
        "No Yahoo request is made when this app starts."
    )
    st.stop()

try:
    df = read_dataset(str(CSV_PATH), CSV_PATH.stat().st_mtime_ns)
except Exception as exc:
    st.error(f"Cannot read static dataset: {exc}")
    st.stop()
if df.empty:
    st.error("The static dataset is empty.")
    st.stop()

snapshot = {}
if META_PATH.exists():
    try:
        snapshot = json.loads(META_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        pass
st.caption(
    f"Source: Yahoo Finance · Data collected: {snapshot.get('collected_at_utc', 'unknown date')} "
    f"· Starting firms: {len(df):,} · "
    f"Beneish index quality exclusions: {snapshot.get('beneish_index_exclusions', 0)} "
    "· Filters use only the saved dataset"
)

st.sidebar.header("Screen controls")
sector_choices = sorted(df["sector"].dropna().astype(str).unique().tolist())
chosen_sectors = st.sidebar.multiselect("Sectors to include", sector_choices, default=sector_choices)
df = df.loc[df["sector"].isin(chosen_sectors)].copy()
screening_universe = df.copy()
sector_relative = st.sidebar.toggle(
    "Sector-relative rankings", value=False,
    help="Subtract the mean of peers in the current screen; at least three valid sector peers are required.",
)

st.sidebar.subheader("1 · Expensive valuation")
valuation_options = ["bm", "eey"]
valuation_metrics = st.sidebar.multiselect(
    "Valuation signals", valuation_options, default=["bm"], format_func=lambda m: LABELS[m]
)
valuation_pct = st.sidebar.slider("Retain most concerning valuation (%)", 5, 100, 50, 5)

st.sidebar.subheader("2 · Earnings quality")
quality_metrics = st.sidebar.multiselect(
    "Accounting-risk signals", ["sta", "snoa", "pman"], default=["sta"],
    format_func=lambda m: LABELS[m],
    help="PMAN needs all eight Beneish inputs and a valid preceding annual period; coverage may be low.",
)
quality_pct = st.sidebar.slider("Retain highest accrual risk (%)", 5, 100, 60, 5)

st.sidebar.subheader("3 · Weak operating quality")
performance_metrics = st.sidebar.multiselect(
    "Profitability signals", ["gpa", "roc"], default=["gpa"], format_func=lambda m: LABELS[m]
)
performance_pct = st.sidebar.slider("Retain weakest profitability (%)", 5, 100, 65, 5)

if not (valuation_metrics and quality_metrics and performance_metrics):
    st.warning("Select at least one financial metric for each of the three screens.")
    st.stop()

v, rejected_v, audit_v = stage_screen(df, valuation_metrics, valuation_pct, sector_relative, "valuation")
q, rejected_q, audit_q = stage_screen(v, quality_metrics, quality_pct, sector_relative, "quality")
p, rejected_p, audit_p = stage_screen(q, performance_metrics, performance_pct, sector_relative, "performance")

audits = [audit_v, audit_q, audit_p]
labels = [
    "Starting companies",
    "After valuation screen",
    "After earnings-quality screen",
    "After profitability screen",
]
counts = [len(df), len(v), len(q), len(p)]
summary_cols = st.columns(4)
for col, label, count in zip(summary_cols, labels, counts):
    col.metric(label, f"{count:,}")
st.caption(
    "Each stage retains firms with stronger relative SHORT warning signals; "
    "missing required data is excluded."
)

st.subheader("Live screening funnel")
audit_df = pd.DataFrame(audits).rename(columns={
    "stage": "Stage", "entering": "Entering", "missing_metric": "Missing required metric",
    "insufficient_sector": "Insufficient sector peers", "excluded_by_cutoff": "Cutoff removals",
    "retained": "Retained", "metrics": "Selected metrics",
})
audit_df["Selected metrics"] = audit_df["Selected metrics"].map(lambda m: ", ".join(m).upper())
st.dataframe(audit_df, hide_index=True, use_container_width=True)

st.caption(
    "Stages are applied in order. 'Missing required metric' and 'Insufficient sector peers' "
    "are independent of the slider cutoffs; no unavailable values are treated as zero."
)

if p.empty:
    st.warning(
        "No firms survived these settings. Check the missing-data counts and try a metric "
        "with higher coverage or a larger retention percentage. No missing inputs were imputed."
    )
else:
    p["short_warning_score"] = (
        100 * p[["valuation_strength", "quality_strength", "performance_strength"]].mean(axis=1)
    ).round(1)
    p = p.sort_values(["short_warning_score", "ticker"], ascending=[False, True])
    st.subheader("Potential SHORT research candidates · Top 10")
    st.caption(
        "**Custom SHORT warning score:** the relative SHORT concern ranking (0–100) "
        "combines the three user-selected screens (higher = more relative warning signals). "
        "This is our own educational ranking, not a textbook formula or an estimated probability of decline. "
        f"Showing {min(10, len(p))} of {len(p)} final-stage survivors. "
        "Financial ratios below are percentages except book-to-market (a multiple)."
    )
    st.dataframe(
        shown_columns(p.head(10), include_risk=True),
        hide_index=True, use_container_width=True, height=395,
    )
    st.download_button(
        "Download surviving companies (CSV)",
        p.rename(columns={
            "short_warning_score": "relative_short_concern_ranking"
        }).to_csv(index=False).encode("utf-8"),
        file_name="short_candidates.csv", mime="text/csv",
    )

    with st.expander("How is the Custom SHORT warning score calculated?"):
        st.markdown(
            "**This ranking was created for this app; it is not an established financial model.** "
            "It compares companies within the selected universe and changes when the "
            "screening settings change.\n\n"
            "1. **Rank selected metrics within each stage.** Low BM and EEY, high STA/SNOA/PMAN, "
            "and low GPA/ROC count as more concerning. When multiple metrics are selected "
            "in one stage, their ranks are added with equal weight.\n"
            "2. **Turn each stage into a relative 0–100 concern ranking.** The most concerning "
            "rank sum receives the highest score among companies with the required data "
            "at that stage. Ties share an average ranking. The eligible comparison group "
            "gets smaller after each cutoff.\n"
            "3. **Average the three stage scores equally:** "
            "`(valuation + earnings quality + profitability) / 3`. "
            "Each stage receives one-third of the final Custom SHORT warning score, "
            "regardless of how many metrics are selected in that stage.\n\n"
            "**Interpretation:** 85/100 means relatively strong warning signals within "
            "the chosen screening setup. It does not mean 85% overvaluation, an 85% "
            "chance of decline, or an 85% probability of a profitable SHORT."
        )

    st.subheader("Candidate analysis")
    st.caption("The figures below are relative SHORT concern rankings out of 100, not predicted returns.")
    top_cols = st.columns(min(3, len(p)))
    for col, (_, company) in zip(top_cols, p.head(3).iterrows()):
        with col:
            st.metric(str(company["ticker"]), f"{company['short_warning_score']:.1f}/100")
            st.caption(str(company.get("company_name") or company["ticker"]))
    symbol = st.selectbox("Investigate a surviving ticker", p["ticker"].tolist())
    chosen = p.loc[p["ticker"].eq(symbol)].iloc[0]
    sector = str(chosen.get("sector", "Unknown sector"))
    st.markdown(f"### {chosen.get('company_name', symbol)} ({symbol})")
    st.caption(f"Sector: {sector} · Industry: {chosen.get('industry', 'Not classified')}")

    st.markdown("**Custom SHORT warning score breakdown**")
    breakdown = st.columns(3)
    for col, label, field in zip(
        breakdown,
        ["Valuation concern", "Earnings-quality concern", "Profitability concern"],
        ["valuation_strength", "quality_strength", "performance_strength"],
    ):
        with col:
            st.metric(label, f"{chosen[field] * 100:.1f}/100")
    st.caption(
        "These are relative rankings within each stage's eligible companies. "
        "Their equal-weight average gives the custom SHORT warning score above. "
        "They are not calibrated probabilities of overvaluation or a price decline."
    )

    stage_signals = [
        ("Valuation concern", valuation_metrics),
        ("Accounting-quality concern", quality_metrics),
        ("Operating concern", performance_metrics),
    ]
    st.markdown("**Provisional SHORT thesis**")
    for category, stage_metrics in stage_signals:
        # In a multi-metric stage, explain the strongest concern relative to
        # the original sector peers, even if it is not independently extreme.
        ranked = sorted(
            stage_metrics,
            key=lambda m: peer_statistics(screening_universe, sector, m, chosen[m])["percentile"],
            reverse=True,
        )
        strongest = ranked[0]
        stats = peer_statistics(screening_universe, sector, strongest, chosen[strongest])
        versus_median_is_concerning = (
            chosen[strongest] < stats["median"] if strongest in LOW_IS_CONCERNING
            else chosen[strongest] > stats["median"]
        )
        interpretation = (
            SHORT_MEANING[strongest] if versus_median_is_concerning
            else "This metric is not worse than its broader peer median, so it is not a standalone SHORT argument."
        )
        st.markdown(
            f"- **{category}:** {strongest.upper()} is "
            f"{metric_display(strongest, chosen[strongest])} versus "
            f"{metric_display(strongest, stats['median'])} median among "
            f"{stats['sample']} {stats['group']} observations. "
            f"{interpretation}"
        )

    tax_source = str(chosen.get("taxes_payable_source", ""))
    if "Total Tax Payable" in tax_source and "pman" in quality_metrics:
        st.warning(
            "PMAN for this company uses Yahoo's broader 'Total Tax Payable' "
            "line rather than a narrower income-tax-payable row. Check the "
            "accounting disclosure before drawing a conclusion."
        )

    st.info(
        "**Investment conclusion:** This is a candidate for further SHORT research, "
        "not an automatic sell order. Before forming a trade thesis, review the "
        "latest filing, whether the accounting flags are explained by the business "
        "model, valuation versus direct competitors, and potential catalysts and "
        "short-sale risks."
    )

    comparisons = []
    for name in LABELS:
        if pd.isna(chosen[name]):
            continue
        stats = peer_statistics(screening_universe, sector, name, chosen[name])
        comparisons.append({
            "Metric": name.upper(),
            "Company": metric_display(name, chosen[name]),
            "Peer median": metric_display(name, stats["median"]),
            "Relative concern percentile": round(stats["percentile"], 1),
            "Peer group": stats["group"],
            "Selected for screening": name in valuation_metrics + quality_metrics + performance_metrics,
        })
    st.markdown("**Financial evidence and peer comparison**")
    st.dataframe(pd.DataFrame(comparisons), hide_index=True, use_container_width=True)
    st.caption(
        "Concern percentile ranks from 0 to 100 within the initial sector group "
        "(or all selected sectors if too few peers); higher means more concerning "
        "for that metric. It is not a probability of loss."
    )
    st.caption(
        f"Fiscal period: {chosen.get('fiscal_period_end', 'Unknown')} · "
        f"Previous period: {chosen.get('prior_fiscal_period_end', 'Unknown')} · "
        "Price used: historical closing price on or before the fiscal period-end"
    )
    st.link_button("View the company's Yahoo Finance page", f"https://finance.yahoo.com/quote/{quote(symbol, safe='')}/")

with st.expander("Inspect companies around each cutoff"):
    for name, kept, rejected in [
        ("Valuation", v, rejected_v),
        ("Earnings quality", q, rejected_q),
        ("Operating performance", p, rejected_p),
    ]:
        st.markdown(f"**{name}**")
        left, right = st.columns(2)
        with left:
            st.caption("Last three excluded by percentage cutoff")
            st.dataframe(shown_columns(rejected.head(3)), hide_index=True, use_container_width=True)
        with right:
            st.caption("Last three retained")
            st.dataframe(shown_columns(kept.tail(3)), hide_index=True, use_container_width=True)

with st.expander("Data coverage and missing information"):
    coverage = pd.DataFrame([
        {"Metric": name.upper(), "Available companies": int(df[name].notna().sum()),
         "Missing companies": int(df[name].isna().sum()),
         "Coverage": f"{df[name].notna().mean():.1%}" if len(df) else "n/a"}
        for name in LABELS
    ])
    st.dataframe(coverage, hide_index=True, use_container_width=True)
    st.caption(
        "Missing values are not guessed. Some balance-sheet components are derived from "
        "reported accounting identities and their source is recorded. Each stage removes "
        "firms missing its selected metrics."
    )
    if not df.empty:
        inspect_ticker = st.selectbox(
            "Inspect one company's data quality", sorted(df["ticker"].tolist()),
            key="data_quality_ticker",
        )
        inspected = df.loc[df["ticker"].eq(inspect_ticker)].iloc[0]
        reason_rows = []
        for code, display in LABELS.items():
            is_available = pd.notna(inspected[code])
            reason_rows.append({
                "Metric": code.upper(),
                "Status": "Available" if is_available else "Unavailable",
                "Explanation": "" if is_available else str(inspected.get(f"{code}_reason", "Not supplied")),
            })
        st.dataframe(pd.DataFrame(reason_rows), hide_index=True, use_container_width=True)
        sources = {
            "Quote currency": inspected.get("quote_currency", "Not recorded"),
            "Statement currency": inspected.get("financial_currency", "Not recorded"),
            "Multiple traded share classes": inspected.get("multiple_traded_share_classes", "Not recorded"),
            "Preferred stock": inspected.get("preferred_stock_source", "Not recorded"),
            "Minority interest": inspected.get("minority_interest_source", "Not recorded"),
            "Short-term debt and leases": inspected.get(
                "current_debt_and_capital_lease_obligation_source", "Not recorded"
            ),
            "Current debt (Beneish TATA)": inspected.get(
                "current_debt_source", "Not recorded"
            ),
            "Current capital lease (debt reconciliation)": inspected.get(
                "current_capital_lease_obligation_source", "Not recorded"
            ),
            "Taxes payable (Beneish TATA)": inspected.get(
                "taxes_payable_source", "Not recorded"
            ),
        }
        st.dataframe(
            pd.DataFrame([{"Item": k, "Source / value": str(v)} for k, v in sources.items()]),
            hide_index=True, use_container_width=True,
        )

    quality_log = DATA_DIR / "quality_exclusions.csv"
    if quality_log.exists():
        st.markdown("**Companies excluded before screening**")
        st.caption(
            "The six companies below had nonpositive Beneish GMI or SGAI inputs. "
            "Their removal does not mean their financial statements are incorrect; "
            "these indices are difficult to interpret in the standard model."
        )
        st.dataframe(pd.read_csv(quality_log), hide_index=True, use_container_width=True)

with st.expander("Course formulas and interpretation"):
    st.markdown(r"""
**Valuation**

- Book-to-Market = common book equity / fiscal-date market equity (shares × closing price).
- Enterprise Earnings Yield = EBIT / TEV.
- **Professor's TEV** = fiscal-date market equity + total debt + preferred stock + minority interest − excess cash.
- **Professor's excess cash** = cash + current assets − current liabilities.

**Earnings quality**

- STA = (net income − operating cash flow) / total assets.
- SNOA = (operating assets − operating liabilities) / total assets.
- Operating assets = total assets − cash.
- Operating liabilities = total assets − short-term debt and leases − long-term debt and leases − minority interest − preferred stock − common book equity.
- PROBM = −4.84 + 0.920 DSRI + 0.528 GMI + 0.404 AQI + 0.892 SGI + 0.115 DEPI − 0.172 SGAI + 4.679 TATA − 0.327 LVGI.
- PMAN = standard normal CDF(PROBM). Beneish TATA uses the balance-sheet accruals formula from the Chapter 3 notebook, **not** STA.

**Profitability**

- GPA = (revenue − cost of goods sold) / total assets.
- ROC = EBIT / (net PP&E + current assets − current liabilities).

**Textbook Magic Formula (BUY research ranking)**

- Independently rank companies by ROC (highest = rank 1) and EEY (highest = rank 1).
- Magic Formula combined rank = ROC rank + EEY rank; lower totals rank first.
- This is a separate research screen from the custom SHORT warning score and uses
  the professor's ROC, TEV and excess-cash conventions recorded above.

For PMAN, all eight components and a consecutive preceding annual fiscal period must exist.
The screening universe also excludes any firm with a reported nonpositive
Beneish GMI or SGAI, even if its PMAN was already unavailable.
Missing values, nonpositive required denominators, and non-applicable models remain unavailable.
Some missing balance-sheet items are reconstructed only when Yahoo's reported
accounting identities reconcile; provenance is disclosed in the dataset.
Quote/statement currency mismatches and multiple traded share classes without
class-specific share counts cannot receive fiscal-date valuation ratios.
Sector-relative mode subtracts sector means; it does **not** divide by sector standard deviations.

All rankings and screening results recompute when the sidebar controls change, using only the saved CSV.
The three stage strengths use tie-aware relative rank percentiles, combined
into an app-created **Custom SHORT warning score** for surviving firms, displayed
as a **Relative SHORT concern ranking** out of 100. It is not a textbook formula.
Only the top ten SHORT survivors are shown in the main results table; the full
survivor set remains available for deeper inspection and CSV download.
""")

st.subheader("Magic Formula | Top 10 BUY research candidates")
st.caption(
    "**Textbook ranking method (distinct from our Custom SHORT warning score):** "
    "rank firms separately by Return on Capital (ROC) and Enterprise Earnings Yield (EEY), "
    "with the highest value receiving rank 1 for each metric. Add both ranks; "
    "the **lowest combined rank sum** is the most attractive under the Magic Formula. "
    "Equal metric values share the same rank, and ties in the combined total are "
    "listed alphabetically."
)
magic = magic_formula_rankings(screening_universe)
st.caption(
    f"{len(magic):,} of {len(screening_universe):,} selected companies have both "
    "ratios available; companies missing either input are omitted. "
    "This independent ranking uses the saved Yahoo Finance snapshot and the course's "
    "ROC and TEV conventions; SHORT-screen retention sliders do not affect it."
)
if magic.empty:
    st.warning("No companies in the selected sectors have both ROC and EEY available.")
else:
    st.dataframe(
        shown_magic_columns(magic.head(10)),
        hide_index=True, use_container_width=True, height=395,
    )
    st.download_button(
        "Download full Magic Formula ranking (CSV)",
        shown_magic_columns(magic).to_csv(index=False).encode("utf-8"),
        file_name="magic_formula_rankings.csv", mime="text/csv",
    )
st.caption(
    "Magic Formula rankings are a BUY **research starting point**, not trade instructions. "
    "High ROC can be inflated by a small capital denominator, and the accounting data and "
    "historical valuation snapshot need checking before any investment decision."
)
