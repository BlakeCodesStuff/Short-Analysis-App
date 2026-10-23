"""ACCT 3343 SHORT Analysis: static-data, sequential interactive screening."""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from screening import LOW_IS_CONCERNING, concern_percentile, magic_formula_rankings, stage_screen


st.set_page_config(page_title="ACCT 3343 | SHORT Screening", page_icon="📉", layout="wide")
DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "short_universe.csv"
META_PATH = DATA_DIR / "snapshot.json"
HISTORY_PATH = DATA_DIR / "financial_history.csv"

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


@st.cache_data(show_spinner=False)
def read_financial_history(path: str, modified_ns: int) -> pd.DataFrame:
    """Load only the annual values used in the company-level charts."""
    fields = [
        "ticker", "financial_currency", "fiscal_period_end", "fiscal_year",
        "total_revenue", "net_income", "operating_cash_flow",
    ]
    frame = pd.read_csv(path, usecols=fields)
    frame["fiscal_period_end"] = pd.to_datetime(frame["fiscal_period_end"], errors="coerce")
    for field in ["total_revenue", "net_income", "operating_cash_flow"]:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return frame


def company_financial_history(
    history: pd.DataFrame, ticker: str, statement_currency: str,
) -> pd.DataFrame:
    """Keep annual dates in chronological order and avoid mixed-currency charts."""
    company = history.loc[history["ticker"].eq(ticker)].copy()
    if statement_currency and statement_currency.lower() != "nan":
        company = company.loc[company["financial_currency"].eq(statement_currency)]
    company = company.dropna(subset=["fiscal_period_end"])
    company = company.sort_values("fiscal_period_end").drop_duplicates(
        subset=["fiscal_period_end"], keep="last"
    )
    company["fiscal_label"] = "FY " + company["fiscal_year"].astype("Int64").astype(str)
    return company


def annual_revenue_chart(history: pd.DataFrame, currency: str) -> go.Figure:
    available = history.dropna(subset=["total_revenue"])
    figure = go.Figure(go.Scatter(
        x=available["fiscal_label"], y=available["total_revenue"],
        mode="lines+markers", name="Revenue",
        line={"color": "#4DCCBD", "width": 3}, marker={"size": 9},
        customdata=available["fiscal_period_end"].dt.strftime("%Y-%m-%d"),
        hovertemplate=(
            "%{x} (ended %{customdata})<br>Revenue: %{y:,.0f} "
            + escape(currency) + "<extra></extra>"
        ),
    ))
    _style_annual_chart(figure, "Revenue", currency)
    return figure


def annual_earnings_cash_chart(history: pd.DataFrame, currency: str) -> go.Figure:
    figure = go.Figure()
    for column, label, color in [
        ("net_income", "Net income", "#8980C9"),
        ("operating_cash_flow", "Operating cash flow", "#4DCCBD"),
    ]:
        available = history.dropna(subset=[column])
        figure.add_trace(go.Bar(
            x=available["fiscal_label"], y=available[column],
            name=label, marker_color=color,
            customdata=available["fiscal_period_end"].dt.strftime("%Y-%m-%d"),
            hovertemplate=(
                "%{x} (ended %{customdata})<br>" + label + ": %{y:,.0f} "
                + escape(currency) + "<extra></extra>"
            ),
        ))
    _style_annual_chart(figure, "Net income and operating cash flow", currency)
    figure.update_layout(barmode="group", legend={"orientation": "h", "y": 1.15})
    return figure


def _style_annual_chart(figure: go.Figure, title: str, currency: str) -> None:
    figure.update_layout(
        title={"text": title, "font": {"size": 16}},
        height=330, margin={"l": 70, "r": 18, "t": 70, "b": 60},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#EAFBF7"},
        xaxis={"title": "Fiscal year", "type": "category", "automargin": True},
        yaxis={
            "title": f"Reported {currency} (scaled)", "tickformat": "~s",
            "gridcolor": "#404565", "zerolinecolor": "#565A78", "automargin": True,
        },
        hoverlabel={"bgcolor": "#282946", "font": {"color": "#EAFBF7"}},
    )


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


def relative_peer_assessment(metric: str, value: float, median: float) -> str:
    """Describe the direction of a raw ratio relative to its peer median."""
    if pd.isna(value) or pd.isna(median):
        return "⚪ Data unavailable"
    if np.isclose(value, median, rtol=0.01, atol=1e-9):
        return "⚪ Near peer median"
    more_concerning = value < median if metric in LOW_IS_CONCERNING else value > median
    return "🔴 Higher SHORT concern" if more_concerning else "🟢 Lower SHORT concern"


def peer_gap_display(metric: str, value: float, median: float) -> str:
    if pd.isna(value) or pd.isna(median):
        return "Unavailable"
    difference = value - median
    if metric == "bm":
        return f"{difference:+.3f}"
    return f"{difference * 100:+.2f} pp"


def mitigating_metric(
    universe: pd.DataFrame, sector: str, company: pd.Series,
    selected_metrics: list[str],
) -> tuple[str, dict] | None:
    """Pick an available indicator that is genuinely less concerning than peers."""
    candidates = []
    for metric in LABELS:
        value = company.get(metric, float("nan"))
        if pd.isna(value):
            continue
        stats = peer_statistics(universe, sector, metric, value)
        if not np.isfinite(stats["percentile"]):
            continue
        if relative_peer_assessment(metric, value, stats["median"]) != "🟢 Lower SHORT concern":
            continue
        candidates.append((stats["percentile"], metric not in selected_metrics, metric, stats))
    if not candidates:
        return None
    _, _, metric, stats = min(candidates)
    return metric, stats


def short_candidate_scatter(
    frame: pd.DataFrame, selected_metrics: list[str], zoom: float = 1.0,
) -> go.Figure:
    """Show the three existing stage rankings without recalculating their scores."""
    if frame.empty:
        raise ValueError("Cannot plot an empty SHORT candidate screen")
    if zoom < 1:
        raise ValueError("Zoom must be 1× or greater")

    # `frame` is already ordered by its overall SHORT warning score.
    top_tickers = set(frame.head(10)["ticker"])
    plot = frame.copy()
    plot["is_top_ten"] = plot["ticker"].isin(top_tickers)
    figure = go.Figure()

    for top_ten, name, symbol, size, outline in [
        (False, "Other surviving companies", "circle", 10, "#171B32"),
        (True, "Top 10 SHORT candidates", "diamond", 15, "#EAFBF7"),
    ]:
        subset = plot.loc[plot["is_top_ten"].eq(top_ten)]
        if subset.empty:
            continue

        hover_text = []
        for _, company in subset.iterrows():
            details = [
                f"<b>{escape(str(company['company_name']))} ({escape(str(company['ticker']))})</b>",
                f"Sector: {escape(str(company['sector']))}",
                f"Valuation concern: {company['valuation_strength'] * 100:.1f}/100",
                f"Earnings-quality concern: {company['quality_strength'] * 100:.1f}/100",
                f"Profitability concern: {company['performance_strength'] * 100:.1f}/100",
                f"Custom SHORT warning score: {company['short_warning_score']:.1f}/100",
            ]
            details.extend(
                f"{metric.upper()}: {metric_display(metric, company[metric])}"
                for metric in selected_metrics
            )
            hover_text.append("<br>".join(details))

        figure.add_trace(go.Scatter(
            x=subset["valuation_strength"] * 100,
            y=subset["quality_strength"] * 100,
            mode="markers",
            name=name,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            marker={
                "color": subset["performance_strength"] * 100,
                "coloraxis": "coloraxis",
                "symbol": symbol,
                "size": size,
                "opacity": 0.9,
                "line": {"color": outline, "width": 1.4 if top_ten else 0.5},
            },
        ))

    # Increasing the zoom focuses on the high-concern upper-right corner,
    # without removing or recomputing any of the underlying companies.
    zoom_lower_bound = 100 - 100 / zoom
    figure.update_layout(
        height=540,
        margin={"l": 76, "r": 96, "t": 46, "b": 82},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#EAFBF7"},
        coloraxis={
            "cmin": 0,
            "cmax": 100,
            "colorscale": [[0, "#4DCCBD"], [0.5, "#8980C9"], [1, "#FF8484"]],
            "colorbar": {"title": {"text": "Profitability<br>concern"}, "ticksuffix": "/100"},
        },
        legend={"orientation": "h", "y": 1.07, "x": 0, "yanchor": "bottom"},
        xaxis={
            "title": "Valuation concern ranking (/100) →",
            "range": [zoom_lower_bound, 101],
            "dtick": 20,
            "gridcolor": "#404565",
            "zeroline": False,
            "automargin": True,
        },
        yaxis={
            "title": "Earnings-quality concern ranking (/100) →",
            "range": [zoom_lower_bound, 101],
            "dtick": 20,
            "gridcolor": "#404565",
            "zeroline": False,
            "automargin": True,
        },
        hoverlabel={"bgcolor": "#282946", "font": {"color": "#EAFBF7"}},
    )
    return figure


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
raw_collection_date = snapshot.get("collected_at_utc", "")
try:
    collection_date = datetime.fromisoformat(raw_collection_date).strftime("%B %d, %Y").replace(" 0", " ")
except (TypeError, ValueError):
    collection_date = "not recorded"
raw_reference_date = snapshot.get("analysis_reference_date", "")
try:
    analysis_reference_date = datetime.fromisoformat(raw_reference_date).strftime("%B %d, %Y").replace(" 0", " ")
except (TypeError, ValueError):
    analysis_reference_date = "not recorded"
fiscal_dates = pd.to_datetime(df["fiscal_period_end"], errors="coerce").dropna()
fiscal_period_range = (
    f"{fiscal_dates.min():%Y-%m-%d} to {fiscal_dates.max():%Y-%m-%d}"
    if not fiscal_dates.empty else "not available"
)
st.caption(
    f"Source: Yahoo Finance · Analysis reference date: {analysis_reference_date} "
    f"· Data collected: {collection_date} (UTC) "
    f"· Fiscal year-ends: {fiscal_period_range} "
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

    st.subheader("SHORT candidate comparison | Interactive scatter plot")
    st.caption(
        "Each dot is a company that survived all three screens. Farther right means greater "
        "relative valuation concern; higher means greater earnings-quality concern. "
        "Colour shows profitability concern (teal = lower, coral = higher), and outlined "
        "diamonds identify the current top 10 SHORT candidates. Hover for company details "
        "and the selected financial ratios. These are relative rankings within each "
        "screening stage, not probabilities or predicted returns."
    )
    chart_zoom = st.select_slider(
        "Scatter plot zoom (toward higher-concern companies)",
        options=[1.0, 1.5, 2.0, 3.0, 4.0],
        value=1.0,
        format_func=lambda zoom: f"{zoom:g}×",
        help="Extra 1.5× and 3× steps sit between the previous 1×, 2×, and 4× levels. "
             "This only changes the plot view; use mouse-wheel/trackpad zoom for finer, "
             "cursor-centred adjustments, or double-click the plot to reset its axes.",
    )
    st.plotly_chart(
        short_candidate_scatter(p, valuation_metrics + quality_metrics + performance_metrics, chart_zoom),
        use_container_width=True,
        theme=None,
        config={"displaylogo": False, "scrollZoom": True},
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

    st.markdown("**Historical financial performance**")
    st.caption(
        "Annual financial statements for this issuer, ordered oldest to newest. "
        "These are reported nominal amounts in the statement currency, not share-price "
        "returns. Missing annual values are omitted, never replaced by zero."
    )
    if HISTORY_PATH.exists():
        try:
            history = read_financial_history(str(HISTORY_PATH), HISTORY_PATH.stat().st_mtime_ns)
            statement_currency = str(chosen.get("financial_currency", ""))
            company_history = company_financial_history(history, symbol, statement_currency)
            historical_values = company_history[[
                "total_revenue", "net_income", "operating_cash_flow"
            ]]
            if historical_values.notna().any().any():
                revenue_col, cash_col = st.columns(2)
                with revenue_col:
                    st.plotly_chart(
                        annual_revenue_chart(company_history, statement_currency),
                        use_container_width=True, theme=None,
                        config={"displaylogo": False},
                    )
                with cash_col:
                    st.plotly_chart(
                        annual_earnings_cash_chart(company_history, statement_currency),
                        use_container_width=True, theme=None,
                        config={"displaylogo": False},
                    )
                st.caption(
                    f"{len(company_history)} annual periods available · "
                    f"Statement currency: {statement_currency}. "
                    "Each graph uses the dates and amounts in the saved historical dataset."
                )
            else:
                st.caption("Historical revenue, income, and cash-flow figures are unavailable for this issuer.")
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f"Historical financial charts could not be loaded: {exc}")
    else:
        st.caption("Historical charts require `data/financial_history.csv` in the deployed repository.")

    stage_signals = [
        ("Valuation concern", valuation_metrics),
        ("Accounting-quality concern", quality_metrics),
        ("Operating concern", performance_metrics),
    ]
    st.markdown("**Balanced financial observations**")
    st.caption(
        "The three screening categories identify financial questions worth investigating. "
        "The comparisons below use the initial selected-sector universe, not just the "
        "companies that survived the screen."
    )
    for category, stage_metrics in stage_signals:
        ranked = sorted(
            stage_metrics,
            key=lambda m: peer_statistics(screening_universe, sector, m, chosen[m])["percentile"]
            if np.isfinite(peer_statistics(screening_universe, sector, m, chosen[m])["percentile"])
            else -1,
            reverse=True,
        )
        strongest = ranked[0]
        stats = peer_statistics(screening_universe, sector, strongest, chosen[strongest])
        versus_median_is_concerning = relative_peer_assessment(
            strongest, chosen[strongest], stats["median"]
        ) == "🔴 Higher SHORT concern"
        interpretation = (
            SHORT_MEANING[strongest] if versus_median_is_concerning
            else "This ratio is not materially worse than the peer median, "
                 "so it is not independently strong evidence for a SHORT position."
        )
        st.markdown(
            f"- **{category}:** {strongest.upper()} is "
            f"{metric_display(strongest, chosen[strongest])} versus "
            f"{metric_display(strongest, stats['median'])} median among "
            f"{stats['sample']} {stats['group']} observations "
            f"({relative_peer_assessment(strongest, chosen[strongest], stats['median'])}). "
            f"{interpretation}"
        )

    selected_all = valuation_metrics + quality_metrics + performance_metrics
    mitigating = mitigating_metric(screening_universe, sector, chosen, selected_all)
    if mitigating:
        metric, stats = mitigating
        scope = "selected screening indicator" if metric in selected_all else "additional available indicator"
        st.markdown(
            f"- **Counterargument / mitigating factor:** {metric.upper()} "
            f"({scope}) is {metric_display(metric, chosen[metric])}, compared with "
            f"the {stats['group']} median of {metric_display(metric, stats['median'])} "
            f"(n={stats['sample']}). This is less concerning than peers and could "
            "weaken the provisional SHORT case; the underlying business context "
            "and latest filings still need review."
        )
    else:
        st.markdown(
            "- **Counterargument / mitigating factor:** None of the available seven "
            "ratios is clearly better than its peer median. This does not rule out "
            "qualitative counterarguments, which require company-specific research."
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
        stats = peer_statistics(screening_universe, sector, name, chosen[name])
        comparisons.append({
            "Metric": name.upper(),
            "Company": metric_display(name, chosen[name]),
            "Peer median": metric_display(name, stats["median"]),
            "Difference vs median": peer_gap_display(name, chosen[name], stats["median"]),
            "Relative to peers": relative_peer_assessment(name, chosen[name], stats["median"]),
            "Concern percentile (/100)": round(stats["percentile"], 1)
            if pd.notna(stats["percentile"]) else float("nan"),
            "Peer group": stats["group"],
            "Peer observations": stats["sample"],
            "Selected for screening": name in selected_all,
        })
    st.markdown("**Financial evidence and peer comparison**")
    st.dataframe(pd.DataFrame(comparisons), hide_index=True, use_container_width=True)
    st.caption(
        "Difference vs median uses percentage points (pp) for ratios other than B/M, "
        "which is displayed as a multiple. Red indicates a less favourable result "
        "for SHORT screening; green indicates a potentially mitigating one. "
        "Concern percentiles range from 0 to 100 within the initial sector group "
        "(or all selected sectors if fewer than five peers). Higher means more "
        "relative concern, not greater probability of loss. Unavailable metrics "
        "remain visible and are never treated as zero."
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
