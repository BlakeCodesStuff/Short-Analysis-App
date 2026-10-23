"""One-time Yahoo Finance download for the static ACCT 3343 SHORT dataset.

Run in Google Colab or locally.  The deployed Streamlit app never imports
yfinance, so using its interactive controls will never trigger new requests.

For a quick dry run change COLLECTION_LIMIT to 15, then restore 1200 before
creating the final dataset.  Do not submit the classroom's 50-firm universe.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery

from metrics import calculate_metrics, identify_beneish_index_exclusions


# User-editable settings. 1200 gives room for exclusions and unavailable data.
COLLECTION_LIMIT = 1200
PAGE_SIZE = 250
PAUSE_SECONDS = 0.35
PRICE_BATCH_SIZE = 40
SAVE_EVERY = 25
OUTPUT_DIRECTORY = Path("data")

# Exact Yahoo accounting row labels.  We only use genuinely matching concepts.
# No preferred-share COUNT is substituted for preferred-stock book VALUE.
FIELDS = {
    "income_stmt": {
        "total_revenue": ["Total Revenue", "Operating Revenue"],
        "cost_of_revenue": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
        "ebit": ["EBIT"],
        "net_income": ["Net Income"],
        "selling_general_and_administration": ["Selling General And Administration"],
    },
    "balance_sheet": {
        "total_assets": ["Total Assets"],
        "current_assets": ["Current Assets", "Total Current Assets"],
        "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
        "cash_and_cash_equivalents": ["Cash And Cash Equivalents", "Cash Financial"],
        "net_property_plant_equipment": ["Net PPE", "Property Plant Equipment Net"],
        "common_stock_equity": ["Common Stock Equity"],
        "stockholders_equity": ["Stockholders Equity"],
        "ordinary_shares_number": ["Ordinary Shares Number"],
        "total_debt": ["Total Debt"],
        "preferred_stock": ["Preferred Stock"],
        "minority_interest": ["Minority Interest"],
        "total_liabilities_net_minority_interest": ["Total Liabilities Net Minority Interest"],
        "total_equity_gross_minority_interest": ["Total Equity Gross Minority Interest"],
        "current_debt_and_capital_lease_obligation": ["Current Debt And Capital Lease Obligation"],
        "current_capital_lease_obligation": ["Current Capital Lease Obligation"],
        "long_term_debt_and_capital_lease_obligation": ["Long Term Debt And Capital Lease Obligation"],
        "long_term_capital_lease_obligation": ["Long Term Capital Lease Obligation"],
        "capital_lease_obligations": ["Capital Lease Obligations"],
        "accounts_receivable": ["Accounts Receivable", "Receivables"],
        "current_debt": ["Current Debt"],
        "long_term_debt": ["Long Term Debt"],
        "taxes_payable": ["Income Tax Payable", "Taxes Payable", "Income Taxes Payable"],
        # This aggregate is deliberately stored separately until its meaning
        # is assessed; it must never silently override a narrower tax line.
        "total_tax_payable": ["Total Tax Payable"],
    },
    "cashflow": {
        "operating_cash_flow": ["Operating Cash Flow", "Total Cash From Operating Activities"],
        # Combined D&A is an explicitly documented approximation for DEPI.
        "depreciation_and_amortization": [
            "Depreciation And Amortization", "Depreciation Amortization Depletion",
            "Reconciled Depreciation",
        ],
    },
}


def discover_tickers(limit: int) -> tuple[list[str], dict[str, dict]]:
    """Return a reproducible top-market-cap US exchange-listed ticker pool."""
    query = EquityQuery("and", [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
        EquityQuery("gt", ["intradaymarketcap", 3e8]),
    ])
    tickers, quote_by_ticker = [], {}
    for offset in range(0, limit, PAGE_SIZE):
        response = yf.screen(
            query, offset=offset, size=min(PAGE_SIZE, limit - offset),
            sortField="intradaymarketcap", sortAsc=False,
        )
        quotes = response.get("quotes", [])
        for quote in quotes:
            ticker = quote.get("symbol")
            if ticker and ticker not in quote_by_ticker:
                tickers.append(ticker)
                quote_by_ticker[ticker] = quote
        print(f"Screener page offset {offset}: {len(quotes)} matches; {len(tickers)} unique")
        if not quotes:
            break
    if not tickers:
        raise RuntimeError("Yahoo screener returned no company symbols")
    return tickers, quote_by_ticker


def _statement(ticker_object, kind: str) -> pd.DataFrame:
    try:
        value = getattr(ticker_object, kind)
        return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _find(statement: pd.DataFrame, period: pd.Timestamp, names: list[str]):
    if statement.empty or period not in statement.columns:
        return np.nan, ""
    for label in names:
        if label in statement.index:
            val = statement.at[label, period]
            if pd.notna(val):
                return val, label
    # Only formatting variants are matched automatically. Never fuzzy-match
    # accounting concepts (e.g. deferred taxes vs. taxes payable).
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(label).casefold()): label
        for label in statement.index
    }
    for label in names:
        actual = normalized.get(re.sub(r"[^a-z0-9]", "", label.casefold()))
        if actual is not None:
            val = statement.at[actual, period]
            if pd.notna(val):
                return val, str(actual)
    return np.nan, ""


def fetch_company(ticker: str, quote: dict) -> tuple[list[dict], list[str], dict]:
    """Read all matching annual periods while preserving failed/missing fields."""
    issues = []
    obj = yf.Ticker(ticker)
    try:
        info = obj.info or {}
    except Exception as exc:
        info = {}
        issues.append(f"metadata unavailable: {type(exc).__name__}")
    statements = {name: _statement(obj, name) for name in FIELDS}
    balance_labels = [str(v) for v in statements["balance_sheet"].index]
    label_audit = {
        "ticker": ticker,
        "available_tax_labels": " | ".join(
            v for v in balance_labels if "tax" in v.casefold()
        ),
        "available_debt_lease_labels": " | ".join(
            v for v in balance_labels
            if any(term in v.casefold() for term in ("debt", "lease", "borrow", "credit"))
        ),
    }
    for name, statement in statements.items():
        if statement.empty:
            issues.append(f"annual {name} unavailable")
    period_set = set()
    for statement in statements.values():
        if not statement.empty:
            period_set.update(pd.to_datetime(statement.columns, errors="coerce"))
    periods = sorted((p for p in period_set if pd.notna(p)), reverse=True)
    rows = []
    for period in periods:
        row = {
            "ticker": ticker,
            "company_name": info.get("longName") or info.get("shortName")
                            or quote.get("longName") or quote.get("shortName") or ticker,
            "sector": info.get("sector") or quote.get("sector"),
            "industry": info.get("industry") or quote.get("industry"),
            # Yahoo quotes can trade in a currency different from the
            # currency of the issuer's financial statements (e.g. ADRs).
            "quote_currency": info.get("currency") or quote.get("currency"),
            "financial_currency": info.get("financialCurrency") or quote.get("financialCurrency"),
            "fiscal_period_end": pd.Timestamp(period).date().isoformat(),
            "fiscal_year": int(pd.Timestamp(period).year),
        }
        for source, mapping in FIELDS.items():
            for name, yahoo_names in mapping.items():
                value, matched_label = _find(statements[source], period, yahoo_names)
                row[name] = value
                if name in ("taxes_payable", "total_tax_payable", "current_debt",
                            "current_capital_lease_obligation"):
                    row[f"{name}_reported_label"] = matched_label
        rows.append(row)
    if not rows:
        issues.append("no annual fiscal period returned")
    return rows, issues, label_audit


def collect_statements(tickers: list[str], quotes: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records, issues, label_audits = [], [], []
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for n, symbol in enumerate(tickers, 1):
        try:
            company_rows, failures, labels = fetch_company(symbol, quotes[symbol])
            records.extend(company_rows)
            label_audits.append(labels)
            for failure in failures:
                issues.append({"ticker": symbol, "issue": failure})
            print(f"[{n}/{len(tickers)}] {symbol}: {len(company_rows)} fiscal years")
        except Exception as exc:
            issues.append({"ticker": symbol, "issue": f"request failed: {type(exc).__name__}: {exc}"})
            print(f"[{n}/{len(tickers)}] {symbol}: failed")
        if n % SAVE_EVERY == 0:
            pd.DataFrame(records).to_csv(OUTPUT_DIRECTORY / "collection_checkpoint.csv", index=False)
            pd.DataFrame(issues, columns=["ticker", "issue"]).to_csv(
                OUTPUT_DIRECTORY / "collection_issues.csv", index=False
            )
            pd.DataFrame(label_audits).to_csv(
                OUTPUT_DIRECTORY / "field_labels_audit.csv", index=False
            )
        time.sleep(PAUSE_SECONDS)
    return (pd.DataFrame(records), pd.DataFrame(issues, columns=["ticker", "issue"]),
            pd.DataFrame(label_audits, columns=["ticker", "available_tax_labels",
                                               "available_debt_lease_labels"]))


def identify_share_classes(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one listing per issuer and bar multi-class proxy valuations.

    We cannot determine each class's historical shares outstanding from the
    consolidated Ordinary Shares Number alone, so we never multiply that
    consolidated count by one share class's price for a multi-class issuer.
    """
    history = history.copy()
    company_labels = (
        history[["ticker", "company_name"]].drop_duplicates("ticker")
        .assign(name_key=lambda d: d["company_name"].fillna("").str.strip().str.casefold())
    )
    # Stable first appearance follows the Yahoo screen's market-cap order.
    duplicated = company_labels["name_key"].ne("") & company_labels.duplicated("name_key", keep=False)
    multi_names = set(company_labels.loc[duplicated, "name_key"])
    secondary = company_labels.loc[duplicated & company_labels.duplicated("name_key"), "ticker"]
    history["multiple_traded_share_classes"] = (
        history["company_name"].fillna("").str.strip().str.casefold().isin(multi_names)
    )
    dropped = history["ticker"].isin(secondary)
    records = []
    if dropped.any():
        kept_by_name = company_labels.loc[~company_labels["ticker"].isin(secondary)]
        preferred = dict(zip(kept_by_name["name_key"], kept_by_name["ticker"]))
        for symbol, name_key in company_labels.loc[company_labels["ticker"].isin(secondary),
                                                   ["ticker", "name_key"]].itertuples(index=False):
            records.append({"ticker": symbol,
                            "issue": f"Other traded share class of {preferred[name_key]} excluded"})
    return history.loc[~dropped].copy(), pd.DataFrame(records, columns=["ticker", "issue"])


def add_historical_prices(financial: pd.DataFrame, issues: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each annual period to the latest prior daily Close within 10 days."""
    financial = financial.copy()
    financial["fiscal_period_end_close"] = np.nan
    financial["fiscal_price_date"] = pd.NaT
    all_tickers = sorted(financial["ticker"].unique())
    financial["fiscal_period_end"] = pd.to_datetime(financial["fiscal_period_end"])
    start = (financial["fiscal_period_end"].min() - pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    end = (financial["fiscal_period_end"].max() + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    problem_records = []
    for i in range(0, len(all_tickers), PRICE_BATCH_SIZE):
        batch = all_tickers[i:i + PRICE_BATCH_SIZE]
        try:
            prices = yf.download(
                batch, start=start, end=end, interval="1d", auto_adjust=False,
                group_by="ticker", threads=True, progress=False,
            )
        except Exception as exc:
            problem_records.extend({"ticker": t, "issue": f"price batch failed: {exc}"} for t in batch)
            continue
        for symbol in batch:
            try:
                if isinstance(prices.columns, pd.MultiIndex):
                    if symbol in prices.columns.get_level_values(0):
                        close = prices[symbol]["Close"]
                    else:
                        close = prices["Close"][symbol]
                else:
                    close = prices["Close"]
                close = close.dropna().sort_index()
                if close.empty:
                    raise ValueError("no close-price observations")
                close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
                mask = financial["ticker"].eq(symbol)
                periods = financial.loc[mask, "fiscal_period_end"]
                for row_idx, period in periods.items():
                    candidates = close.loc[(close.index <= period) & (close.index >= period - pd.Timedelta(days=10))]
                    if not candidates.empty:
                        financial.at[row_idx, "fiscal_period_end_close"] = float(candidates.iloc[-1])
                        financial.at[row_idx, "fiscal_price_date"] = candidates.index[-1]
            except Exception as exc:
                problem_records.append({"ticker": symbol, "issue": f"price matching: {exc}"})
        print(f"Prices: {min(i + PRICE_BATCH_SIZE, len(all_tickers))}/{len(all_tickers)}")
    if problem_records:
        issues = pd.concat([issues, pd.DataFrame(problem_records)], ignore_index=True)
    financial["fiscal_period_end"] = financial["fiscal_period_end"].dt.strftime("%Y-%m-%d")
    financial["fiscal_price_date"] = pd.to_datetime(financial["fiscal_price_date"]).dt.strftime("%Y-%m-%d")
    return financial, issues


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    tickers, quotes = discover_tickers(COLLECTION_LIMIT)
    history, issues, label_audits = collect_statements(tickers, quotes)
    if history.empty:
        raise RuntimeError("No annual financial statements downloaded")
    total_downloaded = history["ticker"].nunique()
    history, share_class_issues = identify_share_classes(history)
    issues = pd.concat([issues, share_class_issues], ignore_index=True)
    history, issues = add_historical_prices(history, issues)
    history.to_csv(OUTPUT_DIRECTORY / "financial_history.csv", index=False)
    label_audits.to_csv(OUTPUT_DIRECTORY / "field_labels_audit.csv", index=False)
    latest = calculate_metrics(history)
    finance_sector = latest["sector"].fillna("").str.lower().isin(["financials", "financial services"])
    sector_missing = latest["sector"].isna()
    latest.loc[finance_sector | sector_missing, "universe_exclusion"] = (
        "Financial sector or missing sector classification"
    )
    latest = latest.loc[~(finance_sector | sector_missing)].copy()
    nonfinancial_before_quality_review = len(latest)
    # Treat nonpositive Beneish index ratios as a data-quality exclusion for
    # this assignment. The original annual records remain in financial_history.
    quality_exclusions = identify_beneish_index_exclusions(latest)
    if not quality_exclusions.empty:
        issues = pd.concat([
            issues,
            quality_exclusions[["ticker", "reason"]].rename(columns={"reason": "issue"}),
        ], ignore_index=True)
        latest = latest.loc[~latest["ticker"].isin(quality_exclusions["ticker"])].copy()
    quality_exclusions.to_csv(OUTPUT_DIRECTORY / "quality_exclusions.csv", index=False)
    latest.to_csv(OUTPUT_DIRECTORY / "short_universe.csv", index=False)
    issues.to_csv(OUTPUT_DIRECTORY / "collection_issues.csv", index=False)
    snapshot = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance via yfinance",
        "requested_company_count": len(tickers),
        "companies_with_statements": total_downloaded,
        "unique_issuers_after_share_class_deduplication": history["ticker"].nunique(),
        "nonfinancial_companies_before_quality_review": nonfinancial_before_quality_review,
        "beneish_index_exclusions": len(quality_exclusions),
        "nonfinancial_companies": len(latest),
        "currency_mismatches_or_unverified": int(latest["valuation_currency_valid"].eq(False).sum()),
        "coverage_by_metric": {c: int(latest[c].notna().sum())
                               for c in ("bm", "eey", "sta", "snoa", "pman", "gpa", "roc")},
    }
    (OUTPUT_DIRECTORY / "snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(json.dumps(snapshot, indent=2))
    print("Saved data/short_universe.csv, quality_exclusions.csv, financial_history.csv, field_labels_audit.csv, issues, and snapshot.json")


if __name__ == "__main__":
    main()
