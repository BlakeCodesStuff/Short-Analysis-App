"""Course-specific accounting metrics for the ACCT 3343 SHORT screen.

Input: one row per ticker and annual fiscal period from Yahoo Finance.
No unavailable amount is silently replaced by zero.  Derived metrics are
NaN if their required inputs or economically meaningful denominators fail.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd


STATEMENT_INPUTS = [
    "total_revenue", "cost_of_revenue", "ebit", "net_income",
    "operating_cash_flow", "total_assets", "current_assets",
    "current_liabilities", "cash_and_cash_equivalents",
    "net_property_plant_equipment", "common_stock_equity",
    "stockholders_equity", "ordinary_shares_number",
    "fiscal_period_end_close", "total_debt", "preferred_stock",
    "minority_interest", "current_debt_and_capital_lease_obligation",
    "current_capital_lease_obligation", "capital_lease_obligations",
    "long_term_capital_lease_obligation",
    "long_term_debt_and_capital_lease_obligation",
    "total_liabilities_net_minority_interest", "total_equity_gross_minority_interest",
    "accounts_receivable", "depreciation_and_amortization",
    "selling_general_and_administration", "current_debt", "long_term_debt",
    "taxes_payable", "total_tax_payable",
]

PRIOR_INPUTS = [
    "total_revenue", "cost_of_revenue", "accounts_receivable",
    "current_assets", "current_liabilities", "cash_and_cash_equivalents",
    "net_property_plant_equipment", "total_assets",
    "depreciation_and_amortization", "selling_general_and_administration",
    "current_debt", "long_term_debt", "taxes_payable",
    "current_debt_used", "taxes_payable_used",
]

SCREEN_METRICS = {
    "bm": "Book-to-Market",
    "eey": "Enterprise Earnings Yield",
    "sta": "Simplified Total Accruals",
    "snoa": "Scaled Net Operating Assets",
    "pman": "Beneish PMAN",
    "gpa": "Gross Profitability / Assets",
    "roc": "Return on Capital",
}


def safe_ratio(numerator: pd.Series, denominator: pd.Series, *, positive: bool = True) -> pd.Series:
    """Ratio preserving missing values and rejecting non-finite observations."""
    n = pd.to_numeric(numerator, errors="coerce")
    d = pd.to_numeric(denominator, errors="coerce")
    allowed = n.notna() & d.notna() & (d.gt(0) if positive else d.ne(0))
    out = n.div(d).where(allowed)
    return out.where(np.isfinite(out))


def _missing_reasons(df: pd.DataFrame, metric: str, required: list[str], invalid_message: str) -> None:
    """Store a concrete reason for every missing metric without imputations."""
    required_missing = df[required].isna()
    labels = required_missing.apply(
        lambda row: ", ".join(row.index[row]), axis=1
    )
    out = pd.Series("", index=df.index, dtype="object")
    unavailable = df[metric].isna()
    out.loc[unavailable] = invalid_message
    has_missing = unavailable & labels.ne("")
    out.loc[has_missing] = "Missing: " + labels.loc[has_missing]
    df[f"{metric}_reason"] = out


def calculate_metrics(raw_history: pd.DataFrame) -> pd.DataFrame:
    """Return one latest fiscal-year row per firm with auditable course metrics.

    All Beneish components use the same current/prior consecutive annual pair.
    STA uses the NI-CFO approach; Beneish TATA uses the professor's explicit
    balance-sheet change formula.  The course-specific excess cash definition
    is cash + current assets - current liabilities, without adjustment.
    """
    if raw_history.empty:
        raise ValueError("No financial history supplied")

    df = raw_history.copy()
    essential = {"ticker", "fiscal_period_end", "sector"}
    missing_columns = essential - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    for col in STATEMENT_INPUTS:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("quote_currency", "financial_currency"):
        if col not in df:
            df[col] = np.nan
        df[col] = df[col].astype("string").str.strip().str.upper()
    if "multiple_traded_share_classes" not in df:
        df["multiple_traded_share_classes"] = False
    df["multiple_traded_share_classes"] = (
        df["multiple_traded_share_classes"].astype("boolean").fillna(False).astype(bool)
    )

    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"], errors="coerce")
    if df["fiscal_period_end"].isna().any() or df["ticker"].isna().any():
        raise ValueError("Invalid ticker or fiscal-period key")
    if df.duplicated(["ticker", "fiscal_period_end"]).any():
        raise ValueError("Duplicate ticker/fiscal period rows need investigation")
    df = df.sort_values(["ticker", "fiscal_period_end"]).copy()

    # Resolve eligible TATA inputs BEFORE shifting periods. An original item
    # remains untouched; *_used and *_source disclose any alternative used.
    df["taxes_payable_used"] = df["taxes_payable"].copy()
    df["taxes_payable_source"] = np.where(
        df["taxes_payable"].notna(), "Yahoo direct income/taxes payable", "Unavailable"
    )
    aggregate_tax_fallback = (
        df["taxes_payable_used"].isna() & df["total_tax_payable"].ge(0)
    )
    df.loc[aggregate_tax_fallback, "taxes_payable_used"] = (
        df.loc[aggregate_tax_fallback, "total_tax_payable"]
    )
    df.loc[aggregate_tax_fallback, "taxes_payable_source"] = (
        "Yahoo Total Tax Payable (aggregate tax-payable fallback)"
    )

    df["current_capital_lease_obligation_used"] = (
        df["current_capital_lease_obligation"].copy()
    )
    df["current_capital_lease_obligation_source"] = np.where(
        df["current_capital_lease_obligation"].notna(),
        "Yahoo Current Capital Lease Obligation", "Unavailable"
    )
    lease_residual = (
        df["capital_lease_obligations"] - df["long_term_capital_lease_obligation"]
    )
    rounding_tolerance = df["total_assets"].abs().mul(1e-6).clip(lower=1.0)
    reconstruct_current_lease = (
        df["current_capital_lease_obligation_used"].isna()
        & df["total_assets"].gt(0) & lease_residual.notna()
        & lease_residual.ge(-rounding_tolerance)
        & lease_residual.le(df["capital_lease_obligations"] + rounding_tolerance)
    )
    df.loc[reconstruct_current_lease, "current_capital_lease_obligation_used"] = (
        lease_residual.loc[reconstruct_current_lease].clip(lower=0)
    )
    df.loc[reconstruct_current_lease, "current_capital_lease_obligation_source"] = (
        "Capital Lease Obligations minus Long Term Capital Lease Obligation"
    )

    df["current_debt_used"] = df["current_debt"].copy()
    df["current_debt_source"] = np.where(
        df["current_debt"].notna(), "Yahoo Current Debt", "Unavailable"
    )
    debt_excluding_lease = (
        df["current_debt_and_capital_lease_obligation"]
        - df["current_capital_lease_obligation_used"]
    )
    reconstruct_current_debt = (
        df["current_debt_used"].isna()
        & df["current_debt_and_capital_lease_obligation"].notna()
        & df["current_capital_lease_obligation_used"].notna()
        & debt_excluding_lease.ge(-rounding_tolerance)
        & debt_excluding_lease.le(
            df["current_debt_and_capital_lease_obligation"] + rounding_tolerance
        )
    )
    df.loc[reconstruct_current_debt, "current_debt_used"] = (
        debt_excluding_lease.loc[reconstruct_current_debt].clip(lower=0)
    )
    df.loc[reconstruct_current_debt, "current_debt_source"] = (
        "Current Debt And Capital Lease Obligation minus Current Capital Lease Obligation"
    )
    # A DIRECTLY REPORTED zero aggregate proves both components are zero;
    # a zero inferred from Total Debt minus long-term debt does not.
    reported_no_current_debt_or_lease = (
        df["current_debt_used"].isna()
        & df["current_debt_and_capital_lease_obligation"].eq(0)
    )
    df.loc[reported_no_current_debt_or_lease, "current_debt_used"] = 0.0
    df.loc[reported_no_current_debt_or_lease, "current_debt_source"] = (
        "Zero: directly reported current debt-and-lease aggregate is zero"
    )

    grp = df.groupby("ticker", sort=False)
    df["prior_fiscal_period_end"] = grp["fiscal_period_end"].shift(1)
    for col in PRIOR_INPUTS:
        df[f"prior_{col}"] = grp[col].shift(1)
    gap = (df["fiscal_period_end"] - df["prior_fiscal_period_end"]).dt.days
    df["valid_prior_year"] = gap.between(330, 400) & (
        df["fiscal_period_end"].dt.year - df["prior_fiscal_period_end"].dt.year == 1
    )

    # Chapter 1/2 profitability and valuation, with fiscal-date market equity.
    df["nwc"] = df["current_assets"] - df["current_liabilities"]
    df["valuation_currency_valid"] = (
        df["quote_currency"].notna() & df["financial_currency"].notna()
        & df["quote_currency"].eq(df["financial_currency"])
    ).fillna(False)
    # Multiplying fiscal shares by a quote is valid only if the statements
    # and market quote use the same currency and all outstanding shares can
    # be valued by that one quoted share class.
    valid_market_equity = (
        df["valuation_currency_valid"] & ~df["multiple_traded_share_classes"]
    )
    df["market_equity"] = (
        df["ordinary_shares_number"] * df["fiscal_period_end_close"]
    ).where(valid_market_equity)

    # Chapter 3's verified balance-sheet reconstructions. Preserve each
    # original Yahoo field and separately record when an identity is used.
    tolerance = df["total_assets"].abs() * 1e-4
    preferred_equity_difference = df["stockholders_equity"] - df["common_stock_equity"]
    valid_preferred_difference = (
        df["preferred_stock"].isna()
        & df["stockholders_equity"].notna()
        & df["common_stock_equity"].notna()
        & df["total_assets"].gt(0)
        & preferred_equity_difference.ge(-tolerance)
        & preferred_equity_difference.le(df["total_assets"] + tolerance)
    )
    df["preferred_stock_used"] = df["preferred_stock"].copy()
    df["preferred_stock_source"] = np.where(
        df["preferred_stock"].notna(), "Yahoo Preferred Stock", "Unavailable"
    )
    df.loc[valid_preferred_difference, "preferred_stock_used"] = (
        preferred_equity_difference.loc[valid_preferred_difference].clip(lower=0)
    )
    df.loc[valid_preferred_difference, "preferred_stock_source"] = (
        "Stockholders Equity minus Common Stock Equity"
    )

    from_gross_equity = (
        df["total_equity_gross_minority_interest"] - df["stockholders_equity"]
    )
    from_liability_identity = (
        df["total_assets"] - df["total_liabilities_net_minority_interest"]
        - df["stockholders_equity"]
    )
    both_sources = from_gross_equity.notna() & from_liability_identity.notna()
    identities_agree = ~(
        both_sources & (from_gross_equity - from_liability_identity).abs().gt(tolerance)
    )
    minority_residual = from_gross_equity.combine_first(from_liability_identity)
    valid_minority_residual = (
        df["minority_interest"].isna() & df["total_assets"].gt(0)
        & minority_residual.notna() & minority_residual.ge(-tolerance)
        & minority_residual.le(df["total_assets"] + tolerance)
        & identities_agree
    )
    df["minority_interest_used"] = df["minority_interest"].copy()
    df["minority_interest_source"] = np.where(
        df["minority_interest"].notna(), "Yahoo Minority Interest", "Unavailable"
    )
    df.loc[valid_minority_residual, "minority_interest_used"] = (
        minority_residual.loc[valid_minority_residual].clip(lower=0)
    )
    df.loc[valid_minority_residual, "minority_interest_source"] = np.where(
        from_gross_equity.loc[valid_minority_residual].notna(),
        "Total Equity Gross Minority Interest minus Stockholders Equity",
        "Assets minus Liabilities Net Minority Interest minus Stockholders Equity",
    )

    current_debt_residual = (
        df["total_debt"] - df["long_term_debt_and_capital_lease_obligation"]
    )
    # Yahoo's "Total Debt" occasionally disagrees with the sum of its
    # current and long-term debt/lease lines. Allow a missing-current-debt
    # reconstruction only for issuers with an independently observed,
    # reconciling annual debt split AND no conflicting observed annual split.
    reported_debt_split = df[[
        "total_debt", "long_term_debt_and_capital_lease_obligation",
        "current_debt_and_capital_lease_obligation",
    ]].notna().all(axis=1)
    debt_reconciliation_tolerance = df["total_assets"].abs().mul(1e-6).clip(lower=1.0)
    debt_gap = (
        current_debt_residual - df["current_debt_and_capital_lease_obligation"]
    )
    confirmed_debt_year = reported_debt_split & debt_gap.abs().le(debt_reconciliation_tolerance)
    conflicting_debt_year = reported_debt_split & debt_gap.abs().gt(debt_reconciliation_tolerance)
    confirmed_issuer = confirmed_debt_year.groupby(df["ticker"]).transform("any")
    conflicting_issuer = conflicting_debt_year.groupby(df["ticker"]).transform("any")
    valid_current_debt_residual = (
        df["current_debt_and_capital_lease_obligation"].isna()
        & df["total_assets"].gt(0) & current_debt_residual.notna()
        & current_debt_residual.ge(-tolerance)
        & current_debt_residual.le(df["total_debt"] + tolerance)
        & confirmed_issuer & ~conflicting_issuer
    )
    df["current_debt_and_capital_lease_obligation_used"] = (
        df["current_debt_and_capital_lease_obligation"].copy()
    )
    df["current_debt_and_capital_lease_obligation_source"] = np.where(
        df["current_debt_and_capital_lease_obligation"].notna(),
        "Yahoo Current Debt And Capital Lease Obligation", "Unavailable"
    )
    df.loc[
        valid_current_debt_residual, "current_debt_and_capital_lease_obligation_used"
    ] = current_debt_residual.loc[valid_current_debt_residual].clip(lower=0)
    df.loc[valid_current_debt_residual,
           "current_debt_and_capital_lease_obligation_source"] = (
        "Reconciled: Total Debt minus Long Term Debt And Capital Lease Obligation"
    )
    df["gpa"] = safe_ratio(
        df["total_revenue"] - df["cost_of_revenue"], df["total_assets"]
    )
    df["roc"] = safe_ratio(
        df["ebit"], df["net_property_plant_equipment"] + df["nwc"]
    )
    df["bm"] = safe_ratio(df["common_stock_equity"], df["market_equity"])
    # Negative book equity is not interpreted as "cheap/expensive" B/M.
    df["bm"] = df["bm"].where(df["common_stock_equity"].gt(0))

    df["excess_cash_course"] = (
        df["cash_and_cash_equivalents"] + df["current_assets"] - df["current_liabilities"]
    )
    df["tev_course"] = (
        df["market_equity"] + df["total_debt"] + df["preferred_stock_used"]
        + df["minority_interest_used"] - df["excess_cash_course"]
    )
    df["eey"] = safe_ratio(df["ebit"], df["tev_course"])

    # Chapter 3 quality signals.
    df["sta"] = safe_ratio(
        df["net_income"] - df["operating_cash_flow"], df["total_assets"]
    )
    operating_assets = df["total_assets"] - df["cash_and_cash_equivalents"]
    operating_liabilities = (
        df["total_assets"] - df["current_debt_and_capital_lease_obligation_used"]
        - df["long_term_debt_and_capital_lease_obligation"]
        - df["minority_interest_used"] - df["preferred_stock_used"]
        - df["common_stock_equity"]
    )
    df["snoa"] = safe_ratio(operating_assets - operating_liabilities, df["total_assets"])

    ar_rate_t = safe_ratio(df["accounts_receivable"], df["total_revenue"])
    ar_rate_p = safe_ratio(df["prior_accounts_receivable"], df["prior_total_revenue"])
    df["dsri"] = safe_ratio(ar_rate_t, ar_rate_p)

    gross_margin_t = safe_ratio(df["total_revenue"] - df["cost_of_revenue"], df["total_revenue"])
    gross_margin_p = safe_ratio(
        df["prior_total_revenue"] - df["prior_cost_of_revenue"], df["prior_total_revenue"]
    )
    df["gmi"] = safe_ratio(gross_margin_p, gross_margin_t, positive=False)
    asset_quality_t = 1 - safe_ratio(
        df["current_assets"] + df["net_property_plant_equipment"], df["total_assets"]
    )
    asset_quality_p = 1 - safe_ratio(
        df["prior_current_assets"] + df["prior_net_property_plant_equipment"],
        df["prior_total_assets"],
    )
    df["aqi"] = safe_ratio(asset_quality_t, asset_quality_p, positive=False)
    df["sgi"] = safe_ratio(df["total_revenue"], df["prior_total_revenue"])
    dep_rate_t = safe_ratio(
        df["depreciation_and_amortization"],
        df["depreciation_and_amortization"] + df["net_property_plant_equipment"],
    )
    dep_rate_p = safe_ratio(
        df["prior_depreciation_and_amortization"],
        df["prior_depreciation_and_amortization"] + df["prior_net_property_plant_equipment"],
    )
    df["depi"] = safe_ratio(dep_rate_p, dep_rate_t, positive=False)
    sga_rate_t = safe_ratio(df["selling_general_and_administration"], df["total_revenue"])
    sga_rate_p = safe_ratio(
        df["prior_selling_general_and_administration"], df["prior_total_revenue"]
    )
    df["sgai"] = safe_ratio(sga_rate_t, sga_rate_p, positive=False)
    leverage_t = safe_ratio(df["long_term_debt"] + df["current_liabilities"], df["total_assets"])
    leverage_p = safe_ratio(
        df["prior_long_term_debt"] + df["prior_current_liabilities"],
        df["prior_total_assets"],
    )
    df["lvgi"] = safe_ratio(leverage_t, leverage_p, positive=False)

    # Explicit professor's Beneish TATA: a balance-sheet change measure.
    total_accruals = (
        (df["current_assets"] - df["prior_current_assets"])
        - (df["cash_and_cash_equivalents"] - df["prior_cash_and_cash_equivalents"])
        - (
            (df["current_liabilities"] - df["prior_current_liabilities"])
            - (df["current_debt_used"] - df["prior_current_debt_used"])
            - (df["taxes_payable_used"] - df["prior_taxes_payable_used"])
        )
        - df["depreciation_and_amortization"]
    )
    df["tata"] = safe_ratio(total_accruals, df["total_assets"])
    beneish = ["dsri", "gmi", "aqi", "sgi", "depi", "sgai", "tata", "lvgi"]
    eligible = (
        df["valid_prior_year"]
        & ~df["sector"].fillna("").str.lower().isin(["financial services", "financials"])
        & df[beneish].notna().all(axis=1)
    )
    df["probm"] = (
        -4.84 + 0.920 * df["dsri"] + 0.528 * df["gmi"]
        + 0.404 * df["aqi"] + 0.892 * df["sgi"]
        + 0.115 * df["depi"] - 0.172 * df["sgai"]
        + 4.679 * df["tata"] - 0.327 * df["lvgi"]
    ).where(eligible)
    normal = NormalDist()
    df["pman"] = df["probm"].map(
        lambda value: normal.cdf(float(value)) if pd.notna(value) else np.nan
    )

    _missing_reasons(df, "bm", ["common_stock_equity", "market_equity"],
                     "Nonpositive book equity or market capitalization")
    _missing_reasons(df, "eey", ["ebit", "market_equity", "total_debt", "preferred_stock_used",
                                 "minority_interest_used", "excess_cash_course"],
                     "Nonpositive TEV under course definition")
    _missing_reasons(df, "roc", ["ebit", "net_property_plant_equipment", "nwc"],
                     "Nonpositive operating capital")
    _missing_reasons(df, "gpa", ["total_revenue", "cost_of_revenue", "total_assets"],
                     "Total assets must be positive")
    _missing_reasons(df, "sta", ["net_income", "operating_cash_flow", "total_assets"],
                     "Total assets must be positive")
    _missing_reasons(
        df, "snoa", ["total_assets", "cash_and_cash_equivalents",
                    "current_debt_and_capital_lease_obligation_used",
                    "long_term_debt_and_capital_lease_obligation", "minority_interest_used",
                    "preferred_stock_used", "common_stock_equity"],
        "Total assets must be positive",
    )
    currency_problem = df["market_equity"].isna() & ~df["valuation_currency_valid"]
    class_problem = df["market_equity"].isna() & df["multiple_traded_share_classes"]
    for metric in ("bm", "eey"):
        df.loc[currency_problem, f"{metric}_reason"] = (
            "Quote currency and financial-statement currency differ or are unverified"
        )
        df.loc[class_problem, f"{metric}_reason"] = (
            "Multiple share classes: class-specific fiscal market capitalization unavailable"
        )
    df["pman_reason"] = ""
    df.loc[df["pman"].isna(), "pman_reason"] = "Undefined Beneish input(s)"
    df.loc[~df["valid_prior_year"], "pman_reason"] = "Missing consecutive annual fiscal period"
    df.loc[df["sector"].fillna("").str.lower().isin(["financial services", "financials"]),
           "pman_reason"] = "Beneish model not applicable to financial companies"
    missing_beneish = df[beneish].isna().any(axis=1) & df["valid_prior_year"]
    missing_nonfinancial = (
        missing_beneish & df["pman"].isna()
        & ~df["sector"].fillna("").str.lower().isin(["financial services", "financials"])
    )
    if missing_nonfinancial.any():
        missing_component_names = (
            df.loc[missing_nonfinancial, beneish]
              .isna().apply(lambda row: ", ".join(row.index[row]), axis=1)
        )
        df.loc[missing_nonfinancial, "pman_reason"] = (
            "Missing Beneish input(s): " + missing_component_names
        )
        tata_prerequisites = [
            "current_assets", "prior_current_assets", "cash_and_cash_equivalents",
            "prior_cash_and_cash_equivalents", "current_liabilities",
            "prior_current_liabilities", "current_debt_used", "prior_current_debt_used",
            "taxes_payable_used", "prior_taxes_payable_used", "depreciation_and_amortization",
        ]
        requires_tata = missing_nonfinancial & df["tata"].isna()
        if requires_tata.any():
            missing_inputs = df.loc[requires_tata, tata_prerequisites].isna().apply(
                lambda row: ", ".join(row.index[row]), axis=1
            )
            df.loc[requires_tata, "pman_reason"] += (
                " (TATA unresolved fields: " + missing_inputs + ")"
            )

    latest = (
        df.sort_values(["ticker", "fiscal_period_end"], ascending=[True, False])
          .drop_duplicates("ticker", keep="first")
          .reset_index(drop=True)
    )
    latest["fiscal_period_end"] = latest["fiscal_period_end"].dt.strftime("%Y-%m-%d")
    latest["prior_fiscal_period_end"] = (
        latest["prior_fiscal_period_end"].dt.strftime("%Y-%m-%d")
    )
    return latest


def identify_beneish_index_exclusions(latest: pd.DataFrame) -> pd.DataFrame:
    """Audit issuers whose Beneish GMI/SGAI cannot be interpreted normally.

    This is a conservative *universe* exclusion requested for the SHORT app,
    not a claim that the underlying financial statements are erroneous.
    Negative gross margins can occur legitimately, but make the GMI ratio
    cross zero; a nonpositive SGAI likewise breaks its normal interpretation.
    Unavailable indices are not treated as zero, and do not trigger removal.
    """
    gmi = pd.to_numeric(latest["gmi"], errors="coerce")
    sgai = pd.to_numeric(latest["sgai"], errors="coerce")
    bad_gmi = gmi.notna() & gmi.le(0)
    bad_sgai = sgai.notna() & sgai.le(0)
    mask = bad_gmi | bad_sgai
    excluded = latest.loc[mask, ["ticker", "company_name", "gmi", "sgai"]].copy()
    excluded["reason"] = [
        ", ".join(label for label, yes in (
            ("Nonpositive Beneish GMI", bool(bad_gmi.loc[index])),
            ("Nonpositive Beneish SGAI", bool(bad_sgai.loc[index])),
        ) if yes)
        for index in excluded.index
    ]
    return excluded.reset_index(drop=True)
