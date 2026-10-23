## The files

- `build_universe.ipynb`: Google Colab notebook that creates the dataset.
- `build_universe.py` and `metrics.py`: auditable Python source for that notebook.
- `prepare_static_snapshot.py`: reproduce exclusions without repeating Yahoo requests.
- `screening.py`: screening and peer-rank logic, tested independently of the UI.
- `app.py`: publicly accessible Streamlit interface. **Never calls Yahoo Finance.**
- `requirements.txt`: dependencies for the deployed app.
- `data/short_universe.csv`: **included** cleaned universe with 949 unique issuers.
- `data/snapshot.json`: records the actual UTC collection timestamp, a separately
  labelled analysis reference date, and the number of collected/valid firms.
- `data/quality_exclusions.csv`: six tickers with GMI/SGAI and exclusion reasons.
- `data/financial_history.csv`: original annual history, including firms omitted
  from the screen, retained for reproducibility.
- `data/collection_issues.csv`: Yahoo issues plus the six new exclusions.
- `data/field_labels_audit.csv`: diagnostic list of Yahoo debt/lease/tax line
  names encountered for each firm (generated with the other data files; not
  included in this full-run download).

## Screening logic

The app uses three user-configurable, sequential stages:

1. **Valuation vulnerability:** low B/M and/or low enterprise earnings yield.
2. **Earnings-quality concerns:** high STA, SNOA, and/or PMAN.
3. **Weak profitability:** low GPA and/or ROC.

Each stage ranks the firms that entered it, removes firms lacking the selected
required metric(s), optionally demeans scores within sector, and retains the
specified worst-performing percentile. The funnel reports missing exclusions
separately from percentile cutoff removals. The final score is a relative
screening index, **not** a calibrated predicted loss or probability of fraud.
The app also explains each surviving company's selected metrics against
peer medians and shows relative concern percentiles, source dates, and
important data-quality limitations.

The **Candidate analysis** section adds two historical charts built exclusively
from `data/financial_history.csv`: annual revenue and annual net income versus
operating cash flow. They show amounts in the company's financial-statement
currency, preserve missing values, and do not mix in current share prices.
The company-level table compares all seven ratios with original sector peers,
displaying the raw company and peer median values, the signed gap, the number
of observations, and a colour-coded directional SHORT concern assessment.
Balanced observations identify the strongest signal in each screening category
and one evidence-based counterargument when a ratio is better than its peer
median. These are screening observations, not fully researched investment cases.

The main SHORT candidate table shows the **top 10** by our app-created
**Custom SHORT warning score**, displayed as a **Relative SHORT concern ranking
(/100)**, rather
than all surviving firms. The full set remains available for company-by-company
inspection and CSV download. This heuristic is **not from the textbook** and
is not a predicted probability of losses. Every selected metric contributes
equally within its own stage; the ranks within each stage are combined and
converted into a relative rank strength, then the three stage strengths are
averaged with equal one-third weights. Comparisons occur within each stage's
progressively smaller eligible pool, so values depend on the screen settings.

An interactive Plotly scatter plot immediately below the SHORT table shows
**every final-stage survivor**, not just the top ten. The horizontal and
vertical coordinates are the already-calculated valuation and earnings-quality
concern rankings (0–100); point colour shows the profitability concern ranking.
Outlined diamonds highlight the ten highest overall SHORT warning scores.
Hover text includes company identity, sector, category rankings, total custom
score, and the ratios selected in the sidebar. The plot updates when screening
controls change; its rankings are not estimated probabilities of loss. The
chart zoom selector includes intermediate 1.5× and 3× magnifications and focuses
the higher-concern corner, while mouse-wheel/trackpad zoom allows even finer
cursor-centred control. Extra spacing keeps axis labels visible.

The source caption shows **October 17, 2026** as the separately labelled
analysis reference date and **October 23, 2026 (UTC)** as the actual data
collection date, both from `snapshot.json`. It also shows the earliest/latest
fiscal year-ends in the screened static dataset (2025-07-31 to 2026-07-31 in
the included snapshot). The analysis reference date is not the collection date.

The separate **Magic Formula** section at the bottom is the course's BUY
research-ranking method. It uses *all selected-sector companies* with valid
ROC and enterprise earnings yield, not only SHORT survivors. Companies are
ranked independently on ROC (highest = 1) and EEY (highest = 1); each firm's
two ranks are added and the lowest sum appears first. The app shows the top
10, documents missing-input exclusions, and offers a full ranking CSV export.
The Magic Formula is a research aid, not an automatic buy recommendation.

With the included 949 companies, default B/M → STA → GPA settings retain
**383 → 227 → 146** companies. These counts update with the settings.
PMAN is available for 457 firms; 186 use the labelled broader `Total Tax
Payable` fallback and deserve additional company-specific review.

## Important course-specific conventions

- Excess cash = **cash + current assets − current liabilities**, as specified
  by the professor. This can differ from conventional definitions.
- STA = `(net_income − operating_cash_flow) / total_assets`.
- Beneish TATA is the *balance-sheet changes* model in the Chapter 3 notebook,
  rather than the simpler STA formula.
- The standard eight-factor Beneish coefficients feed PROBM; PMAN is the
  **standard normal CDF** of that score, not a logistic transformation.
- Nonpositive valuation denominators and missing accounting fields are not
  silently replaced with zero. Book value below zero is not ranked as merely
  "extremely expensive" by B/M.
- The collector uses annual share counts and historical prices, not current
  market cap mixed with an older balance sheet. If annual share counts don't
  match the exact fiscal date, that limitation remains and is disclosed.
- Missing minority interest may be derived from reported total equity or an
  independently checked assets-minus-liabilities equity identity. Preferred
  stock can be reconstructed as stockholders' equity minus common equity when
  the stated balance-sheet categories support the difference. Short-term debt
  and leases may be derived from total debt minus long-term debt and leases
  **only if a reported debt split reconciles independently for that issuer,
  without conflicting reported years**. The collector records the source.
- For Beneish TATA, the collector checks direct `Current Debt` and reported
  `Income Tax Payable` / `Taxes Payable` labels, including harmless differences
  in punctuation, spacing, and capitalization. When needed, current debt may
  equal directly reported current debt-plus-lease obligations minus an
  independently reported current lease obligation. A current lease obligation
  itself can sometimes be reconciled from total and long-term lease obligations.
- If no narrower tax-payable row is available, reported `Total Tax Payable` is
  available as a **clearly labelled aggregate fallback**. It may include a
  broader category of taxes, so individual PMAN values using this fallback
  warrant review. The raw Yahoo labels are kept in the history/audit files.
- A *preferred shares number* is **never** substituted as a dollar-denominated
  preferred-stock value; missing debt or taxes payable are **not** presumed zero.
- Valuation is unavailable when the Yahoo quote currency does not match the
  statement currency, including foreign-currency ADR reporting, or when a
  listed company has multiple traded share classes without class-level share
  counts. Only one row per issuer is retained, to avoid counting duplicate
  share classes as separate companies.

The app's purpose is to identify leads for additional valuation research, not
to assert that screened firms have manipulated earnings or should be sold short.
