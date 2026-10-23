# ACCT 3343 SHORT Analysis App

An interactive Streamlit financial statement screener for **potentially overvalued
US-listed nonfinancial stocks**, using a one-time, static Yahoo Finance dataset.

**The October 23, 2026 snapshot is already included:** 949 issuers after six
disclosed Beneish-index quality exclusions, sourced from the 1,200-ticker
Colab collection. No additional data collection is needed to launch this app.

## The files

- `build_universe.ipynb`: Google Colab notebook that creates the dataset.
- `build_universe.py` and `metrics.py`: auditable Python source for that notebook.
- `prepare_static_snapshot.py`: reproduce exclusions without repeating Yahoo requests.
- `screening.py`: screening and peer-rank logic, tested independently of the UI.
- `app.py`: publicly accessible Streamlit interface. **Never calls Yahoo Finance.**
- `requirements.txt`: dependencies for the deployed app.
- `data/short_universe.csv`: **included** cleaned universe with 949 unique issuers.
- `data/snapshot.json`: records the date and number of collected/valid firms.
- `data/quality_exclusions.csv`: six tickers with GMI/SGAI and exclusion reasons.
- `data/financial_history.csv`: original annual history, including firms omitted
  from the screen, retained for reproducibility.
- `data/collection_issues.csv`: Yahoo issues plus the six new exclusions.
- `data/field_labels_audit.csv`: diagnostic list of Yahoo debt/lease/tax line
  names encountered for each firm (generated with the other data files; not
  included in this full-run download).

## Step 1: use the included data, or optionally regenerate it

You can **skip this step** and proceed directly to deployment. The static CSV
is already present; the app never collects it when launched or filtered.

If you need a fresh data collection later:

1. Go to https://colab.research.google.com/ and choose **Upload notebook**.
2. Upload `build_universe.ipynb` from this project.
3. In the notebook's final run cell, change `COLLECTION_LIMIT = 1200` to
   `COLLECTION_LIMIT = 15` for a short test; click **Runtime > Run all**.
4. Review the output, `data/collection_issues.csv`, and
   `data/field_labels_audit.csv`. After fixing any
   blocking API issues, restore `COLLECTION_LIMIT = 1200` and run the notebook
   again for the full dataset. Expect a long run for hundreds of companies.
5. Under Colab's Files panel, download `data/short_universe.csv`,
   `data/snapshot.json`, and `data/quality_exclusions.csv`. Save
   `collection_issues.csv` and `field_labels_audit.csv` for the audit.
6. Confirm the output reports several hundred nonfinancial firms and sensible
   nonmissing metric coverage. **Do not publish the 15-company pilot dataset.**

The Colab collector deliberately reports missing inputs without substituting
unsupported values. Certain missing balance-sheet components can be calculated
from verifiable, reconciled identities, with their sources separately labelled in the CSV;
truly absent inputs remain missing. The Beneish PMAN model requires many
two-year accounting fields and may have lower coverage than simpler GPA/STA
ratios. Its model is not applicable to financial-sector firms.

The exporter excludes a firm when its reported Beneish GMI or SGAI is
nonpositive. This conservative screening choice avoids interpreting unusual
indices; **it does not imply that the underlying financial statements are
incorrect**. In this snapshot, the rule removes six of 955 nonfinancial firms.

## Step 2: create your GitHub project

1. Create a new repository in your own GitHub account, such as
   `acct3343-short-analysis`.
2. Upload the *contents* of this project directory (not the enclosing directory)
   so `app.py` and `requirements.txt` appear in the repository root.
3. The included `data` folder already contains the real dataset and snapshot;
   upload it alongside the Python files.
4. No downloads are required when someone opens the app; the
   saved CSV is bundled with the deployed repository.

## Step 3: publish with Streamlit Community Cloud

1. Open https://share.streamlit.io/ and connect your GitHub account.
2. Choose **Create app** and select the repository and branch.
3. Set the entrypoint to `app.py` and deploy.
4. Test the public link in a private/incognito browser window. Change every
   slider and metric selection, inspect missing-data counts, and confirm the
   candidates update without a Yahoo request.
5. Submit **only that working public Streamlit URL** to Brightspace.

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

With the included 949 companies, default B/M → STA → GPA settings retain
**383 → 227 → 146** companies. These counts update with the settings.
PMAN is available for 457 firms; 186 use the labelled broader `Total Tax
Payable` fallback and deserve additional company-specific review.

### Important course-specific conventions

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

## Testing locally

From the project directory, run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
streamlit run app.py
```

The published interface needs `app.py`, `screening.py`, `requirements.txt`,
and the precomputed `data/short_universe.csv` and `data/snapshot.json`.
Keep the other files to document the source and exclusions.

## Validation checklist before submitting

- [ ] Data was collected for a large independent universe, not the class 50.
- [ ] Final data and metric coverage counts are visible and plausible.
- [ ] Spot-check at least three firms against Yahoo's reported annual lines.
- [ ] Confirm the retained companies change when sliders and metrics change.
- [ ] Try raw and sector-relative ranking and inspect peer minimum exclusions.
- [ ] Check invalid denominators and missing fields do not produce fake zeros.
- [ ] Confirm historical fiscal price matches the statement's reporting date.
- [ ] Verify quote and financial-statement currencies match on valued firms.
- [ ] Review identity-derived balance-sheet components and the multi-class log.
- [ ] Inspect the tax/debt label audit and the `*_source` disclosures for PMAN.
- [ ] Make sure the public Streamlit app loads in an unsigned-in browser.
