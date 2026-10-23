"""Mechanical conversion of the validated Python collector into a Colab notebook."""

from pathlib import Path
import json

BASE = Path(__file__).resolve().parent


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def make():
    metric_source = (BASE / "metrics.py").read_text(encoding="utf-8")
    collector = (BASE / "build_universe.py").read_text(encoding="utf-8")
    collector = collector.replace(
        "from metrics import calculate_metrics, identify_beneish_index_exclusions\n", ""
    )
    collector = collector.replace('if __name__ == "__main__":\n    main()\n', "")
    cells = [
        markdown("""# ACCT 3343 SHORT | Static Yahoo Finance dataset

This notebook downloads US-listed issuers, computes the course's
financial metrics, and exports **static CSV files** used by the Streamlit app.

**How to use:**

1. Run the install cell.
2. Run the two definitions cells (they only define functions).
3. For a pilot, edit `COLLECTION_LIMIT = 1200` to `15` in the collection cell and
   run the final cell. Check data coverage, the issue log, and
   `data/field_labels_audit.csv` for the exact debt/tax rows Yahoo supplied.
4. Restore `COLLECTION_LIMIT = 1200` and rerun the final cell for the full assignment.
   Collection can take considerable time. Keep Colab active.
5. Download `data/short_universe.csv` and `data/snapshot.json` from Colab's
   Files sidebar and place them in the GitHub application's `data/` directory.

The class's 50-company demo dataset is **never used**. Exact synonyms and
formatting variants are checked for Yahoo's annual debt/tax rows. Certain
missing amounts can be recovered through independently validated identities;
unresolved amounts remain missing. `Total Tax Payable` is separately flagged
when used as a broader alternative to `Income Tax Payable`. The app itself
makes no Yahoo Finance requests.

For the SHORT app, firms whose calculated Beneish GMI or SGAI is nonpositive
are excluded from the exported screening universe and listed separately in
`data/quality_exclusions.csv`. The full annual history remains available for audit.
"""),
        code("!pip install -q yfinance pandas numpy\n"),
        markdown("## Financial calculations (professor-specific formulas)\n"),
        code(metric_source),
        markdown("## Yahoo Finance screener, historical statements, and price matching\n"),
        code(collector),
        markdown("## Run the collection and save the dataset\n\n"
                 "For testing, set `COLLECTION_LIMIT = 15` here. Restore 1200 for the final assignment.\n"),
        code("COLLECTION_LIMIT = 1200\nmain()\n"),
        markdown("## Download and review results\n\n"
                 "In Colab, expand the Files panel. Under `data/`, download: "
                 "`short_universe.csv`, `snapshot.json`, and optionally "
                 "`collection_issues.csv`, `quality_exclusions.csv`, "
                 "`financial_history.csv`, and "
                 "`field_labels_audit.csv`. The audit lists available Yahoo "
                 "debt/tax labels for each ticker to help diagnose PMAN gaps. "
                 "Check that hundreds of independent companies have valid screening metrics.\n"),
    ]
    output = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"},
                     "colab": {"provenance": []}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (BASE / "build_universe.ipynb").write_text(
        json.dumps(output, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    make()
