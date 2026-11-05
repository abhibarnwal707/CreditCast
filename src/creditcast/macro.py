"""Phase 2a: FRED macro series (quarterly) and the Fed 2026 supervisory scenarios.

Run:  python -m creditcast.macro
Outputs: data/external/fred_quarterly.csv, data/external/fed_2026_scenarios.csv
"""
import os
import re

import pandas as pd
import pdfplumber
from dotenv import load_dotenv

from .config import ROOT, load_config

EXT = ROOT / "data" / "external"
FED_PDF = ROOT / "docs" / "reference" / "2026-final-supervisory-stress-test-scenarios-20260204.pdf"

# FRED series -> how to aggregate to quarters (user prompt: quarter average for rates, end-of-quarter for index levels)
FRED_AGG = {
    "UNRATE": "mean",         # unemployment rate, monthly
    "MORTGAGE30US": "mean",   # 30y fixed mortgage rate, weekly
    "GS10": "mean",           # 10y Treasury, monthly
    "USSTHPI": "last",        # FHFA all-transactions HPI, quarterly index level
    "GDPC1": "last",          # real GDP, quarterly level
}

FED_COLS = ["real_gdp_growth", "nominal_gdp_growth", "real_dpi_growth", "nominal_dpi_growth", "unemployment",
            "cpi_inflation", "tbill_3m", "treasury_5y", "treasury_10y", "bbb_yield", "mortgage_rate", "prime_rate",
            "dow_jones", "house_price_index", "cre_price_index", "vix"]


def fetch_fred() -> pd.DataFrame:
    from fredapi import Fred

    load_dotenv(ROOT / ".env")
    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    cols = {}
    for sid, how in FRED_AGG.items():
        s = fred.get_series(sid).dropna()
        s.index = pd.DatetimeIndex(s.index)
        q = s.resample("QE").agg(how).dropna()
        q.index = q.index.to_period("Q")
        cols[sid] = q
        print(f"[fred] {sid:<13} {how:<5} {q.index.min()} .. {q.index.max()}  ({len(q)} quarters)")
    df = pd.DataFrame(cols)
    df.index.name = "quarter"
    return df


def parse_fed_scenarios() -> pd.DataFrame:
    """Extract Table 3.A (baseline) and 4.A (severely adverse), domestic variables, from the Fed PDF.

    pdfplumber sees only the header block as a table, so the data rows are parsed from the page text:
    'Q1 2026 1.9 4.8 ... 23.0' = date + 16 numbers (column order checked against the table header cells).
    """
    row_re = re.compile(r"^(Q[1-4]) (20\d\d) ((?:-?[\d,]+\.?\d* ?){16})$")
    rows = []
    with pdfplumber.open(FED_PDF) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for tag, scen in (("Table 3.A.", "baseline"), ("Table 4.A.", "severely_adverse")):
                if tag not in text:
                    continue
                body = text.split(tag, 1)[1]
                body = body.split("Table 3.B.", 1)[0].split("Table 4.B.", 1)[0]  # stop at the international table
                for line in body.splitlines():
                    m = row_re.match(line.strip())
                    if m:
                        vals = [float(x.replace(",", "")) for x in m.group(3).split()]
                        rows.append({"scenario": scen, "quarter": f"{m.group(2)}{m.group(1)}", **dict(zip(FED_COLS, vals))})
    df = pd.DataFrame(rows)
    assert len(df) == 26, f"expected 2 x 13 quarters, got {len(df)}"
    return df


def main() -> None:
    EXT.mkdir(parents=True, exist_ok=True)
    fred = fetch_fred()
    fred.to_csv(EXT / "fred_quarterly.csv")
    fed = parse_fed_scenarios()
    fed.to_csv(EXT / "fed_2026_scenarios.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    show = ["quarter", "real_gdp_growth", "unemployment", "treasury_10y", "mortgage_rate", "house_price_index"]
    for scen, g in fed.groupby("scenario"):
        print(f"\n=== Fed 2026 {scen} (from the PDF, verify by eye) ===")
        print(g[show].to_string(index=False))


if __name__ == "__main__":
    main()
