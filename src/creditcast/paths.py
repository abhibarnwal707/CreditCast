"""Macro paths for the ECL engine: realised history + Fed scenario (13 quarters) + mean reversion afterwards.

Variables: unemployment (%), house_price_index (FHFA USSTHPI level, chained with the Fed scenario HPI growth),
mortgage_rate (%), treasury_10y (%).

Run:  python -m creditcast.paths  -> data/external/macro_paths.csv
"""
import numpy as np
import pandas as pd

from .config import ROOT, load_config

FED_TO_FHFA = {"unemployment": "UNRATE", "mortgage_rate": "MORTGAGE30US", "treasury_10y": "GS10"}


def load_history(cfg: dict) -> pd.DataFrame:
    f = pd.read_csv(ROOT / "data" / "external" / "fred_quarterly.csv", index_col="quarter")
    f.index = pd.PeriodIndex(f.index, freq="Q")
    return f


def long_run(cfg: dict, hist: pd.DataFrame) -> dict:
    m = cfg["macro"]
    h = hist.loc[m["long_run_start"]: m["history_end"]]
    return {
        "unemployment": float(h["UNRATE"].mean()),
        "mortgage_rate": float(h["MORTGAGE30US"].mean()),
        "treasury_10y": float(h["GS10"].mean()),
        "hpi_growth_q": float(np.log(h["USSTHPI"]).diff().mean()),   # mean log growth per quarter
    }


def scenario_path(scen: pd.DataFrame, hist: pd.DataFrame, cfg: dict, n_total: int | None = None) -> pd.DataFrame:
    """scen: 13 Fed rows for one scenario (index = quarter). Returns n_total quarters starting at the first scenario quarter."""
    n_total = n_total or cfg["scenario"]["max_loan_quarters"]
    n_rev = cfg["macro"]["reversion_quarters"]
    lr = long_run(cfg, hist)
    jump = pd.Period(cfg["scenario"]["jump_off"], freq="Q")
    hpi0 = float(hist.loc[jump, "USSTHPI"])
    fed_hpi = scen["house_price_index"].values
    # Q4-2025 Fed HPI level is not published in the tables; the first baseline level (Q1 2026) is the proxy (docs/assumptions.md)
    fed_jump = float(load_fed_baseline_first_hpi())
    rows = []
    quarters = pd.period_range(scen.index[0], periods=n_total, freq="Q")
    last = {k: float(scen[k].iloc[-1]) for k in FED_TO_FHFA}
    hpi_prev = hpi0
    for i, q in enumerate(quarters):
        if i < len(scen):
            r = {k: float(scen[k].iloc[i]) for k in FED_TO_FHFA}
            level = hpi0 * fed_hpi[i] / fed_jump
            src = "fed"
        else:
            w = min((i - len(scen) + 1) / n_rev, 1.0)    # linear reversion weight, 1 = fully at long-run mean
            r = {k: last[k] + w * (lr[k] - last[k]) for k in FED_TO_FHFA}
            g_last = np.log(fed_hpi[-1] / fed_hpi[-2])
            g = g_last + w * (lr["hpi_growth_q"] - g_last)
            level = hpi_prev * np.exp(g)
            src = "reversion"
        hpi_prev = level
        rows.append({"quarter": str(q), "source": src, **r, "hpi": level})
    df = pd.DataFrame(rows)
    df["hpi_yoy"] = df["hpi"].pct_change(4)
    # first 4 quarters of yoy need realised history
    full = pd.concat([hist["USSTHPI"].loc[:jump].rename("hpi").to_frame().assign(quarter=lambda d: d.index.astype(str))[["quarter", "hpi"]], df[["quarter", "hpi"]]])
    full["hpi_yoy"] = full["hpi"].pct_change(4)
    df["hpi_yoy"] = df["quarter"].map(full.set_index("quarter")["hpi_yoy"])
    df["unemployment_chg"] = df["unemployment"].diff()
    df.loc[0, "unemployment_chg"] = df.loc[0, "unemployment"] - float(hist.loc[jump, "UNRATE"])
    return df


def load_fed_baseline_first_hpi() -> float:
    fed = pd.read_csv(ROOT / "data" / "external" / "fed_2026_scenarios.csv")
    return fed[(fed["scenario"] == "baseline")].sort_values("quarter")["house_price_index"].iloc[0]


def all_paths(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    hist = load_history(cfg)
    fed = pd.read_csv(ROOT / "data" / "external" / "fed_2026_scenarios.csv")
    out = []
    for s, g in fed.groupby("scenario"):
        g = g.sort_values("quarter").set_index("quarter")
        out.append(scenario_path(g, hist, cfg).assign(scenario=s))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    df = all_paths()
    df.to_csv(ROOT / "data" / "external" / "macro_paths.csv", index=False)
    print(df.groupby("scenario").head(3).to_string())
    print(df[df["quarter"].isin(["2029Q1", "2029Q2", "2030Q1", "2040Q4"])].to_string())
