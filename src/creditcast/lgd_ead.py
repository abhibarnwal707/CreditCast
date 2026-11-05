"""Phase 5: LGD and EAD.

LGD per default event = P(liquidation | default) x loss severity.
 * P(liquidation | default) is a logistic function of mark-to-market LTV at the start of the default quarter.
 * Severity is linear in the house-price change since origination (HPI at start of the default quarter / HPI at origination - 1):
   net severity is NOT monotone in MTM LTV in this data (mortgage-insured high-LTV loans recover through MI), but it falls
   significantly when house prices have risen and rises when they fall.
 * Severity data: Freddie Mac 'Actual Loss' / zero-balance removal UPB for defaults that completed liquidation (ZB codes 02,03,09,15).
 * Liquidation probability: defaults that reached 90+ days but later prepaid (01), were sold as reperforming (16) or had a
   pre-event defect (96) lose nothing; defaults still open at the data cutoff are excluded (censored).
EAD = scheduled (amortised) balance at the start of the quarter from the loan's current balance, rate and remaining term.

Run:  python -m creditcast.lgd_ead  -> results/lgd.json, results/figures/lgd_by_mtm_ltv.png
"""
import json

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import ROOT, load_config

LGD_FLOOR, LGD_CAP = 0.0, 1.0   # severity clipped to [0, 1] for modelling (gains/over-recoveries floored, >100% capped)
LIQ_CODES = ("02", "03", "09", "15")


# ---------------------------------------------------------------- EAD
def amortisation_balances(upb: float, annual_rate_pct: float, rem_months: int, n_quarters: int) -> np.ndarray:
    """Scheduled balance at the START of each quarter k = 0..n_quarters-1 (k=0 -> current UPB). No prepayment, level-payment loan."""
    i = annual_rate_pct / 1200.0
    n = int(rem_months)
    out = np.zeros(n_quarters)
    if upb <= 0 or n <= 0:
        return out
    pay = upb / n if i == 0 else upb * i / (1 - (1 + i) ** -n)
    for k in range(n_quarters):
        m = 3 * k
        if m >= n:
            break
        if i == 0:
            bal = upb - pay * m
        else:
            bal = upb * (1 + i) ** m - pay * (((1 + i) ** m - 1) / i)
        out[k] = max(bal, 0.0)
    return out


# ---------------------------------------------------------------- LGD
def lgd_function(model: dict, mtm_ltv, hpi_chg):
    """Cure-adjusted LGD = P(liquidation | MTM LTV) x severity(HPI change since origination). Returns (lgd, p_liq, severity)."""
    x = np.asarray(mtm_ltv, dtype=float) / 10.0
    h = np.asarray(hpi_chg, dtype=float) * 10.0          # per 10 pp
    p_liq = 1 / (1 + np.exp(-(model["liq_intercept"] + model["liq_slope"] * x)))
    sev = np.clip(model["sev_intercept"] + model["sev_slope"] * h, model["sev_min"], LGD_CAP)
    return p_liq * sev, p_liq, sev


def fit_lgd(d: pd.DataFrame) -> dict:
    """d columns: mtm_ltv (at default), liquidated (0/1, resolved defaults only), severity (liquidated only)."""
    X = sm.add_constant(d["mtm_ltv"] / 10.0)
    liq = sm.GLM(d["liquidated"], X, family=sm.families.Binomial()).fit()
    s = d[d["liquidated"] == 1]
    ols = sm.OLS(s["severity"].clip(LGD_FLOOR, LGD_CAP), sm.add_constant(s["hpi_chg_orig"] * 10.0)).fit()
    return {"liq_intercept": float(liq.params.iloc[0]), "liq_slope": float(liq.params.iloc[1]),
            "sev_intercept": float(ols.params.iloc[0]), "sev_slope": float(ols.params.iloc[1]),
            "sev_min": float(max(0.05, s["severity"].clip(LGD_FLOOR, LGD_CAP).quantile(0.01)))}


def main() -> None:
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    feat = pd.read_parquet(ROOT / "data" / "interim" / "features.parquet", columns=["loan_id", "qidx", "default", "mtm_ltv", "hpi_chg_orig", "start_upb", "vintage", "split"])
    dflt = feat[feat["default"] == 1].drop(columns="default")
    losses = pd.read_parquet(ROOT / "data" / "interim" / "losses.parquet")
    con = duckdb.connect(str(ROOT / "data" / "interim" / "creditcast.duckdb"), read_only=True)
    last = con.execute("SELECT loan_id, arg_max(zb_code, period) AS last_zb FROM perf GROUP BY loan_id").df()
    d = dflt.merge(last, on="loan_id", how="left").merge(
        losses[["loan_id", "zb_removal_upb", "actual_loss"]], on="loan_id", how="left")
    n_def = len(d)
    d["status"] = np.select([d["last_zb"].isin(LIQ_CODES), d["last_zb"].isin(["01", "16", "96"])], ["liquidated", "no_loss_exit"], default="open")
    print(f"[rows] default events {n_def:,}: " + ", ".join(f"{k} {v:,}" for k, v in d["status"].value_counts().items()))

    usable = d[(d["status"] == "liquidated") & d["actual_loss"].notna() & (d["zb_removal_upb"] > 0) & d["mtm_ltv"].notna() & d["hpi_chg_orig"].notna()].copy()
    usable["severity"] = usable["actual_loss"] / usable["zb_removal_upb"]
    print(f"[rows] liquidated defaults with a usable LGD: {len(usable):,} (liquidated total {int((d['status']=='liquidated').sum()):,})")

    resolved = d[d["status"] != "open"].copy()
    resolved["liquidated"] = (resolved["status"] == "liquidated").astype(int)
    resolved = resolved.merge(usable[["loan_id", "severity"]], on="loan_id", how="left")
    # liquidated loans with no usable severity (recent disposals) are dropped from the liquidation-probability sample too
    resolved = resolved[(resolved["status"] != "liquidated") | resolved["severity"].notna()].dropna(subset=["mtm_ltv", "hpi_chg_orig"])
    model = fit_lgd(resolved)
    print("[lgd] model:", {k: round(v, 4) for k, v in model.items()})
    assert model["sev_slope"] < 0 and model["liq_slope"] > 0, "LGD must rise when house prices fall (higher MTM LTV, lower HPI change)"

    # bootstrap uncertainty (small sample)
    boots = []
    for _ in range(300):
        b = resolved.sample(len(resolved), replace=True, random_state=int(rng.integers(1e9)))
        try:
            m = fit_lgd(b)
        except Exception:
            continue
        pts = [lgd_function(m, x, h)[0] for x, h in ((60.0, 0.4), (100.0, -0.1), (130.0, -0.3))]
        boots.append([m["sev_slope"], m["liq_slope"], *pts])
    boots = np.array(boots)
    ci = lambda i: [float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))]

    bins = [0, 60, 80, 100, 120, 1000]
    labels = ["<60", "60-80", "80-100", "100-120", ">120"]
    resolved["bucket"] = pd.cut(resolved["mtm_ltv"], bins, labels=labels, right=False)
    rows = []
    for b, g in resolved.groupby("bucket", observed=True):
        liq = g[g["liquidated"] == 1]
        pred = lgd_function(model, g["mtm_ltv"].values, g["hpi_chg_orig"].values)[0].mean()
        rows.append({"mtm_ltv_bucket": str(b), "defaults_resolved": int(len(g)), "liquidated": int(len(liq)),
                     "liquidation_rate": float(len(liq) / len(g)), "mean_severity": float(liq["severity"].clip(0, 1).mean()) if len(liq) else None,
                     "cure_adjusted_lgd_actual": float((liq["severity"].clip(0, 1) * 1.0).sum() / len(g)) if len(g) else None,
                     "cure_adjusted_lgd_model": float(pred)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    out = {
        "default_events": int(n_def),
        "default_status": {k: int(v) for k, v in d["status"].value_counts().items()},
        "defaults_with_usable_lgd": int(len(usable)),
        "mean_severity_unclipped": float(usable["severity"].mean()),
        "mean_lgd": float(usable["severity"].clip(LGD_FLOOR, LGD_CAP).mean()),
        "mean_lgd_definition": "mean of Actual Loss / zero-balance removal UPB over liquidated defaults, clipped to [0,1]",
        "median_severity": float(usable["severity"].median()),
        "share_severity_gt_1": float((usable["severity"] > 1).mean()),
        "share_severity_lt_0": float((usable["severity"] < 0).mean()),
        "cure_adjusted_mean_lgd_per_default_event": float(lgd_function(model, resolved["mtm_ltv"].values, resolved["hpi_chg_orig"].values)[0].mean()),
        "liquidation_probability_given_default": float(resolved["liquidated"].mean()),
        "model": model,
        "model_uncertainty_95ci": {"severity_slope_per_10pp_hpi_change": ci(0), "liq_slope_per_10pts_mtm_ltv": ci(1),
                                   "lgd_mtm60_hpi+40pct": ci(2), "lgd_mtm100_hpi-10pct": ci(3), "lgd_mtm130_hpi-30pct": ci(4)},
        "severity_r2_note": "severity regression has low R-squared (about 2%): HPI change is statistically significant but explains little of the dispersion",
        "lgd_by_scenario_point": {"mtm60_hpi+40pct": float(lgd_function(model, 60.0, 0.4)[0]), "mtm100_hpi-10pct": float(lgd_function(model, 100.0, -0.1)[0]),
                                  "mtm130_hpi-30pct": float(lgd_function(model, 130.0, -0.3)[0])},
        "by_mtm_ltv_bucket": rows,
        "notes": ["Defaults still open at the data cutoff are excluded (censored); this probably understates liquidation probability.",
                  "Exits 01 (prepaid), 16 (reperforming sale) and 96 (defect) are treated as zero loss.",
                  "Severity above 100% (legal costs, accrued interest) is capped at 100%; gains floored at 0%."],
    }
    (ROOT / "results" / "lgd.json").write_text(json.dumps(out, indent=2))
    print(f"[lgd] mean severity (clipped) {out['mean_lgd']:.3f}; cure-adjusted LGD per default event {out['cure_adjusted_mean_lgd_per_default_event']:.3f}; "
          f"P(liq|default) {out['liquidation_probability_given_default']:.3f}")

    h = np.linspace(0.6, -0.4, 100)                       # HPI change since origination: +60% ... -40%
    x = 100 * 0.8 / (1 + h)                               # MTM LTV of an 80% LTV loan with no amortisation
    lg = lgd_function(model, x, h)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h * 100, lg[0], label="cure-adjusted LGD", color="#1f4e79")
    ax.plot(h * 100, lg[2], "--", label="severity | liquidated", color="#c0504d")
    ax.set_xlabel("house-price change since origination (%), 80% LTV loan")
    ax.set_ylabel("LGD")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "lgd_by_mtm_ltv.png", dpi=150)


if __name__ == "__main__":
    main()
