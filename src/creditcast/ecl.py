"""Phase 6: CECL-style lifetime expected credit loss engine.

For every loan alive at the jump-off (end of 2025Q4) project quarter by quarter to maturity (cap 120 quarters):
    macro path   = Fed scenario for 13 quarters, then linear reversion to the long-run mean (paths.py)
    h_d(t)       = default hazard (champion logistic GLM, pd_model.py)
    h_p(t)       = prepayment hazard (second logistic hazard, fitted here)
    S(t)         = prod_{s<=t} (1 - h_d(s) - h_p(s))
    ECL          = sum_t  S(t-1) * h_d(t) * LGD(t) * EAD(t)
EAD(t) = scheduled start-of-quarter balance; LGD(t) = cure-adjusted LGD at MTM LTV / HPI change of quarter t (lgd_ead.py).
Features for projected quarter t use macro dated t-1 (same convention as estimation, no look-ahead). ECL is undiscounted.

Run:  python -m creditcast.ecl -> results/ecl.json, results/prepay_model.json
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import ROOT, assign_split, load_config, q_to_idx
from .features import AGE_CENTER, macro_by_qidx
from .lgd_ead import lgd_function
from .paths import all_paths
from .pd_model import SCALE, predict_glm

PREPAY_FEATURES = ["rate_incentive", "age_yrs", "age_yrs_sq", "mtm_ltv", "credit_score", "hpi_yoy"]
PREPAY_SCALE = {"rate_incentive": 1.0, "age_yrs": 1.0, "age_yrs_sq": 1.0, "mtm_ltv": 10.0, "credit_score": 100.0, "hpi_yoy": 0.1}


# ------------------------------------------------------------------ prepayment hazard
def fit_prepay(cfg: dict, max_qidx: int | None = None) -> dict:
    cols = ["qidx", "prepay", "default"] + PREPAY_FEATURES
    f = pd.read_parquet(ROOT / "data" / "interim" / "features.parquet", columns=cols)
    f["split"] = assign_split(f["qidx"], cfg).values
    f = f[f["split"] != "covid"]           # projection model: all non-COVID quarters incl. the recent high-rate regime
    if max_qidx is not None:               # back-test: fit only on data available at the back-test origin
        f = f[f["qidx"] <= max_qidx]
    X = sm.add_constant(pd.DataFrame({k: f[k] / PREPAY_SCALE[k] for k in PREPAY_FEATURES}))
    res = sm.GLM(f["prepay"].astype(float), X, family=sm.families.Binomial()).fit()
    out = {"features": PREPAY_FEATURES, "scale": PREPAY_SCALE, "params": {k: float(v) for k, v in res.params.items()},
           "p_values": {k: float(v) for k, v in res.pvalues.items()}, "n": int(len(f)), "prepay_events": int(f["prepay"].sum()),
           "fit_period": "all non-COVID quarters (projection model, not part of the OOT PD validation)"}
    from sklearn.metrics import roc_auc_score
    out["auc_in_sample"] = float(roc_auc_score(f["prepay"], res.predict(X)))
    return out


# ------------------------------------------------------------------ portfolio + projection
def load_portfolio(cfg: dict, jump: int | None = None) -> pd.DataFrame:
    jump = q_to_idx(cfg["scenario"]["jump_off"]) if jump is None else jump
    f = pd.read_parquet(ROOT / "data" / "interim" / "features.parquet")
    p = f[f["qidx"] == jump + 1].copy()
    p = p[(p["start_rem_months"] > 0) & (p["start_upb"] > 0)].reset_index(drop=True)
    p["orig_value"] = p["orig_upb"] / (p["ltv"] / 100.0)
    p["hpi_orig"] = macro_by_qidx()["hpi"].reindex(p["orig_qidx"].values).values
    return p


def amortisation_matrix_long(upb, rate_pct, rem, n_q):
    """Vectorised scheduled start-of-quarter balances for loans repeated n_q[i] times. Returns (loan_index, t, balance)."""
    li = np.repeat(np.arange(len(upb)), n_q)
    start = np.cumsum(n_q) - n_q
    t = np.arange(len(li)) - np.repeat(start, n_q) + 1          # 1..n_q
    m = 3 * (t - 1)
    i = rate_pct[li] / 1200.0
    n = rem[li].astype(float)
    pay = np.where(i > 0, upb[li] * i / (1 - (1 + i) ** -n), upb[li] / n)
    grow = (1 + i) ** m
    bal = np.where(i > 0, upb[li] * grow - pay * ((grow - 1) / np.where(i > 0, i, 1)), upb[li] - pay * m)
    bal = np.where(m >= n, 0.0, np.maximum(bal, 0.0))
    return li, t, bal


def prepare_long(port: pd.DataFrame, max_q: int) -> dict:
    n_q = np.minimum(np.ceil(port["start_rem_months"].values / 3).astype(int), max_q)
    li, t, ead = amortisation_matrix_long(port["start_upb"].values.astype(float), port["start_rate"].values.astype(float),
                                          port["start_rem_months"].values.astype(float), n_q)
    return {"li": li, "t": t, "ead": ead, "n_q": n_q}


def macro_lookup(path: pd.DataFrame, jump_qidx: int, max_t: int) -> dict:
    """Arrays indexed by projection quarter t=1..max_t giving macro values dated t-1 (+ lags for 4q changes)."""
    hist = macro_by_qidx()
    hist = hist.loc[:jump_qidx]
    p = path.copy()
    p["qidx"] = jump_qidx + 1 + np.arange(len(p))
    full = pd.concat([hist[["unemp", "hpi", "mort"]],
                      p.set_index("qidx")[["unemployment", "hpi", "mortgage_rate"]].rename(columns={"unemployment": "unemp", "mortgage_rate": "mort"})])
    t = np.arange(1, max_t + 1)
    q = jump_qidx + t
    g = lambda s, lag: s.reindex(q - lag).values
    return {"unemp_lag": g(full["unemp"], 1), "unemp_chg_4q": g(full["unemp"], 1) - g(full["unemp"], 5),
            "hpi_yoy": g(full["hpi"], 1) / g(full["hpi"], 5) - 1, "mort_lag": g(full["mort"], 1), "hpi_lag": g(full["hpi"], 1)}


def project(port: pd.DataFrame, L: dict, path: pd.DataFrame, pd_params: dict, prepay: dict, lgd_model: dict, cfg: dict,
            jump: int | None = None) -> dict:
    jump = q_to_idx(cfg["scenario"]["jump_off"]) if jump is None else jump
    li, t, ead = L["li"], L["t"], L["ead"]
    M = macro_lookup(path, jump, int(t.max()))
    mt = lambda k: M[k][t - 1]
    age = (port["start_age"].values[li] + 3 * (t - 1)) / 12.0
    hpi_ratio = mt("hpi_lag") / port["hpi_orig"].values[li]
    mtm = 100.0 * ead / (port["orig_value"].values[li] * hpi_ratio)
    rate_inc = port["orig_rate"].values[li] - mt("mort_lag")
    df = pd.DataFrame({
        "credit_score": port["credit_score"].values[li], "dti": port["dti"].values[li], "orig_rate": port["orig_rate"].values[li],
        "age_yrs": age, "age_yrs_sq": (age - AGE_CENTER) ** 2, "mtm_ltv": mtm, "rate_incentive": rate_inc,
        "unemp_lag": mt("unemp_lag"), "unemp_chg_4q": mt("unemp_chg_4q"), "hpi_yoy": mt("hpi_yoy"),
        "purpose_cashout": port["purpose_cashout"].values[li], "purpose_refi_nocash": port["purpose_refi_nocash"].values[li],
        "occ_investor": port["occ_investor"].values[li], "occ_second": port["occ_second"].values[li]})
    hd = predict_glm(pd_params, df)
    z = prepay["params"]["const"] + sum(prepay["params"][k] * df[k].values / PREPAY_SCALE[k] for k in PREPAY_FEATURES)
    hp = 1 / (1 + np.exp(-z))
    tot = np.clip(hd + hp, 0, 0.999)                       # keep S >= 0
    lgd, _, _ = lgd_function(lgd_model, mtm, hpi_ratio - 1)
    # survival: S(t-1) = prod_{s<t}(1 - h_d - h_p) within each loan
    lg = np.log1p(-tot)
    cs = pd.Series(lg).groupby(li).cumsum().values
    s_prev = np.exp(cs - lg)
    s_t = np.exp(cs)
    loss = s_prev * hd * lgd * ead
    return {"li": li, "t": t, "hd": hd, "hp": hp, "lgd": lgd, "ead": ead, "s_prev": s_prev, "s": s_t, "ecl": loss,
            "default_prob": s_prev * hd}


def summarise(port: pd.DataFrame, r: dict, horizon: int | None = None) -> dict:
    li, t = r["li"], r["t"]
    upb0 = port["start_upb"].values
    m = np.ones(len(t), bool) if horizon is None else t <= horizon
    w = upb0[li]
    out = {"ecl_dollars": float(r["ecl"][m].sum()), "starting_upb": float(upb0.sum())}
    out["ecl_pct_of_upb"] = out["ecl_dollars"] / out["starting_upb"]
    seg = port["segment"].values[li]
    out["by_segment"] = {}
    for s in np.unique(seg):
        sm_ = m & (seg == s)
        u = upb0[port["segment"].values == s].sum()
        out["by_segment"][int(s)] = {"loans": int((port["segment"] == s).sum()), "starting_upb": float(u),
                                     "ecl_dollars": float(r["ecl"][sm_].sum()), "ecl_pct_of_upb": float(r["ecl"][sm_].sum() / u)}
    return out


def quarterly_profile(port: pd.DataFrame, r: dict, max_t: int = 40) -> pd.DataFrame:
    """UPB-weighted portfolio quarterly default probability, cumulative default probability, loss rate and survival."""
    li, t = r["li"], r["t"]
    upb0 = port["start_upb"].values
    w = upb0[li]
    tot = upb0.sum()
    d = pd.DataFrame({"t": t, "dp": r["default_prob"] * w, "loss": r["ecl"], "alive": r["s_prev"] * w, "ead_alive": r["s_prev"] * r["ead"]})
    g = d.groupby("t").sum().reindex(range(1, max_t + 1)).fillna(0.0)
    g["quarterly_default_rate"] = g["dp"] / tot
    g["cum_default_rate"] = g["quarterly_default_rate"].cumsum()
    g["quarterly_loss_rate"] = g["loss"] / tot
    g["cum_loss_rate"] = g["quarterly_loss_rate"].cumsum()
    g["survival_upb_weighted"] = g["alive"] / tot
    return g.reset_index()


def main() -> None:
    cfg = load_config()
    prepay = fit_prepay(cfg)
    print("[prepay] AUC (in-sample) %.3f, events %d" % (prepay["auc_in_sample"], prepay["prepay_events"]))
    print("[prepay] coefs:", {k: round(v, 3) for k, v in prepay["params"].items()})
    (ROOT / "results" / "prepay_model.json").write_text(json.dumps(prepay, indent=2))
    pd_params = json.loads((ROOT / "results" / "pd_glm_coefficients.json").read_text())["params"]
    lgd_model = json.loads((ROOT / "results" / "lgd.json").read_text())["model"]
    port = load_portfolio(cfg)
    print(f"[rows] loans alive at jump-off {cfg['scenario']['jump_off']}: {len(port):,}; starting UPB ${port['start_upb'].sum():,.0f}")
    L = prepare_long(port, cfg["scenario"]["max_loan_quarters"])
    print(f"[rows] projected loan-quarters: {len(L['t']):,}")
    paths = all_paths(cfg)
    out = {"jump_off": cfg["scenario"]["jump_off"], "loans": int(len(port)), "starting_upb": float(port["start_upb"].sum()), "scenarios": {}}
    for scen in ("baseline", "severely_adverse"):
        r = project(port, L, paths[paths["scenario"] == scen].reset_index(drop=True), pd_params, prepay, lgd_model, cfg)
        s = summarise(port, r)
        prof = quarterly_profile(port, r, 40)
        s["lifetime_cumulative_default_rate_upb_weighted"] = float(r["default_prob"][:].dot(port["start_upb"].values[r["li"]]) / port["start_upb"].sum())
        s["mean_lgd_exposure_weighted"] = float((r["lgd"] * r["default_prob"] * r["ead"]).sum() / (r["default_prob"] * r["ead"]).sum())
        s["pd_curve_cum_default_rate"] = {int(a): float(b) for a, b in zip(prof["t"], prof["cum_default_rate"])}
        s["pd_12m_upb_weighted"] = float(prof["cum_default_rate"].iloc[3])
        out["scenarios"][scen] = s
        print(f"[ecl] {scen:<17} lifetime ECL {s['ecl_pct_of_upb']:.3%} of UPB (${s['ecl_dollars']:,.0f}); lifetime CDR {s['lifetime_cumulative_default_rate_upb_weighted']:.2%}; "
              f"12m PD {s['pd_12m_upb_weighted']:.3%}; LGD {s['mean_lgd_exposure_weighted']:.1%}")
        print("      by segment:", {k: f"{v['ecl_pct_of_upb']:.2%}" for k, v in s["by_segment"].items()})
    out["ecl_ratio_severe_to_baseline"] = out["scenarios"]["severely_adverse"]["ecl_dollars"] / out["scenarios"]["baseline"]["ecl_dollars"]
    out["notes"] = ["ECL is undiscounted; EAD = scheduled start-of-quarter balance (no curtailments).",
                    "Macro: Fed scenario for 13 quarters then linear reversion over 4 quarters to the 1990-2025 mean.",
                    "Prepayment hazard included: S(t) = prod(1 - h_default - h_prepay)."]
    (ROOT / "results" / "ecl.json").write_text(json.dumps(out, indent=2))
    print("[done] results/ecl.json")


if __name__ == "__main__":
    main()
