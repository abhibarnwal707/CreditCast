"""Phase 8: out-of-time validation, calibration, forward back-test, PSI monitoring, COVID appendix.

The OOT test set (2017Q1-2019Q4) is used here for the first time.
Run:  python -m creditcast.validate -> results/backtest.json, results/psi.json, results/pd_metrics.json (OOT added),
                                       docs/monitoring_report.md, results/figures/{backtest_*.png,rolling_auc.png}
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

from .config import ROOT, assign_split, idx_to_q, load_config, q_to_idx
from .ecl import fit_prepay, load_portfolio, prepare_long, project
from .features import macro_by_qidx
from .monitor import PSI_SHIFT, PSI_WATCH, psi, psi_flag, rolling_quarterly_auc
from .pd_model import calibration_table, metrics, predict_glm


def hosmer_lemeshow(y, p, g=10) -> dict:
    t = calibration_table(y, p, g)
    exp = t["predicted"] * t["n"]
    obs = t["defaults"]
    chi2 = float((((obs - exp) ** 2) / (exp * (1 - t["predicted"]))).sum())
    return {"chi2": chi2, "df": g - 2, "p_value": float(1 - stats.chi2.cdf(chi2, g - 2)),
            "note": "rare events: expected counts in low deciles are tiny, so the test is only indicative"}


def forward_backtest(cfg, feat_all, pdp, origin="2015Q4", end="2019Q4") -> dict:
    jump = q_to_idx(origin)
    n_q = q_to_idx(end) - jump
    pp = fit_prepay(cfg, max_qidx=jump)                               # only data available at the origin
    lgd = json.loads((ROOT / "results" / "lgd.json").read_text())["model"]
    port = load_portfolio(cfg, jump)
    port = port.merge(feat_all[["loan_id", "segment"]].drop_duplicates("loan_id"), on="loan_id", how="left", suffixes=("", "_f")) if "segment" not in port else port
    L = prepare_long(port, n_q)
    mac = macro_by_qidx().loc[jump + 1: jump + n_q]
    path = pd.DataFrame({"unemployment": mac["unemp"].values, "hpi": mac["hpi"].values, "mortgage_rate": mac["mort"].values})   # REALISED macro
    r = project(port, L, path, pdp, pp, lgd, cfg, jump=jump)
    N = len(port)
    pred_q = pd.Series(r["default_prob"]).groupby(r["t"]).sum() / N        # expected default share of the origin cohort per quarter
    ids = set(port["loan_id"])
    a = feat_all[feat_all["loan_id"].isin(ids) & (feat_all["qidx"] > jump) & (feat_all["qidx"] <= jump + n_q)]
    act_q = a.groupby("qidx")["default"].sum() / N
    act_q.index = act_q.index - jump
    act_q = act_q.reindex(range(1, n_q + 1)).fillna(0.0)
    cum_pred, cum_act = float(pred_q.sum()), float(act_q.sum())
    return {"origin": origin, "end": end, "loans_alive_at_origin": int(N), "quarters": n_q,
            "predicted_cumulative_default_rate": cum_pred, "actual_cumulative_default_rate": cum_act,
            "predicted_defaults": cum_pred * N, "actual_defaults": cum_act * N, "ratio_pred_to_actual": cum_pred / cum_act,
            "by_quarter": [{"quarter": idx_to_q(jump + int(t)), "pred": float(pred_q.get(t, 0)), "actual": float(act_q[t])} for t in range(1, n_q + 1)],
            "note": "loans alive at origin projected with REALISED macro; prepayment model refit on data to the origin only"}


def main() -> None:
    cfg = load_config()
    feat = pd.read_parquet(ROOT / "data" / "interim" / "features.parquet")
    feat["split"] = assign_split(feat["qidx"], cfg).values
    pm = json.loads((ROOT / "results" / "pd_metrics.json").read_text())
    selected = pm["features"]
    pdp = json.loads((ROOT / "results" / "pd_glm_coefficients.json").read_text())["params"]
    clf = xgb.XGBClassifier()
    clf.load_model(str(ROOT / "data" / "interim" / "xgb_challenger.json"))
    feat["p_glm"] = predict_glm(pdp, feat)
    feat["p_xgb"] = clf.predict_proba(feat[selected].astype(float))[:, 1]
    oot, tr = feat[feat.split == "oot"], feat[feat.split == "train"]
    print(f"[oot] loan-quarters {len(oot):,}, defaults {int(oot['default'].sum())}")

    bt = {"oot_window": f"{cfg['splits']['oot_start']}-{cfg['splits']['oot_end']}"}
    for name, col in (("glm", "p_glm"), ("xgboost", "p_xgb")):
        pm[name]["oot"] = metrics(oot["default"], oot[col])
        r = pm[name]["oot"]
        print(f"[oot] {name:<8} AUC {r['auc']:.4f} Gini {r['gini']:.4f} KS {r['ks']:.4f} Brier {r['brier']:.5f} pred {r['mean_pred']:.4%} actual {r['mean_actual']:.4%}")
    pm["glm"]["calibration_oot"] = calibration_table(oot["default"], oot["p_glm"]).round(6).to_dict("records")
    pm["xgboost"]["calibration_oot"] = calibration_table(oot["default"], oot["p_xgb"]).round(6).to_dict("records")
    pm["oot_note"] = "OOT set used for the first time in validate.py, after model selection on validation"
    (ROOT / "results" / "pd_metrics.json").write_text(json.dumps(pm, indent=2))
    bt["oot_metrics_glm"], bt["oot_metrics_xgboost"] = pm["glm"]["oot"], pm["xgboost"]["oot"]
    bt["hosmer_lemeshow_glm_oot"] = hosmer_lemeshow(oot["default"], oot["p_glm"])
    bt["calibration_by_decile_glm_oot"] = pm["glm"]["calibration_oot"]

    seg_rows = []
    for s, g in oot.groupby("segment"):
        k, n, pbar = int(g["default"].sum()), len(g), float(g["p_glm"].mean())
        bp = stats.binomtest(k, n, pbar).pvalue
        seg_rows.append({"segment": int(s), "loan_quarters": n, "defaults": k, "pred_defaults": float(g["p_glm"].sum()),
                         "pred_rate": pbar, "actual_rate": k / n, "binomial_test_p": float(bp)})
    bt["calibration_by_segment_glm_oot"] = seg_rows
    print(pd.DataFrame(seg_rows).round(5).to_string(index=False))

    q = oot.groupby("qidx").agg(n=("default", "size"), d=("default", "sum"), pred=("p_glm", "mean"))
    q["actual"] = q["d"] / q["n"]
    mape = float((np.abs(q["pred"] - q["actual"]) / q["actual"]).replace([np.inf], np.nan).mean())
    bt["oot_quarterly_default_rate"] = {"mape": mape, "by_quarter": [{"quarter": idx_to_q(i), "pred": float(r.pred), "actual": float(r.actual), "defaults": int(r.d)} for i, r in q.iterrows()],
                                        "overall_pred": float(oot["p_glm"].mean()), "overall_actual": float(oot["default"].mean())}
    print(f"[oot] quarterly default rate: predicted {oot['p_glm'].mean():.4%} vs actual {oot['default'].mean():.4%}; MAPE {mape:.1%}")

    bt["forward_backtest"] = forward_backtest(cfg, feat, pdp)
    fb = bt["forward_backtest"]
    print(f"[backtest] loans alive at {fb['origin']}: {fb['loans_alive_at_origin']:,}; cumulative default {fb['origin']}->{fb['end']}: "
          f"predicted {fb['predicted_cumulative_default_rate']:.3%} ({fb['predicted_defaults']:.0f}) vs actual {fb['actual_cumulative_default_rate']:.3%} ({fb['actual_defaults']:.0f})")

    # COVID appendix + rolling AUC
    roll = rolling_quarterly_auc(feat[feat["qidx"] >= q_to_idx("2015Q1")], "p_glm")
    roll["quarter"] = roll["qidx"].map(idx_to_q)
    roll["period"] = assign_split(roll["qidx"], cfg).values
    cov = feat[feat.split == "covid"]
    post = feat[feat["qidx"] > q_to_idx(cfg["splits"]["covid_exclude"][1])]
    bt["covid_appendix"] = {"covid_window": cfg["splits"]["covid_exclude"],
                            "covid": metrics(cov["default"], cov["p_glm"]), "post_covid_2022_on": metrics(post["default"], post["p_glm"]),
                            "oot": pm["glm"]["oot"],
                            "explanation": "2020-2021 90+ DPD delinquency is dominated by CARES-Act forbearance, not credit deterioration: the actual default rate "
                                           "spikes (the model, built on credit drivers, under-predicts it) while ranking power stays reasonable. These quarters are excluded from all headline metrics."}
    c = bt["covid_appendix"]
    print(f"[covid] 2020-21: actual {c['covid']['mean_actual']:.3%} vs predicted {c['covid']['mean_pred']:.3%}, AUC {c['covid']['auc']:.3f}; post-2021 AUC {c['post_covid_2022_on']['auc']:.3f}")
    bt["rolling_quarterly_auc"] = roll.round(5).replace({np.nan: None}).to_dict("records")
    (ROOT / "results" / "backtest.json").write_text(json.dumps(bt, indent=2, default=float))

    # PSI: training vs OOT (headline) and vs recent (2024Q1+), score + each model variable
    recent = feat[feat["qidx"] >= q_to_idx("2024Q1")]
    ps = {"thresholds": {"watch": PSI_WATCH, "shifted": PSI_SHIFT}, "reference": "training period (<= 2014Q4)", "vs_oot": {}, "vs_recent_2024_on": {}}
    for label, cur in (("vs_oot", oot), ("vs_recent_2024_on", recent)):
        for v in ["p_glm"] + selected:
            val = psi(tr[v].values, cur[v].values)
            ps[label][("predicted_pd_score" if v == "p_glm" else v)] = {"psi": val, "flag": psi_flag(val)}
        vals = [x["psi"] for x in ps[label].values()]
        ps[label + "_summary"] = {"variables_monitored": len(vals), "max_psi": max(vals),
                                  "max_psi_variable": max(ps[label], key=lambda k: ps[label][k]["psi"]),
                                  "n_stable": sum(x["flag"] == "stable" for x in ps[label].values()),
                                  "n_watch": sum(x["flag"] == "watch" for x in ps[label].values()),
                                  "n_shifted": sum(x["flag"] == "shifted" for x in ps[label].values())}
    ps["note"] = "macro variables (unemployment, HPI growth, rates) shift by construction between the crisis-heavy training window and the benign 2017-2019 window"
    (ROOT / "results" / "psi.json").write_text(json.dumps(ps, indent=2))
    s = ps["vs_oot_summary"]
    print(f"[psi] training vs OOT: {s['variables_monitored']} variables, max PSI {s['max_psi']:.3f} ({s['max_psi_variable']}); stable {s['n_stable']}, watch {s['n_watch']}, shifted {s['n_shifted']}")

    # figures
    figdir = ROOT / "results" / "figures"
    fig, ax = plt.subplots(figsize=(7, 4))
    x = [r["quarter"] for r in bt["oot_quarterly_default_rate"]["by_quarter"]]
    ax.plot(x, [r["actual"] * 100 for r in bt["oot_quarterly_default_rate"]["by_quarter"]], "o-", label="actual", color="#c0504d")
    ax.plot(x, [r["pred"] * 100 for r in bt["oot_quarterly_default_rate"]["by_quarter"]], "s--", label="predicted (GLM)", color="#1f4e79")
    ax.set_ylabel("quarterly default rate (%)")
    ax.set_title("Out-of-time 2017-2019: predicted vs actual")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "backtest_oot_default_rate.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(roll["qidx"] / 4, roll["auc"], "o-", ms=3, color="#1f4e79")
    c0, c1 = (q_to_idx(v) for v in cfg["splits"]["covid_exclude"])
    ax.axvspan(c0 / 4, (c1 + 1) / 4, color="grey", alpha=0.25, label="COVID forbearance")
    ax.set_ylabel("quarterly AUC")
    ax.set_title("Rolling quarterly AUC of the PD model (quarters with >= 3 defaults)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "rolling_auc.png", dpi=150)
    plt.close(fig)

    # one-page monitoring report generated from the JSON (no hand-typed numbers)
    g, x_ = pm["glm"]["oot"], pm["xgboost"]["oot"]
    lines = ["# PD model monitoring report (generated by `creditcast.validate`)", "",
             f"**Out-of-time window:** {bt['oot_window']}  |  **Model:** logistic GLM discrete-time hazard (champion), XGBoost challenger", "",
             "## 1. Performance (out-of-time)", "",
             "| Model | AUC | Gini | KS | Brier | Mean predicted | Mean actual |", "|---|---|---|---|---|---|---|",
             f"| GLM | {g['auc']:.3f} | {g['gini']:.3f} | {g['ks']:.3f} | {g['brier']:.5f} | {g['mean_pred']:.4%} | {g['mean_actual']:.4%} |",
             f"| XGBoost | {x_['auc']:.3f} | {x_['gini']:.3f} | {x_['ks']:.3f} | {x_['brier']:.5f} | {x_['mean_pred']:.4%} | {x_['mean_actual']:.4%} |", "",
             f"Hosmer-Lemeshow style test (GLM, OOT): chi2 = {bt['hosmer_lemeshow_glm_oot']['chi2']:.1f}, p = {bt['hosmer_lemeshow_glm_oot']['p_value']:.3f} (indicative only: rare events).",
             f"Quarterly default-rate MAPE (OOT): {mape:.1%}.", "",
             "## 2. Calibration by K-Means segment (OOT)", "", "| Segment | Loan-quarters | Defaults | Predicted defaults | Binomial test p |", "|---|---|---|---|---|"]
    lines += [f"| {r['segment']} | {r['loan_quarters']:,} | {r['defaults']} | {r['pred_defaults']:.1f} | {r['binomial_test_p']:.3f} |" for r in seg_rows]
    lines += ["", "## 3. Forward back-test", "",
              f"Loans alive at {fb['origin']} ({fb['loans_alive_at_origin']:,}), realised macro, {fb['origin']} to {fb['end']}: predicted cumulative default "
              f"{fb['predicted_cumulative_default_rate']:.3%} ({fb['predicted_defaults']:.0f} loans) vs actual {fb['actual_cumulative_default_rate']:.3%} ({fb['actual_defaults']:.0f} loans).", "",
              "## 4. Population stability (training vs OOT)", "",
              f"{s['variables_monitored']} variables monitored; max PSI {s['max_psi']:.3f} ({s['max_psi_variable']}); stable {s['n_stable']}, watch {s['n_watch']}, shifted {s['n_shifted']}. Thresholds: < {PSI_WATCH} stable, {PSI_WATCH}-{PSI_SHIFT} watch, > {PSI_SHIFT} shifted.", "",
              "| Variable | PSI | Flag |", "|---|---|---|"]
    lines += [f"| {k} | {v['psi']:.3f} | {v['flag']} |" for k, v in sorted(ps["vs_oot"].items(), key=lambda kv: -kv[1]["psi"])]
    lines += ["", "## 5. COVID appendix", "",
              f"2020-2021 actual default rate {c['covid']['mean_actual']:.3%} vs predicted {c['covid']['mean_pred']:.3%} (AUC {c['covid']['auc']:.3f}). {c['explanation']}", "",
              "## 6. Monitoring plan", "", "- Quarterly: AUC/Gini/KS, predicted vs actual default rate, PSI of score and drivers (thresholds above).",
              "- Annual: recalibration review, challenger comparison, assumptions review.", "- Triggers: PSI > 0.25 on the score, or actual/predicted default ratio outside [0.5, 2] for two consecutive quarters."]
    (ROOT / "docs" / "monitoring_report.md").write_text("\n".join(lines) + "\n")
    print("[done] results/backtest.json, psi.json, pd_metrics.json (OOT), docs/monitoring_report.md")


if __name__ == "__main__":
    main()
