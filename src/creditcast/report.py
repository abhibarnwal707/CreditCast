"""Phase 9: generate docs/model_development_document.md from results/*.json (no hand-typed numbers).

Run: python -m creditcast.report
"""
import json

import pandas as pd

from .config import ROOT, load_config


def J(name):
    return json.loads((ROOT / "results" / name).read_text())


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def write_readme(ds, pm, seg, lgd, ecl, st, bt, ps, cfg) -> None:
    g, x = pm["glm"], pm["xgboost"]
    b, a = ecl["scenarios"]["baseline"], ecl["scenarios"]["severely_adverse"]
    sb, sa = st["scenarios"]["baseline"], st["scenarios"]["severely_adverse"]
    o, fb, s = bt["oot_quarterly_default_rate"], bt["forward_backtest"], ps["vs_oot_summary"]
    t = f"""# CreditCast

CECL-style expected credit loss and Fed 2026 stress testing on Freddie Mac mortgage loans: a small-scale version of the PD / LGD / EAD loss-forecasting, back-testing and monitoring pipeline a bank's credit-risk modelling team runs.
**Retail mortgages, not wholesale**; {ds['loans']:,} sampled loans, not the full dataset.

## Pipeline

```
Freddie Mac loans (origination + monthly performance)  --DuckDB SQL-->  loan-quarter panel
        + FRED macro + Fed 2026 scenarios (13 quarters, then mean reversion)
        |  K-Means segments, features at start of quarter (no look-ahead)
        v
PD: logistic GLM discrete-time hazard (XGBoost challenger)   LGD: P(liquidation) x severity   EAD: amortisation   Prepayment hazard
        |
        v
ECL = sum_t S(t-1) x PD(t) x LGD(t) x EAD(t)  -->  baseline lifetime ECL, severely adverse stress loss, sensitivity grid
        |
        v
Out-of-time validation, calibration, forward back-test, PSI monitoring, COVID appendix, model development document
```

## Results (generated from `results/*.json`)

| Item | Result |
|---|---|
| Sample | {ds['loans']:,} loans, {ds['loan_quarters']:,} loan-quarters, {ds['default_events']:,} default events (train {ds['by_split']['train']['defaults']:,}, OOT {ds['by_split']['oot']['defaults']:,}) |
| PD GLM, out-of-time {bt['oot_window']} | AUC {g['oot']['auc']:.3f}, Gini {g['oot']['gini']:.3f}, KS {g['oot']['ks']:.3f} |
| XGBoost challenger, out-of-time | AUC {x['oot']['auc']:.3f}, Gini {x['oot']['gini']:.3f}, KS {x['oot']['ks']:.3f} |
| OOT calibration | predicted {pct(o['overall_pred'], 3)} vs actual {pct(o['overall_actual'], 3)} default rate per quarter (under-predicts the level) |
| Forward back-test {fb['origin']}-{fb['end']} | predicted {pct(fb['predicted_cumulative_default_rate'], 2)} vs actual {pct(fb['actual_cumulative_default_rate'], 2)} cumulative default |
| LGD | mean severity {pct(lgd['mean_lgd'], 1)} on {lgd['defaults_with_usable_lgd']:,} liquidated defaults; cure-adjusted {pct(lgd['cure_adjusted_mean_lgd_per_default_event'], 1)} per default event |
| Lifetime ECL (baseline / severely adverse) | {pct(b['ecl_pct_of_upb'], 3)} / {pct(a['ecl_pct_of_upb'], 3)} of starting UPB |
| 9-quarter loss rate (baseline / severely adverse) | {pct(sb['loss_rate_9q'], 4)} / {pct(sa['loss_rate_9q'], 4)} ({st['ratio_severely_adverse_to_baseline_loss_9q']:.1f}x) |
| K-Means segments | k = {seg['k']}, silhouette {seg['silhouette']:.3f} |
| PSI, training vs OOT | {s['variables_monitored']} variables, max {s['max_psi']:.2f} ({s['max_psi_variable']}), macro drivers shift by construction |

Absolute loss levels are small because the portfolio still alive at the Q4 2025 jump-off is seasoned, high-FICO and carries large house-price gains; read the scenario ratio, not the level.

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e .
# place Freddie Mac sample files in data/raw/sample_YYYY/, the user guide and Fed scenario PDFs in docs/reference/, FRED_API_KEY in .env
make all      # data, macro, arima, features, train, lgd, ecl, stress, validate, report, resume_numbers
make test
```

Config: `config.yaml` (seed 42, time splits). Raw data is not committed (`data/` is git-ignored). XGBoost on macOS needs `libomp`.

## Documentation
`docs/model_development_document.md` (SR 11-7 style, generated), `docs/monitoring_report.md`, `docs/assumptions.md`, `docs/limitations.md`, `results/RESUME_NUMBERS.md`.

## Limitations
- Retail mortgage portfolio, not wholesale (no ratings, revolvers or low-default portfolio effects).
- 50,000-loan sample from 7 vintages; national (not regional) macro variables; Standard dataset only.
- The PD GLM ranks well out of time but under-predicts the default level (see results); it was not tuned on the test set.
- COVID forbearance (2020-2021) is excluded from metrics and shown in an appendix.
- Default is absorbing; no delinquency roll-rate model; undiscounted ECL; LGD is extrapolated for deeply in-the-money loans and has low explanatory power.
- No Streamlit demo and no verified CI run yet.
"""
    (ROOT / "README.md").write_text(t)


def main() -> None:
    cfg = load_config()
    ds, pm, seg, lgd, ecl, st, bt, ps, ar, ft = (J(n) for n in ("data_summary.json", "pd_metrics.json", "segments.json", "lgd.json", "ecl.json",
                                                              "stress.json", "backtest.json", "psi.json", "arima.json", "features.json"))
    pp = J("prepay_model.json")
    coef = pd.read_csv(ROOT / "results" / "pd_coefficients.csv")
    b, a = ecl["scenarios"]["baseline"], ecl["scenarios"]["severely_adverse"]
    sb, sa = st["scenarios"]["baseline"], st["scenarios"]["severely_adverse"]
    g, x = pm["glm"], pm["xgboost"]
    L = []
    w = L.append
    w("# CreditCast: Model Development Document\n")
    w("*SR 11-7 style development document. Every figure is generated from `results/*.json` by `python -m creditcast.report`.*\n")
    w("## 1. Purpose and scope\n")
    w("CreditCast estimates CECL-style lifetime expected credit loss (ECL) and stress losses under the Federal Reserve 2026 supervisory scenarios for a "
      "portfolio of **retail single-family mortgages** (Freddie Mac Standard dataset). Components: probability of default (PD) as a discrete-time logistic hazard, "
      "loss given default (LGD), exposure at default (EAD), a prepayment hazard, macroeconomic scenario paths and an ECL engine. "
      "**This is a retail mortgage portfolio, not a wholesale (corporate) portfolio**; the methodology (PD x LGD x EAD, CECL reserve, stress testing, back-testing, monitoring) is the same, the inputs differ.\n")
    w("## 2. Data\n")
    c = ds["cleaning"]
    w(f"- **Source:** Freddie Mac Single-Family Loan-Level Dataset, Standard sample files, origination vintages {', '.join(str(v) for v in ds['loans_per_vintage'])} "
      f"({c['raw_loans']:,} loans). Pipe-delimited files without headers; the schema comes from the user guide (`src/creditcast/schema.py`).")
    w("- **Cleaning (one rule per loan, first failing rule):** " + "; ".join(f"{k} removed {v:,}" for k, v in c["removed_by_rule"].items()) + f". Valid after cleaning: {c['valid_after_cleaning']:,}.")
    w(f"- **Sample:** {ds['loans']:,} loans, equal quota per vintage, deterministic hash with seed {cfg['seed']}. "
      f"Panel: **{ds['loan_quarters']:,} loan-quarters** ({ds['observation_window'][0]} to {ds['observation_window'][1]}), {ds['default_events']:,} default events, {ds['prepayment_events']:,} prepayments.")
    w("- **Default:** first quarter with 90+ days delinquency (REO acquisition included) or a credit-event termination (zero-balance codes 02, 03, 09, 15); absorbing. "
      "**Prepayment:** code 01. Codes 16 and 96 are treated as censored. Delinquency status is not a model feature.")
    w("- **Time splits (by outcome quarter):** train <= " + cfg["splits"]["train_end"] + ", validation " + cfg["splits"]["val_start"] + "-" + cfg["splits"]["val_end"] +
      ", out-of-time " + cfg["splits"]["oot_start"] + "-" + cfg["splits"]["oot_end"] + "; 2020-2021 (COVID forbearance) is excluded from metrics.\n")
    w("| Split | Loan-quarters | Defaults | Default rate / quarter |\n|---|---|---|---|")
    for k in ("train", "validation", "oot", "covid", "other"):
        r = ds["by_split"][k]
        w(f"| {k} | {r['loan_quarters']:,} | {r['defaults']:,} | {pct(r['default_rate_per_quarter'], 3)} |")
    w("\nEver-default rate by vintage: " + ", ".join(f"{k}: {pct(v['ever_default_rate'], 1)}" for k, v in ds["default_rate_by_origination_year"].items()) + ".\n")
    w("**Macro data:** 5 FRED series (UNRATE, USSTHPI, MORTGAGE30US, GDPC1, GS10), converted to quarters. **Scenarios:** Fed 2026 baseline and severely adverse, Q1 2026-Q1 2029, jump-off Q4 2025.\n")
    w("## 3. Methodology\n")
    w("**Features** (start of quarter): FICO, DTI, original rate, loan age and age squared (centred), mark-to-market LTV (current UPB / (original value x HPI(q-1)/HPI(origination))), "
      "rate incentive, purpose and occupancy dummies, and macro variables dated q-1 (unemployment, 4-quarter unemployment change, HPI growth). "
      f"VIF threshold 10; selected variables: {', '.join(pm['features'])}. Original LTV was excluded because it changed sign once MTM LTV was included.\n")
    w(f"**Segmentation:** K-Means on standardised FICO, LTV, DTI, rate, fitted on training loans only; k = {seg['k']} by silhouette ({seg['silhouette']:.3f}).\n")
    w("| Segment | Loans | Mean FICO | Mean LTV | Mean DTI | Train default rate / quarter |\n|---|---|---|---|---|---|")
    for p in seg["segment_profiles"]:
        w(f"| {p['segment']} | {p['loans']:,} | {p['mean_fico']:.0f} | {p['mean_ltv']:.1f} | {p['mean_dti']:.1f} | {pct(p['train_default_rate_per_quarter'], 3)} |")
    w("\n**PD:** statsmodels GLM, Binomial family, logit link, cluster-robust standard errors by loan, fitted on the training period. Coefficients (per stated unit):\n")
    w("| Variable | Unit | Coef | Odds ratio | p-value | Expected sign | OK |\n|---|---|---|---|---|---|---|")
    for r in coef.itertuples():
        w(f"| {r.variable} | {r.unit if isinstance(r.unit, str) else ''} | {r.coef:.4f} | {r.odds_ratio:.3f} | {r.p_value:.4f} | {r.expected_sign if isinstance(r.expected_sign, str) else ''} | "
          f"{'' if pd.isna(r.sign_ok) else ('yes' if r.sign_ok else 'NO')} |")
    w(f"\n**LGD:** LGD = P(liquidation | default) x severity. {lgd['default_events']:,} default events: {lgd['default_status']}. {lgd['defaults_with_usable_lgd']:,} liquidated defaults have a usable loss; "
      f"mean severity {pct(lgd['mean_lgd'], 1)} (median {pct(lgd['median_severity'], 1)}); P(liquidation | default) {pct(lgd['liquidation_probability_given_default'], 1)}; "
      f"cure-adjusted mean LGD per default event {pct(lgd['cure_adjusted_mean_lgd_per_default_event'], 1)}. Severity falls with house-price gains since origination "
      f"(slope {lgd['model']['sev_slope']:.4f} per +10 pp, 95% CI {lgd['model_uncertainty_95ci']['severity_slope_per_10pp_hpi_change'][0]:.4f} to {lgd['model_uncertainty_95ci']['severity_slope_per_10pp_hpi_change'][1]:.4f}); "
      f"the liquidation probability rises with MTM LTV.\n")
    w("| MTM LTV bucket | Resolved defaults | Liquidation rate | Mean severity | Cure-adjusted LGD (actual) | (model) |\n|---|---|---|---|---|---|")
    for r in lgd["by_mtm_ltv_bucket"]:
        w(f"| {r['mtm_ltv_bucket']} | {r['defaults_resolved']:,} | {pct(r['liquidation_rate'], 1)} | {pct(r['mean_severity'], 1)} | {pct(r['cure_adjusted_lgd_actual'], 1)} | {pct(r['cure_adjusted_lgd_model'], 1)} |")
    w("\n**EAD:** scheduled amortisation from the current balance, rate and remaining term (level payment, no curtailment); start-of-quarter balance.\n")
    w(f"**Prepayment:** second logistic hazard (rate incentive, age, MTM LTV, FICO, HPI growth), {pp['prepay_events']:,} events, in-sample AUC {pp['auc_in_sample']:.3f}.\n")
    w("**Macro paths and reversion:** Fed scenario for 13 quarters (reasonable and supportable period), then linear reversion over 4 quarters to the 1990-2025 mean "
      "(unemployment, mortgage rate, 10y yield) and to the mean HPI growth. ARIMA (below) is a sanity check and supports the mean-reverting extension.\n")
    w("**ECL:** `ECL = sum_t S(t-1) x h_d(t) x LGD(t) x EAD(t)` with `S(t) = prod (1 - h_d - h_p)`, up to 120 quarters, undiscounted.\n")
    w("**ARIMA (rolling-origin backtest, 1-8 quarters ahead, first origin " + ar["eval_first_origin"] + "):**\n")
    w("| Series | d | Order | Avg RMSE ARIMA | Random walk | Mean | vs random walk | vs mean |\n|---|---|---|---|---|---|---|---|")
    for k, v in ar["series"].items():
        bt_ = v["backtest"]
        w(f"| {k} | {v['d']} | {tuple(v['order'])} | {bt_['avg_rmse_1_to_8q']['arima']:.3f} | {bt_['avg_rmse_1_to_8q']['random_walk']:.3f} | {bt_['avg_rmse_1_to_8q']['mean']:.3f} | "
          f"{bt_['improvement_vs_random_walk_pct']:+.1f}% | {bt_['improvement_vs_mean_pct']:+.1f}% |")
    w("\nARIMA has weak forecast skill, as expected for macro series; it does not beat the random walk for unemployment. It is not used to drive the stress paths.\n")
    w("## 4. Assumptions and limitations\n")
    w("See `docs/assumptions.md` and `docs/limitations.md`. Key items: retail not wholesale; sample of 50,000 loans from 7 vintages; national macro variables; "
      "COVID forbearance excluded; absorbing default (no cures); no delinquency roll-rate model; undiscounted ECL; small, seasoned jump-off portfolio; LGD extrapolation beyond the observed house-price range.\n")
    w("## 5. Results\n")
    w(f"Jump-off portfolio at {ecl['jump_off']}: {ecl['loans']:,} loans alive, starting UPB ${ecl['starting_upb']:,.0f}.\n")
    w("| Scenario | Lifetime ECL (% of UPB) | Lifetime ECL ($) | Lifetime cum. default rate | 12-month PD | Exposure-weighted LGD |\n|---|---|---|---|---|---|")
    for n, s in (("Baseline", b), ("Severely adverse", a)):
        w(f"| {n} | {pct(s['ecl_pct_of_upb'], 3)} | ${s['ecl_dollars']:,.0f} | {pct(s['lifetime_cumulative_default_rate_upb_weighted'], 2)} | {pct(s['pd_12m_upb_weighted'], 3)} | {pct(s['mean_lgd_exposure_weighted'], 1)} |")
    w("\nLifetime ECL by K-Means segment (baseline / severely adverse): " + "; ".join(
        f"segment {k}: {pct(b['by_segment'][k]['ecl_pct_of_upb'], 3)} / {pct(a['by_segment'][k]['ecl_pct_of_upb'], 3)}" for k in b["by_segment"]) + ".\n")
    w("Absolute loss levels are small because the surviving portfolio is seasoned, high-FICO and carries large house-price gains (mean MTM LTV near 23%); the scenario ratio is the more informative number.\n")
    w("## 6. Benchmark vs XGBoost challenger\n")
    w("| Model | Sample | AUC | Gini | KS | Brier | Mean predicted | Mean actual |\n|---|---|---|---|---|---|---|---|")
    for n, m in (("GLM", g), ("XGBoost", x)):
        for s in ("train", "validation", "oot"):
            r = m[s]
            w(f"| {n} | {s} | {r['auc']:.3f} | {r['gini']:.3f} | {r['ks']:.3f} | {r['brier']:.5f} | {pct(r['mean_pred'], 3)} | {pct(r['mean_actual'], 3)} |")
    d_oot = x["oot"]["auc"] - g["oot"]["auc"]
    w(f"\nSelection rule (validation only): champion = GLM unless XGBoost's validation AUC is more than 0.02 higher. Difference on validation: {pm['selection']['xgb_minus_glm_validation_auc']:+.4f}, so the GLM is the champion. "
      f"On the OOT set XGBoost is {d_oot:+.4f} AUC points relative to the GLM.\n")
    w("## 7. Sensitivity and stress results\n")
    w(f"9-quarter horizon ({st['horizon']}), loss = cumulative ECL / starting UPB.\n")
    w("| Scenario | 9-quarter loss rate | 9-quarter cum. default rate | Peak quarterly default rate | Peak quarter |\n|---|---|---|---|---|")
    for n, s in (("Baseline", sb), ("Severely adverse", sa)):
        w(f"| {n} | {pct(s['loss_rate_9q'], 4)} | {pct(s['cum_default_rate_9q'], 3)} | {pct(s['peak_quarterly_default_rate'], 3)} | {s['peak_quarter']} |")
    w(f"\nSeverely adverse / baseline: **{st['ratio_severely_adverse_to_baseline_loss_9q']:.1f}x** (9-quarter loss), {st['ratio_peak_default_rate']:.1f}x (peak default rate).\n")
    gr = st["sensitivity_grid"]
    w("9-quarter loss rate (% of starting UPB), peak unemployment (rows) x peak house-price decline (columns); monotone in both directions: "
      f"{gr['monotone_in_unemployment'] and gr['monotone_in_hpi_decline']}.\n")
    w("| Peak unemployment | " + " | ".join(f"HPI -{d}%" for d in gr["hpi_decline_pct"]) + " |\n|---|" + "---|" * len(gr["hpi_decline_pct"]))
    for u, row in zip(gr["unemployment_peak_pct"], gr["loss_rate_9q"]):
        w(f"| {u}% | " + " | ".join(f"{v * 100:.4f}" for v in row) + " |")
    w("\n## 8. Back-testing and ongoing monitoring\n")
    o = bt["oot_quarterly_default_rate"]
    w(f"- **Out-of-time ({bt['oot_window']}):** GLM AUC {g['oot']['auc']:.3f}, Gini {g['oot']['gini']:.3f}, KS {g['oot']['ks']:.3f}. "
      f"Predicted default rate {pct(o['overall_pred'], 3)} vs actual {pct(o['overall_actual'], 3)} per quarter (the model under-predicts the level by a factor of "
      f"{o['overall_actual'] / o['overall_pred']:.1f}; ranking is good, calibration is not); quarterly MAPE {pct(o['mape'], 1)}.")
    w(f"- **Hosmer-Lemeshow style test:** chi2 {bt['hosmer_lemeshow_glm_oot']['chi2']:.1f}, p = {bt['hosmer_lemeshow_glm_oot']['p_value']:.3g}. Segment binomial tests (below) also reject calibration: every segment is under-predicted.\n")
    w("| Segment | Loan-quarters | Defaults | Predicted | Binomial p |\n|---|---|---|---|---|")
    for r in bt["calibration_by_segment_glm_oot"]:
        w(f"| {r['segment']} | {r['loan_quarters']:,} | {r['defaults']} | {r['pred_defaults']:.1f} | {r['binomial_test_p']:.3g} |")
    fb = bt["forward_backtest"]
    w(f"\n- **Forward back-test:** {fb['loans_alive_at_origin']:,} loans alive at {fb['origin']}, realised macro, to {fb['end']}: predicted cumulative default "
      f"{pct(fb['predicted_cumulative_default_rate'], 2)} ({fb['predicted_defaults']:.0f} loans) vs actual {pct(fb['actual_cumulative_default_rate'], 2)} ({fb['actual_defaults']:.0f} loans); ratio {fb['ratio_pred_to_actual']:.2f}.")
    s = ps["vs_oot_summary"]
    w(f"- **PSI (training vs OOT):** {s['variables_monitored']} variables monitored (score + drivers), max PSI {s['max_psi']:.2f} ({s['max_psi_variable']}); {s['n_stable']} stable, {s['n_watch']} watch, {s['n_shifted']} shifted. "
      "Macro variables shift by construction between the crisis-heavy training window and the benign OOT window.")
    cv = bt["covid_appendix"]
    w(f"- **COVID appendix:** 2020-2021 actual default rate {pct(cv['covid']['mean_actual'], 3)} vs predicted {pct(cv['covid']['mean_pred'], 3)}; forbearance-driven delinquency, not credit stress. Excluded from headline metrics.")
    w("- **Monitoring plan:** quarterly AUC/Gini/KS, predicted vs actual default rate and PSI (thresholds 0.10 / 0.25); annual recalibration and challenger review. See `docs/monitoring_report.md`.")
    w("- **Recommended remediation (not applied, to keep the OOT test clean):** recalibrate the intercept on a more recent window and re-test on a fresh hold-out.\n")
    w("## 9. Change log\n")
    w("| Version | Change |\n|---|---|\n| 0.1 | Initial development: panel, macro, features, PD/LGD/EAD, ECL, stress, validation. |\n"
      "| 0.1 | Original LTV dropped from PD after the first fit (wrong sign with MTM LTV). |\n"
      "| 0.1 | LGD severity re-specified on house-price change since origination (flat against MTM LTV). |")
    (ROOT / "docs" / "model_development_document.md").write_text("\n".join(L) + "\n")
    write_readme(ds, pm, seg, lgd, ecl, st, bt, ps, cfg)
    words = len(" ".join(L).split())
    print(f"[done] docs/model_development_document.md ({words:,} words, about {words / 450:.1f} pages)")


if __name__ == "__main__":
    main()
