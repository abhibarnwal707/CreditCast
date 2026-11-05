"""Phase 11: results/RESUME_NUMBERS.md, every value read from results/*.json by script. 'NOT BUILT' for anything not built.

Run: python -m creditcast.resume_numbers
"""
import json
import re
import subprocess
import sys

from .config import ROOT, load_config

NB = "NOT BUILT"


def J(n):
    return json.loads((ROOT / "results" / n).read_text())


def sh(cmd):
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300).stdout.strip()
    except Exception:
        return ""


def main() -> None:
    cfg = load_config()
    ds, pm, seg, lgd, ecl, st, bt, ps, ar = (J(n) for n in ("data_summary.json", "pd_metrics.json", "segments.json", "lgd.json", "ecl.json", "stress.json", "backtest.json", "psi.json", "arima.json"))
    g, x = pm["glm"], pm["xgboost"]
    sb, sa = st["scenarios"]["baseline"], st["scenarios"]["severely_adverse"]
    b = ecl["scenarios"]["baseline"]
    o = bt["oot_quarterly_default_rate"]
    n_tests = len(re.findall(r"^tests/.*::", sh([sys.executable, "-m", "pytest", "--collect-only", "-q"]), flags=re.M))
    doc = ROOT / "docs" / "model_development_document.md"
    words = len(doc.read_text().split()) if doc.exists() else 0
    remote = sh(["git", "remote", "get-url", "origin"]) or "no 'origin' remote configured"
    r = []
    add = lambda n, item, val, src: r.append(f"| {n} | {item} | {val} | {src} |")
    pct = lambda v, d=2: f"{v * 100:.{d}f}%"
    arima_rows = "; ".join(f"{k}: ARIMA{tuple(v['order'])}, RMSE vs random walk {v['backtest']['improvement_vs_random_walk_pct']:+.1f}%, vs mean {v['backtest']['improvement_vs_mean_pct']:+.1f}%" for k, v in ar["series"].items())
    add(1, "Loans / loan-quarters / vintages", f"{ds['loans']:,} loans; {ds['loan_quarters']:,} loan-quarters; vintages {', '.join(str(v) for v in ds['loans_per_vintage'])}; window {ds['observation_window'][0]}-{ds['observation_window'][1]}", "data_summary.json")
    add(2, "Default events (train / validation / OOT / total)", f"{ds['by_split']['train']['defaults']:,} / {ds['by_split']['validation']['defaults']:,} / {ds['by_split']['oot']['defaults']:,} / {ds['default_events']:,}", "data_summary.json")
    add(3, "PD GLM OOT AUC / Gini / KS", f"{g['oot']['auc']:.3f} / {g['oot']['gini']:.3f} / {g['oot']['ks']:.3f} (validation AUC {g['validation']['auc']:.3f})", "pd_metrics.json")
    add(4, "XGBoost challenger OOT AUC / Gini / KS", f"{x['oot']['auc']:.3f} / {x['oot']['gini']:.3f} / {x['oot']['ks']:.3f}; XGBoost minus GLM = {x['oot']['auc'] - g['oot']['auc']:+.3f} AUC (validation {pm['selection']['xgb_minus_glm_validation_auc']:+.3f})", "pd_metrics.json")
    add(5, "Mean LGD / defaults with usable LGD", f"mean severity {pct(lgd['mean_lgd'], 1)} on {lgd['defaults_with_usable_lgd']:,} liquidated defaults; cure-adjusted LGD per default event {pct(lgd['cure_adjusted_mean_lgd_per_default_event'], 1)}", "lgd.json")
    add(6, "Lifetime ECL rate, baseline (% of UPB)", f"{pct(b['ecl_pct_of_upb'], 3)} (severely adverse {pct(ecl['scenarios']['severely_adverse']['ecl_pct_of_upb'], 3)}); jump-off {ecl['jump_off']}, {ecl['loans']:,} loans", "ecl.json")
    add(7, "9-quarter loss rate: baseline vs severely adverse, ratio", f"{pct(sb['loss_rate_9q'], 4)} vs {pct(sa['loss_rate_9q'], 4)}; ratio {st['ratio_severely_adverse_to_baseline_loss_9q']:.1f}x", "stress.json")
    add(8, "Peak quarterly default rate under stress", f"{pct(sa['peak_quarterly_default_rate'], 3)} in {sa['peak_quarter']} (baseline peak {pct(sb['peak_quarterly_default_rate'], 3)})", "stress.json")
    add(9, "K-Means k and silhouette", f"k = {seg['k']}, silhouette {seg['silhouette']:.3f}", "segments.json")
    add(10, "ARIMA orders and RMSE improvement vs naive", arima_rows, "arima.json")
    add(11, "Predicted vs actual default rate, OOT (and MAPE)", f"predicted {pct(o['overall_pred'], 3)} vs actual {pct(o['overall_actual'], 3)} per quarter; quarterly MAPE {pct(o['mape'], 1)}. Forward back-test {bt['forward_backtest']['origin']}-{bt['forward_backtest']['end']}: predicted {pct(bt['forward_backtest']['predicted_cumulative_default_rate'], 2)} vs actual {pct(bt['forward_backtest']['actual_cumulative_default_rate'], 2)} cumulative default", "backtest.json")
    s = ps["vs_oot_summary"]
    add(12, "Max PSI and variables monitored", f"max PSI {s['max_psi']:.2f} ({s['max_psi_variable']}); {s['variables_monitored']} variables monitored (score + drivers); stable {s['n_stable']}, watch {s['n_watch']}, shifted {s['n_shifted']}", "psi.json")
    add(13, "FRED series used", f"{len(cfg['fred_series'])} ({', '.join(cfg['fred_series'])})", "config.yaml")
    add(14, "pytest tests / CI status", f"{n_tests} tests; CI workflow file present (.github/workflows/tests.yml) but NOT RUN yet: CI status unverified", "pytest --collect-only")
    add(15, "Model document pages", f"{NB} as PDF/pages (no pandoc); markdown document of {words:,} words generated from results" if words else NB, "docs/model_development_document.md")
    add(16, "GitHub link / Demo link", f"{remote} / Demo: {NB} (no Streamlit app)", "git remote")
    text = ["# RESUME_NUMBERS (generated by `make resume_numbers`; every value read from results/*.json)", "",
            "| # | Item | Value | Source |", "|---|---|---|---|"] + r
    text += ["", "## Libraries used", "", "Python: duckdb, pandas, numpy, scipy, statsmodels, scikit-learn, xgboost, matplotlib, pdfplumber, fredapi, pyarrow, PyYAML, pytest.  SQL: DuckDB SQL (`sql/clean_sample.sql`, `sql/build_panel.sql`).", "",
             "## Built vs not built", "",
             "- Built: data/panel (SQL), FRED + Fed scenarios, ARIMA, K-Means, logistic GLM PD, XGBoost challenger, LGD, EAD, prepayment hazard, ECL engine, stress test + sensitivity grid, OOT validation, forward back-test, PSI monitoring, COVID appendix, model document (markdown), tests.",
             f"- {NB}: Streamlit demo, verified CI run, PDF model document.", "",
             "## Caveats to keep in any wording",
             f"- OOT calibration: the GLM under-predicts the default level (predicted {pct(o['overall_pred'], 3)} vs actual {pct(o['overall_actual'], 3)} per quarter); ranking (Gini/KS) is good.",
             "- Retail mortgages, not wholesale; 50,000-loan sample from 7 vintages; the surviving jump-off portfolio is small and seasoned, so absolute loss rates are small (use the ratio).",
             "- ARIMA does not beat a random walk for unemployment (see item 10); do not claim ARIMA skill beyond what is shown."]
    (ROOT / "results" / "RESUME_NUMBERS.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
