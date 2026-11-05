# CreditCast

CECL-style expected credit loss and Fed 2026 stress testing on Freddie Mac mortgage loans: a small-scale version of the PD / LGD / EAD loss-forecasting, back-testing and monitoring pipeline a bank's credit-risk modelling team runs.
**Retail mortgages, not wholesale**; 50,000 sampled loans, not the full dataset.

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
| Sample | 50,000 loans, 1,031,382 loan-quarters, 3,747 default events (train 2,462, OOT 254) |
| PD GLM, out-of-time 2017Q1-2019Q4 | AUC 0.786, Gini 0.572, KS 0.446 |
| XGBoost challenger, out-of-time | AUC 0.798, Gini 0.595, KS 0.459 |
| OOT calibration | predicted 0.055% vs actual 0.112% default rate per quarter (under-predicts the level) |
| Forward back-test 2015Q4-2019Q4 | predicted 0.86% vs actual 1.55% cumulative default |
| LGD | mean severity 51.5% on 1,358 liquidated defaults; cure-adjusted 22.7% per default event |
| Lifetime ECL (baseline / severely adverse) | 0.022% / 0.130% of starting UPB |
| 9-quarter loss rate (baseline / severely adverse) | 0.0124% / 0.0974% (7.9x) |
| K-Means segments | k = 3, silhouette 0.252 |
| PSI, training vs OOT | 15 variables, max 7.70 (unemp_lag), macro drivers shift by construction |

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
