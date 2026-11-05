# CreditCast: Model Development Document

*SR 11-7 style development document. Every figure is generated from `results/*.json` by `python -m creditcast.report`.*

## 1. Purpose and scope

CreditCast estimates CECL-style lifetime expected credit loss (ECL) and stress losses under the Federal Reserve 2026 supervisory scenarios for a portfolio of **retail single-family mortgages** (Freddie Mac Standard dataset). Components: probability of default (PD) as a discrete-time logistic hazard, loss given default (LGD), exposure at default (EAD), a prepayment hazard, macroeconomic scenario paths and an ECL engine. **This is a retail mortgage portfolio, not a wholesale (corporate) portfolio**; the methodology (PD x LGD x EAD, CECL reserve, stress testing, back-testing, monitoring) is the same, the inputs differ.

## 2. Data

- **Source:** Freddie Mac Single-Family Loan-Level Dataset, Standard sample files, origination vintages 2004, 2006, 2007, 2012, 2014, 2016, 2018 (350,000 loans). Pipe-delimited files without headers; the schema comes from the user guide (`src/creditcast/schema.py`).
- **Cleaning (one rule per loan, first failing rule):** r1_credit_score_missing_or_invalid removed 153; r2_ltv_missing_or_invalid removed 18; r3_dti_missing_or_invalid removed 30,854. Valid after cleaning: 318,975.
- **Sample:** 50,000 loans, equal quota per vintage, deterministic hash with seed 42. Panel: **1,031,382 loan-quarters** (2004Q2 to 2026Q2), 3,747 default events, 40,422 prepayments.
- **Default:** first quarter with 90+ days delinquency (REO acquisition included) or a credit-event termination (zero-balance codes 02, 03, 09, 15); absorbing. **Prepayment:** code 01. Codes 16 and 96 are treated as censored. Delinquency status is not a model feature.
- **Time splits (by outcome quarter):** train <= 2014Q4, validation 2015Q1-2016Q4, out-of-time 2017Q1-2019Q4; 2020-2021 (COVID forbearance) is excluded from metrics.

| Split | Loan-quarters | Defaults | Default rate / quarter |
|---|---|---|---|
| train | 446,756 | 2,462 | 0.551% |
| validation | 119,247 | 131 | 0.110% |
| oot | 227,095 | 254 | 0.112% |
| covid | 111,962 | 776 | 0.693% |
| other | 126,322 | 124 | 0.098% |

Ever-default rate by vintage: 2004: 7.2%, 2006: 14.4%, 2007: 16.3%, 2012: 1.6%, 2014: 3.1%, 2016: 4.6%, 2018: 5.4%.

**Macro data:** 5 FRED series (UNRATE, USSTHPI, MORTGAGE30US, GDPC1, GS10), converted to quarters. **Scenarios:** Fed 2026 baseline and severely adverse, Q1 2026-Q1 2029, jump-off Q4 2025.

## 3. Methodology

**Features** (start of quarter): FICO, DTI, original rate, loan age and age squared (centred), mark-to-market LTV (current UPB / (original value x HPI(q-1)/HPI(origination))), rate incentive, purpose and occupancy dummies, and macro variables dated q-1 (unemployment, 4-quarter unemployment change, HPI growth). VIF threshold 10; selected variables: credit_score, dti, orig_rate, age_yrs, age_yrs_sq, mtm_ltv, rate_incentive, unemp_lag, unemp_chg_4q, hpi_yoy, purpose_cashout, purpose_refi_nocash, occ_investor, occ_second. Original LTV was excluded because it changed sign once MTM LTV was included.

**Segmentation:** K-Means on standardised FICO, LTV, DTI, rate, fitted on training loans only; k = 3 by silhouette (0.252).

| Segment | Loans | Mean FICO | Mean LTV | Mean DTI | Train default rate / quarter |
|---|---|---|---|---|---|
| 0 | 17,608 | 702 | 78.2 | 39.7 | 0.914% |
| 1 | 8,538 | 759 | 43.1 | 30.5 | 0.135% |
| 2 | 23,679 | 763 | 77.7 | 32.1 | 0.078% |

**PD:** statsmodels GLM, Binomial family, logit link, cluster-robust standard errors by loan, fitted on the training period. Coefficients (per stated unit):

| Variable | Unit | Coef | Odds ratio | p-value | Expected sign | OK |
|---|---|---|---|---|---|---|
| const |  | -7.3751 | 0.001 | 0.0000 |  |  |
| credit_score | +100 points | -0.8214 | 0.440 | 0.0000 | - | yes |
| dti | +10 pts | 0.2168 | 1.242 | 0.0000 |  |  |
| orig_rate | +1 pp | 0.6174 | 1.854 | 0.0000 |  |  |
| age_yrs | +1 year | 0.1375 | 1.147 | 0.0000 |  |  |
| age_yrs_sq | +1 (centred) | -0.0145 | 0.986 | 0.0116 |  |  |
| mtm_ltv | +10 pts | 0.2679 | 1.307 | 0.0000 | + | yes |
| rate_incentive | +1 pp | -0.0759 | 0.927 | 0.1514 |  |  |
| unemp_lag | +1 pp | 0.0961 | 1.101 | 0.0001 | + | yes |
| unemp_chg_4q | +1 pp | 0.1292 | 1.138 | 0.0000 |  |  |
| hpi_yoy | +10 pp | -0.3489 | 0.705 | 0.0000 | - | yes |
| purpose_cashout | yes | 0.5916 | 1.807 | 0.0000 |  |  |
| purpose_refi_nocash | yes | 0.4127 | 1.511 | 0.0000 |  |  |
| occ_investor | yes | -0.0933 | 0.911 | 0.2974 |  |  |
| occ_second | yes | -0.0296 | 0.971 | 0.7793 |  |  |

**LGD:** LGD = P(liquidation | default) x severity. 3,747 default events: {'no_loss_exit': 1677, 'liquidated': 1495, 'open': 575}. 1,358 liquidated defaults have a usable loss; mean severity 51.5% (median 51.6%); P(liquidation | default) 44.7%; cure-adjusted mean LGD per default event 22.7%. Severity falls with house-price gains since origination (slope -0.0327 per +10 pp, 95% CI -0.0498 to -0.0180); the liquidation probability rises with MTM LTV.

| MTM LTV bucket | Resolved defaults | Liquidation rate | Mean severity | Cure-adjusted LGD (actual) | (model) |
|---|---|---|---|---|---|
| <60 | 699 | 22.0% | 44.0% | 9.7% | 10.4% |
| 60-80 | 1,004 | 40.6% | 54.2% | 22.0% | 20.4% |
| 80-100 | 1,100 | 59.7% | 54.3% | 32.4% | 29.5% |
| 100-120 | 232 | 59.9% | 38.8% | 23.2% | 37.7% |

**EAD:** scheduled amortisation from the current balance, rate and remaining term (level payment, no curtailment); start-of-quarter balance.

**Prepayment:** second logistic hazard (rate incentive, age, MTM LTV, FICO, HPI growth), 31,136 events, in-sample AUC 0.633.

**Macro paths and reversion:** Fed scenario for 13 quarters (reasonable and supportable period), then linear reversion over 4 quarters to the 1990-2025 mean (unemployment, mortgage rate, 10y yield) and to the mean HPI growth. ARIMA (below) is a sanity check and supports the mean-reverting extension.

**ECL:** `ECL = sum_t S(t-1) x h_d(t) x LGD(t) x EAD(t)` with `S(t) = prod (1 - h_d - h_p)`, up to 120 quarters, undiscounted.

**ARIMA (rolling-origin backtest, 1-8 quarters ahead, first origin 2000Q4):**

| Series | d | Order | Avg RMSE ARIMA | Random walk | Mean | vs random walk | vs mean |
|---|---|---|---|---|---|---|---|
| unemployment | 1 | (2, 1, 3) | 2.764 | 1.815 | 1.984 | -52.2% | -39.3% |
| hpi_growth | 0 | (2, 0, 2) | 5.839 | 6.602 | 6.563 | +11.6% | +11.0% |
| mortgage_rate | 1 | (1, 1, 0) | 0.849 | 0.864 | 3.881 | +1.7% | +78.1% |

ARIMA has weak forecast skill, as expected for macro series; it does not beat the random walk for unemployment. It is not used to drive the stress paths.

## 4. Assumptions and limitations

See `docs/assumptions.md` and `docs/limitations.md`. Key items: retail not wholesale; sample of 50,000 loans from 7 vintages; national macro variables; COVID forbearance excluded; absorbing default (no cures); no delinquency roll-rate model; undiscounted ECL; small, seasoned jump-off portfolio; LGD extrapolation beyond the observed house-price range.

## 5. Results

Jump-off portfolio at 2025Q4: 5,860 loans alive, starting UPB $705,053,490.

| Scenario | Lifetime ECL (% of UPB) | Lifetime ECL ($) | Lifetime cum. default rate | 12-month PD | Exposure-weighted LGD |
|---|---|---|---|---|---|
| Baseline | 0.022% | $156,899 | 0.81% | 0.182% | 3.1% |
| Severely adverse | 0.130% | $915,582 | 1.95% | 0.477% | 7.2% |

Lifetime ECL by K-Means segment (baseline / severely adverse): segment 0: 0.070% / 0.402%; segment 1: 0.011% / 0.049%; segment 2: 0.016% / 0.097%.

Absolute loss levels are small because the surviving portfolio is seasoned, high-FICO and carries large house-price gains (mean MTM LTV near 23%); the scenario ratio is the more informative number.

## 6. Benchmark vs XGBoost challenger

| Model | Sample | AUC | Gini | KS | Brier | Mean predicted | Mean actual |
|---|---|---|---|---|---|---|---|
| GLM | train | 0.827 | 0.654 | 0.508 | 0.00543 | 0.551% | 0.551% |
| GLM | validation | 0.890 | 0.779 | 0.666 | 0.00109 | 0.088% | 0.110% |
| GLM | oot | 0.786 | 0.572 | 0.446 | 0.00112 | 0.055% | 0.112% |
| XGBoost | train | 0.849 | 0.698 | 0.547 | 0.00540 | 0.551% | 0.551% |
| XGBoost | validation | 0.891 | 0.781 | 0.656 | 0.00109 | 0.141% | 0.110% |
| XGBoost | oot | 0.798 | 0.595 | 0.459 | 0.00112 | 0.099% | 0.112% |

Selection rule (validation only): champion = GLM unless XGBoost's validation AUC is more than 0.02 higher. Difference on validation: +0.0010, so the GLM is the champion. On the OOT set XGBoost is +0.0115 AUC points relative to the GLM.

## 7. Sensitivity and stress results

9-quarter horizon (2026Q1 to 2028Q1), loss = cumulative ECL / starting UPB.

| Scenario | 9-quarter loss rate | 9-quarter cum. default rate | Peak quarterly default rate | Peak quarter |
|---|---|---|---|---|
| Baseline | 0.0124% | 0.362% | 0.046% | 2026Q2 |
| Severely adverse | 0.0974% | 1.410% | 0.262% | 2027Q1 |

Severely adverse / baseline: **7.9x** (9-quarter loss), 5.7x (peak default rate).

9-quarter loss rate (% of starting UPB), peak unemployment (rows) x peak house-price decline (columns); monotone in both directions: True.

| Peak unemployment | HPI -10% | HPI -20% | HPI -30% | HPI -40% |
|---|---|---|---|---|
| 6% | 0.0221 | 0.0340 | 0.0552 | 0.0971 |
| 8% | 0.0286 | 0.0445 | 0.0729 | 0.1286 |
| 10% | 0.0375 | 0.0587 | 0.0966 | 0.1706 |
| 12% | 0.0494 | 0.0779 | 0.1285 | 0.2267 |

## 8. Back-testing and ongoing monitoring

- **Out-of-time (2017Q1-2019Q4):** GLM AUC 0.786, Gini 0.572, KS 0.446. Predicted default rate 0.055% vs actual 0.112% per quarter (the model under-predicts the level by a factor of 2.0; ranking is good, calibration is not); quarterly MAPE 43.3%.
- **Hosmer-Lemeshow style test:** chi2 154.3, p = 0. Segment binomial tests (below) also reject calibration: every segment is under-predicted.

| Segment | Loan-quarters | Defaults | Predicted | Binomial p |
|---|---|---|---|---|
| 0 | 30,924 | 118 | 58.4 | 5.56e-12 |
| 1 | 31,480 | 34 | 8.1 | 1.01e-11 |
| 2 | 164,691 | 102 | 58.1 | 1.77e-07 |

- **Forward back-test:** 13,820 loans alive at 2015Q4, realised macro, to 2019Q4: predicted cumulative default 0.86% (119 loans) vs actual 1.55% (214 loans); ratio 0.55.
- **PSI (training vs OOT):** 15 variables monitored (score + drivers), max PSI 7.70 (unemp_lag); 6 stable, 3 watch, 6 shifted. Macro variables shift by construction between the crisis-heavy training window and the benign OOT window.
- **COVID appendix:** 2020-2021 actual default rate 0.693% vs predicted 0.115%; forbearance-driven delinquency, not credit stress. Excluded from headline metrics.
- **Monitoring plan:** quarterly AUC/Gini/KS, predicted vs actual default rate and PSI (thresholds 0.10 / 0.25); annual recalibration and challenger review. See `docs/monitoring_report.md`.
- **Recommended remediation (not applied, to keep the OOT test clean):** recalibrate the intercept on a more recent window and re-test on a fresh hold-out.

## 9. Change log

| Version | Change |
|---|---|
| 0.1 | Initial development: panel, macro, features, PD/LGD/EAD, ECL, stress, validation. |
| 0.1 | Original LTV dropped from PD after the first fit (wrong sign with MTM LTV). |
| 0.1 | LGD severity re-specified on house-price change since origination (flat against MTM LTV). |
