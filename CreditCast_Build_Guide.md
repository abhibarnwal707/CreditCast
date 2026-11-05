# CreditCast: Build Guide (for Claude Code)

**Goal:** build a real, defensible credit-risk modelling project in about 2 weekends (fast path: 1 to 2 days), then walk away with every number your resume needs.
**Rule #1:** only put on the resume what the code produces. Every number comes from `make results`, never from memory or from a chat.

---

## 1. What is CreditCast?

**One sentence:** CreditCast answers the question a bank's risk team asks every quarter: *"If the economy gets worse, how much money will we lose on the loans we have made?"*

**The story in plain words**

1. A bank lends money to many borrowers. Some borrowers stop paying (**default**).
2. When a loan defaults, the bank does not lose everything. It recovers part by selling the house. What it loses is the **Loss Given Default (LGD)**.
3. Accounting rule **CECL** says: *today*, set aside reserve money for the **lifetime** expected losses of every loan you hold.
4. Regulators (the Federal Reserve's **stress test**, CCAR/DFAST) also ask: *"If a severe recession hits, how bad do losses get?"* The Fed publishes a hypothetical recession (**severely adverse scenario**) and every large bank must run its loan book through it.
5. JPMorgan's Wholesale Credit Risk QR team builds the statistical models behind both answers, then back-tests, monitors and documents them. That is your job description.

**CreditCast does exactly that pipeline** on public loan data (Freddie Mac mortgages), at small scale:

```
Freddie Mac loans (origination + monthly performance)
        |  clean, filter, build loan-quarter panel (SQL + Python)
        v
Loan features + macro (FRED: unemployment, house prices, mortgage rate)
        |  K-Means segments loans by risk profile
        v
PD model: logistic regression (GLM), discrete-time hazard     <- "chance this loan defaults next quarter"
LGD model: loss on defaulted loans (segment / LTV based)      <- "share of balance lost"
EAD: amortisation schedule                                    <- "balance outstanding at default"
        |
        v
ECL engine:  ECL = sum over future quarters of  S(t-1) x PD(t) x LGD(t) x EAD(t)
        |
        +--> Baseline scenario (Fed 2026)       -> CECL-style reserve estimate
        +--> Severely adverse scenario (Fed 2026) -> stress loss estimate (9-quarter loss)
        |
        v
Validation: out-of-time back-test, calibration, PSI monitoring, model documentation
```

### Glossary (learn these, you will be asked)

| Term | Meaning |
|---|---|
| **PD** | Probability of Default: chance a loan defaults in a period |
| **LGD** | Loss Given Default: fraction of the balance lost after recoveries |
| **EAD** | Exposure at Default: balance outstanding when default happens |
| **ECL** | Expected Credit Loss = PD x LGD x EAD, summed over the loan's life |
| **CECL** | Current Expected Credit Loss: the US accounting standard requiring lifetime expected losses today |
| **Stress test / CCAR / DFAST** | Fed exercise: project losses under a hypothetical severe recession |
| **Panel data** | Many loans observed over many time periods (loan x quarter rows) |
| **Discrete-time hazard model** | A logistic regression on loan-quarter rows where the label is "defaults this quarter, given survived so far". It gives a lifetime PD curve |
| **GLM** | Generalised Linear Model. Logistic regression is a GLM with a Binomial distribution and a logit link |
| **Out-of-time (OOT) test** | Train on earlier years, test on later years (never a random split in time series data) |
| **Back-testing** | Compare model predictions with what actually happened |
| **PSI** | Population Stability Index: measures how much a variable's distribution has shifted (below 0.1 stable, 0.1 to 0.25 watch, above 0.25 shifted) |
| **SR 11-7** | The Fed/OCC model-risk guidance that dictates how banks document and validate models |
| **Reasonable and supportable (R&S) forecast** | The forecast window CECL lets you trust (use the scenario), after which you revert to long-run averages |

### How it maps to the JPMC job description

| JD requirement | CreditCast component |
|---|---|
| Loss forecasting models for regulatory purposes | PD x LGD x EAD ECL engine, Fed 2026 scenarios |
| Back-testing, monitoring, annual review | Phase 8: OOT test, calibration, PSI, monitoring report |
| GLM, time series, clustering, decision trees, logistic regression | GLM PD model, ARIMA, K-Means, XGBoost challenger |
| Large panel datasets, cleaning, filtering, feature engineering | Freddie Mac loan-quarter panel in DuckDB SQL |
| Python, version control, testing, reproducibility | Git, pytest, seeds, config file, Makefile |
| Model documentation for regulatory review | SR 11-7 style development document |
| SQL | DuckDB SQL panel build |

### What it is NOT (say this yourself in interviews)

* It is a **retail mortgage** portfolio. The JPMC team models **wholesale** (corporate) loans. The method (PD, LGD, EAD, CECL, stress testing) is the same. The inputs differ (ratings instead of FICO, revolvers with undrawn commitments, very few defaults). Do not call it wholesale.
* It uses a **sample** of Freddie Mac loans, not the full 55 million.

---

## 2. Time plan

| Path | What | Time |
|---|---|---|
| **Fast path** (enough for the resume, defensible) | Phases 0 to 7, quick Phase 8 (OOT + calibration + PSI), short README | 1 to 2 days with Claude Code |
| **Full path** | + prepayment model, sensitivity grid, SR 11-7 document, tests, CI, Streamlit demo | 2 weekends |

**Your job vs Claude Code's job**

* **You:** register and download data, verify the Fed scenario numbers by eye, read every plot and table, make sure you can explain every step.
* **Claude Code:** write the code, tests and docs, run the pipeline.
* After each phase, ask Claude Code: *"Explain what you built as if I'm in an interview, then quiz me with 5 questions."* If you cannot answer, do not move on.

---

## 3. Phase 0: do this yourself before opening Claude Code (about 1 hour)

1. **Freddie Mac loan-level data** (free, non-commercial use). Register at <https://freddiemac.com/research/datasets/sf-loanlevel-dataset> (access goes through Clarity Data Intelligence) and accept the terms.
   * Download the **sample files** for several origination years. Aim for a mix: some pre-crisis years (for example 2004 to 2008, so the 2008 to 2012 defaults are in the data) and some recent years (2012 to 2023). In the past the sample files held about 50,000 loans per origination year; **confirm the count when you download**. You will subsample to a total of about 50,000 loans.
   * Also download the **User Guide / file layout PDF**. Files are **pipe-delimited with no header row**, so Claude Code must take column names and zero-balance codes from that PDF, not guess.
2. **FRED API key** (free): <https://fred.stlouisfed.org> > My Account > API Keys. Series you will use:
   * `UNRATE` unemployment rate (monthly)
   * `USSTHPI` FHFA house price index, quarterly (or `CSUSHPINSA` Case-Shiller national)
   * `MORTGAGE30US` 30-year fixed mortgage rate (weekly)
   * `GDPC1` real GDP (quarterly), `GS10` 10-year Treasury yield (monthly)
3. **Fed 2026 supervisory scenarios** PDF: <https://www.federalreserve.gov/publications/files/2026-final-supervisory-stress-test-scenarios-20260204.pdf>
   * Baseline and severely adverse paths for 28 variables, **13 quarters (Q1 2026 to Q1 2029)**, jump-off Q4 2025. The variable tables are at the end of the PDF.
   * Variables you need: unemployment rate, house prices, real GDP, conforming 30-year mortgage rate, 10-year Treasury yield.
   * **Check these figures against the PDF yourself** because they are on your resume: unemployment peaks at **10%** (Q3 2027), house prices fall about **30%** from Q4 2025, real GDP falls **4.6%** (Q4 2025 to the Q2 2027 trough).
4. **Tools:** Python 3.11+, about 10 GB free disk. Create the (empty) repo `abhibarnwal707/CreditCast` locally with `git init`.
5. **Do not commit raw Freddie Mac files** (large, and the data has terms of use). Commit code, config, and small aggregated outputs only.

---

## 4. Paste this into `CLAUDE.md` at the repo root

```markdown
# CreditCast: project rules

Purpose: CECL-style expected credit loss + Fed 2026 stress testing on Freddie Mac mortgage loans.
Portfolio: ~50,000 loans sampled across origination years; retail mortgages (NOT wholesale).

## Non-negotiable rules
1. Never hard-code or invent a result. Every number on the resume must come from code and be saved in results/*.json.
2. Fixed random seed (config.yaml: seed=42) everywhere. `make all` reproduces everything from raw data.
3. Time-based splits only. Train <= 2014Q4, validation 2015Q1-2016Q4, out-of-time test 2017Q1-2019Q4.
   COVID forbearance (2020-2021) distorts delinquency, so exclude it from metrics and document it.
4. No leakage: features are measured at the START of the quarter; the label is "defaults during this quarter".
   Fit scalers, K-Means and any transform on the training period only.
5. Read column names, zero-balance codes and loss fields from the Freddie Mac user guide in docs/reference/.
   Never guess a column. Store the schema in src/creditcast/schema.py.
6. Do not commit raw data. data/ is in .gitignore. Commit only small aggregated outputs and code.
7. Every assumption goes into docs/assumptions.md. Every limitation goes into docs/limitations.md.
8. Write pytest tests for every non-trivial function. Keep `pytest` green.
9. Use DuckDB SQL (files in sql/) for the panel build; pandas/statsmodels/scikit-learn for modelling.
10. Print row counts and default counts after every data step so I can sanity-check.

## Definitions (verify against the user guide)
- Default = first quarter in which a loan reaches 90+ days delinquent, or terminates with a
  credit-event zero-balance code (third-party sale, short sale, REO disposition, note sale).
- Prepayment = zero-balance code for prepaid/matured.
- Risk set = loans still alive and not yet defaulted at the start of the quarter.
- Delinquency status is NOT used as a model feature (it cannot be projected without a roll-rate model); list as a limitation.
```

Suggested layout:

```
CreditCast/
  CLAUDE.md  README.md  Makefile  config.yaml  pyproject.toml  .gitignore
  sql/            build_panel.sql, features.sql
  src/creditcast/ schema.py data.py macro.py segment.py pd_model.py lgd_ead.py ecl.py stress.py validate.py monitor.py
  tests/
  docs/           assumptions.md limitations.md model_development_document.md reference/
  results/        data_summary.json pd_metrics.json lgd.json arima.json ecl.json stress.json backtest.json psi.json RESUME_NUMBERS.md
  app/streamlit_app.py
  data/           raw/ interim/ external/   (gitignored)
```

> Tip: start Claude Code in plan mode for each phase. Ask for the plan first, read it, then say "go".

---

## 5. Phases (paste each prompt into Claude Code)

### Phase 1: Load data and build the loan-quarter panel

```text
Phase 1. Read docs/reference/ (Freddie Mac user guide) and create src/creditcast/schema.py with the
column names for the origination and monthly performance files (pipe-delimited, no header).
Load the sample files for the origination years in data/raw/ into DuckDB. Build a stratified
random sample of ~50,000 loans across origination years (config.yaml: n_loans=50000, seed=42),
keeping both pre-crisis and recent vintages.
Write sql/build_panel.sql that creates a loan-QUARTER panel: one row per loan per calendar quarter
while the loan is alive and not yet defaulted. Aggregate monthly rows to quarters.
Clean and filter: remove invalid or missing credit score / LTV / DTI codes, document every rule
and the number of loans/rows each rule removes.
Define default and prepayment as in CLAUDE.md. Save results/data_summary.json with: loans, loan-quarters,
default events, prepayment events, vintage range, observation window, and default rate by origination year.
Plot default rate by calendar quarter (the 2008-2012 crisis bump must be visible).
Write tests for the default-flag logic on small synthetic delinquency sequences.
```

**Done when:** you can see the 2008 to 2012 default spike; you know the exact number of loans, loan-quarters and default events; the removed-row counts are documented.
**Watch out:** if default events in the training window are below about 500, tell Claude Code to include more pre-crisis vintages.

### Phase 2: Macro data, Fed scenarios, ARIMA

```text
Phase 2. Using the FRED API (key in .env, never committed) pull UNRATE, USSTHPI, MORTGAGE30US, GDPC1, GS10.
Convert to quarterly (quarter average for rates, end-of-quarter for index levels) and save to data/external/.
Load the Fed 2026 baseline and severely adverse paths (Q1 2026-Q1 2029, jump-off Q4 2025) from the PDF into
data/external/fed_2026_scenarios.csv. Extract the tables with pdfplumber and print them so I can verify them by eye.
Add a test that asserts: severely adverse unemployment peaks at 10% and is above baseline in every quarter;
the house price index falls ~30% from Q4 2025; real GDP falls ~4.6%.
Time series: for unemployment, HPI growth and mortgage rate run ADF stationarity tests, choose ARIMA orders by AIC,
and evaluate with an expanding-window rolling-origin backtest (1 to 8 quarters ahead) against a random-walk and
a mean forecast. Save RMSE and the improvement over naive in results/arima.json.
Use ARIMA for two things: (a) a data-driven baseline to sanity-check against the Fed baseline, and (b) extending the
macro path beyond Q1 2029 with mean reversion (loans live much longer than 13 quarters).
Document the reversion rule (for example linear reversion to the historical mean over 4 quarters) in docs/assumptions.md.
```

**Understand this:** loans last up to 30 years but the Fed gives 13 quarters. CECL calls the trusted period "reasonable and supportable"; after it you revert to long-run averages.

### Phase 3: Features and K-Means segmentation

```text
Phase 3. Build sql/features.sql / src/creditcast/data.py loan-quarter features measured at the START of each quarter:
credit score, original LTV, DTI, original interest rate, loan age (and age squared), loan purpose, occupancy,
mark-to-market LTV = current UPB / (original property value x HPI_t / HPI_at_origination), where original value =
original UPB / (LTV/100), rate incentive = original rate - current 30-year mortgage rate, plus macro: unemployment rate,
unemployment change, HPI year-over-year growth.
Segmentation: standardise origination features (credit score, LTV, DTI, rate), fit K-Means on the TRAINING period only,
choose k with the elbow method and silhouette score (try k=2..8), save k and the silhouette in results/segments.json and
profile each segment (mean FICO, LTV, DTI, default rate). Add the segment as a feature and use it for reporting.
Check for multicollinearity with VIF and drop or combine variables that exceed 10. Test that no feature uses future information.
```

### Phase 4: PD model (logistic GLM, discrete-time hazard)

```text
Phase 4. Fit the PD model: statsmodels GLM, Binomial family, logit link, on the training period loan-quarter panel,
label = default during the quarter. Cluster-robust standard errors by loan.
Save the coefficient table with odds ratios, p-values and sign checks (FICO down => PD up is wrong; assert expected signs:
higher FICO lowers PD, higher LTV / MTM LTV raises PD, higher unemployment raises PD, higher HPI growth lowers PD).
Fit an XGBoost challenger with the same features and compare. Report AUC, Gini (= 2*AUC - 1) and KS on train, validation
and out-of-time test. Compute a calibration table and plot by decile (predicted vs actual default rate).
Select the final model on the validation period only; do not touch the OOT test set until the end.
Save results/pd_metrics.json. Write tests (probabilities between 0 and 1, expected coefficient signs).
```

**Understand this:** logit link means log-odds are linear in the features. Be ready to explain deviance, odds ratios and why a time-based split matters.

### Phase 5: LGD and EAD

```text
Phase 5. LGD: for defaulted loans that completed liquidation (credit-event zero-balance codes) compute realised loss
from the Freddie Mac loss fields (net sales proceeds, expenses, recoveries, actual loss; follow the user guide), divided
by UPB at default. Report the number of defaults with a usable LGD, mean LGD, and LGD by mark-to-market-LTV bucket.
Model LGD with simple, robust methods (LTV-bucket averages or a regression on MTM LTV at default). LGD must rise as house
prices fall (so it reacts to the stress scenario). Report uncertainty because the sample is small.
EAD: project UPB with the standard amortisation formula from the loan's current balance, rate and remaining term.
Test that the schedule amortises to zero and that EAD never exceeds the current UPB.
Save results/lgd.json.
```

**Understand this:** for a mortgage EAD is just the scheduled balance. For a corporate revolver EAD includes a share of the undrawn commitment (credit conversion factor). Say that if asked.

### Phase 6: ECL engine (CECL-style)

```text
Phase 6. Build src/creditcast/ecl.py. For every loan alive at the jump-off date (Q4 2025) project quarter by quarter until
maturity (cap at 120 quarters):
  macro path = Fed scenario for the first 13 quarters, then reversion to long-run (from Phase 2)
  hazard h(t) from the PD model, survival S(t) = product of (1 - h(s)) for s<=t
  ECL = sum over t of S(t-1) x h(t) x LGD(t) x EAD(t)
Prepayment (recommended): fit a second logistic hazard for prepayment (label = prepaid quarter, features: rate incentive,
age, MTM LTV) and use S(t) = product of (1 - h_default - h_prepay). If you skip it, document that lifetime loss is
overstated and list it as a limitation.
Report: lifetime ECL as % of starting UPB under the baseline scenario, ECL by K-Means segment, and the lifetime PD curve.
Test: survival is between 0 and 1 and non-increasing; ECL >= 0; ECL under severely adverse >= baseline.
Save results/ecl.json.
```

### Phase 7: Stress testing (Fed 2026 severely adverse)

```text
Phase 7. Run the portfolio through the Fed baseline and severely adverse scenarios.
Report: 9-quarter cumulative loss as % of starting UPB (the standard CCAR planning horizon: Q1 2026 to Q1 2028), peak quarterly
default rate and when it occurs, the ratio severely adverse / baseline, and the loss path by quarter.
Also run a sensitivity grid: peak unemployment (for example 6, 8, 10, 12%) x house price decline (10, 20, 30, 40%),
and plot the 9-quarter loss rate as a heatmap. Check monotonicity: more stress means more loss in every cell.
Save results/stress.json and the figures.
```

### Phase 8: Validation, back-testing and monitoring

```text
Phase 8. Out-of-time validation on 2017Q1-2019Q4 (the first time the test set is used):
 - AUC, Gini, KS, Brier score
 - calibration by decile and by K-Means segment, binomial test per segment, Hosmer-Lemeshow style test
 - predicted vs actual quarterly default rate using realised macro, with mean absolute percentage error
Forecast-from-origin back-test: take loans alive at 2015Q4, forecast cumulative default 2016-2019 with realised macro,
and compare with what actually happened.
Monitoring: PSI of the predicted hazard score and of each key feature, training vs OOT, and rolling quarterly AUC.
Flag PSI > 0.1 (watch) and > 0.25 (shifted). Write results/backtest.json, results/psi.json and a one-page monitoring report.
Add a COVID appendix: show how performance changes in 2020-2021 and explain forbearance (do not hide it).
```

### Phase 9: Model development document (SR 11-7 style)

```text
Phase 9. Write docs/model_development_document.md (6 to 10 pages; also export to PDF with pandoc if available):
1 Purpose and scope  2 Data (sources, filters, sample, definitions)  3 Methodology (PD, LGD, EAD, ECL, scenarios, reversion)
4 Assumptions and limitations  5 Results (tables from results/*.json)  6 Benchmark vs XGBoost challenger
7 Sensitivity and stress results  8 Back-testing and ongoing monitoring plan (metrics, thresholds, frequency)  9 Change log.
Every number must be pulled from results/*.json, not typed by hand.
```

### Phase 10: Engineering polish and demo

```text
Phase 10. Add: Makefile (make data features train validate ecl stress report all), config.yaml, pyproject.toml,
pytest suite (default logic, LGD calc, amortisation, survival bounds, PSI of identical samples is 0, scenario monotonicity),
a GitHub Actions workflow that runs pytest, and a README with: project summary, architecture diagram, results table (from
results/*.json), how to reproduce, limitations.
Streamlit app (app/streamlit_app.py) that reads only small pre-computed files from results/ (no raw data): pick the scenario,
move unemployment and house-price sliders from the grid, see the loss curve and segment breakdown.
Make sure no raw data is tracked by git.
```

**Done when:** `git clone`, then `make all`, then `pytest` all work from a clean checkout; the Streamlit app runs locally; deploy it free on Streamlit Community Cloud if you want a Demo link.

### Phase 11: Resume harvest (the last prompt)

```text
Phase 11. Create results/RESUME_NUMBERS.md with exactly the fields in the "Resume Harvest" table of CreditCast_Build_Guide.md.
Each value must be read from results/*.json by script (make resume_numbers). Print "NOT BUILT" for anything we did not build.
Also output: the exact list of Python/SQL libraries used, the number of pytest tests, the number of pages of the model
development document, and the GitHub repo URL.
```

---

## 6. ⭐ RESUME HARVEST: everything I need at the end ⭐

> **Send me the contents of `results/RESUME_NUMBERS.md`.** With those numbers I will turn CreditCast from "In Progress" into a complete project, add the GitHub link (and the Demo link if deployed), and write the cover letter. Anything marked NOT BUILT stays off the resume.

### 6.1 Numbers to capture

| # | Item | Where it comes from | Resume use |
|---|---|---|---|
| 1 | Number of loans (about 50,000), loan-quarters, vintage range | `data_summary.json` | "on 50,000 Freddie Mac loans (X M loan-quarters)" |
| 2 | Default events (train / OOT) | `data_summary.json` | credibility of sample size |
| 3 | PD model: **OOT AUC, Gini, KS** | `pd_metrics.json` | headline performance |
| 4 | XGBoost challenger AUC vs GLM | `pd_metrics.json` | "benchmarked against XGBoost" |
| 5 | Mean LGD and number of defaults with LGD | `lgd.json` | LGD bullet |
| 6 | **Lifetime ECL rate (% of UPB), baseline** | `ecl.json` | CECL result |
| 7 | **9-quarter loss rate: baseline vs severely adverse, ratio** | `stress.json` | stress-testing result |
| 8 | Peak quarterly default rate under stress | `stress.json` | optional |
| 9 | K-Means k and silhouette score | `segments.json` | segmentation |
| 10 | ARIMA orders and RMSE improvement vs naive (%) | `arima.json` | time-series |
| 11 | **Predicted vs actual default rate OOT** (and MAPE) | `backtest.json` | back-testing |
| 12 | Max PSI and number of variables monitored | `psi.json` | monitoring |
| 13 | Number of FRED series used | Phase 2 | "FRED indicators" |
| 14 | Number of pytest tests, CI status | Phase 10 | engineering hygiene |
| 15 | Model document page count | Phase 9 | documentation |
| 16 | **GitHub link**, Demo link (only if deployed) | repo | links on the resume |

### 6.2 Resume claim, and what must exist before it is allowed

| Claim on the current resume | Must exist in the repo |
|---|---|
| "50,000 Freddie Mac mortgage loans" | Phase 1 sample, exact N in `data_summary.json` |
| "PD, LGD, EAD, CECL-style ECL" | Phases 4 to 6 |
| "Logistic Regression (GLM)" | statsmodels Binomial GLM, logit link |
| "FRED economic indicators" | Phase 2 series |
| "ARIMA macroeconomic forecasting" | Phase 2, with a rolling-origin evaluation |
| "K-Means portfolio segmentation" | Phase 3 |
| "Fed 2026 baseline and severely adverse scenarios" | Phases 2 and 7 |
| "10% unemployment, 30% house price decline, 4.6% GDP contraction" | verified by test against the Fed PDF |
| "Out-of-time validation" | Phase 8 test on 2017 to 2019 |
| "PSI monitoring" | Phase 8 |
| "Model documentation" | Phase 9 |

### 6.3 Bullet templates (I will fill these with your real numbers)

* Built a CECL-style Expected Credit Loss framework on **50,000** Freddie Mac mortgage loans (**⟦X M⟧** loan-quarters), estimating PD, LGD (**⟦mean LGD⟧%**) and EAD with loan-level SQL/Python preprocessing and K-Means segmentation (**⟦k⟧** segments).
* Developed a macro-driven discrete-time logistic regression (GLM) PD model with FRED indicators and ARIMA forecasts, reaching out-of-time Gini **⟦x⟧** / KS **⟦y⟧** and beating an XGBoost challenger or matching it within **⟦z⟧** AUC points.
* Stress tested the portfolio under Federal Reserve 2026 scenarios: 9-quarter loss rate **⟦a%⟧** baseline vs **⟦b%⟧** severely adverse (**⟦b/a⟧x**); back-tested predicted vs actual default rate (**⟦p%⟧ vs ⟦q%⟧**), monitored PSI (max **⟦r⟧**) and documented the model in a **⟦n⟧**-page SR 11-7 style report with **⟦t⟧** pytest tests.

### 6.4 Final checklist before the link goes on the resume

- [ ] `git clone` into a fresh folder, then `make all` and `pytest` both succeed
- [ ] README has a results table and a "Limitations" section
- [ ] No raw data committed (`git ls-files | grep data/raw` returns nothing)
- [ ] Repo is public and not empty
- [ ] I can explain every bullet without notes
- [ ] `results/RESUME_NUMBERS.md` has no "NOT BUILT" rows for anything claimed on the resume

---

## 7. Interview prep (answer these out loud)

1. **What is CECL, and how is it different from the old incurred-loss approach?** Lifetime expected loss now, forward-looking, with a reasonable and supportable forecast and then reversion.
2. **Why logistic regression / GLM?** Interpretable, regulatory-standard; log-odds linear in features; know the link function, deviance and odds ratios.
3. **Why a discrete-time hazard instead of a single 12-month PD?** You need lifetime PD for CECL; it handles censoring and prepayment.
4. **How did you define default, and why?** 90+ days or credit-event termination; the reasoning behind each code.
5. **How did you avoid leakage?** Start-of-quarter features, time-based splits, K-Means and scalers fitted on training data only.
6. **Why OOT rather than a random split?** Time dependence and regime change; a random split overstates performance.
7. **AUC is high; is the model good?** Not necessarily: Gini and KS measure ranking, calibration measures levels; you need both.
8. **How does the macro economy enter the model?** Unemployment, HPI growth and mortgage-LTV; limitation: national, not regional, house prices.
9. **Why ARIMA, and how good is it?** Stationarity and differencing; weak forecast skill; used for sanity checks and mean-reverting extension.
10. **What is PSI and what thresholds do you use?** Below 0.1 stable, 0.1 to 0.25 watch, above 0.25 shifted.
11. **How would this change for a wholesale portfolio?** Rating-based PD, LGD by seniority/collateral, EAD with credit conversion factors on revolvers, few defaults (low-default portfolio), correlation and concentration.
12. **What are the limitations?** Retail not wholesale, a sample of about 50,000, national HPI, COVID forbearance, no modifications, no delinquency-state model.

---

## 8. Common pitfalls

* **Leakage:** using end-of-quarter delinquency or fitting K-Means on all data.
* **Too few defaults:** include pre-crisis vintages and check event counts in Phase 1.
* **COVID forbearance:** 2020 to 2021 delinquency is not credit stress; exclude it from metrics and say so.
* **Mixing Standard and Non-Standard Freddie files:** pick one on purpose and record it.
* **Loading huge files at once:** stream with DuckDB; do not read everything into pandas.
* **Unverified Fed numbers:** the 10% / 30% / 4.6% figures are on your resume. Check them against the PDF.
* **Claiming what is not built:** if prepayment, the Streamlit demo or the document is skipped, tell me and I will keep it off the resume.

---

**Sources**
* Freddie Mac Single-Family Loan-Level Dataset: <https://freddiemac.com/research/datasets/sf-loanlevel-dataset>
* Freddie Mac SF LLD user guide: <https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf>
* Fed 2026 final supervisory stress test scenarios: <https://www.federalreserve.gov/publications/files/2026-final-supervisory-stress-test-scenarios-20260204.pdf>
* FRED: <https://fred.stlouisfed.org>
