# Assumptions

Living list. Each phase appends its own section. Numbers quoted here come from `results/*.json`.

## Phase 1: data and panel
- **Dataset:** Freddie Mac Single-Family Loan-Level Dataset, **Standard** sample files (not Non-Standard), user guide Release 48.
  Vintages used: 2004, 2006, 2007 (pre-crisis, so the 2008-2012 defaults are in the data) and 2012, 2014, 2016, 2018 (recent).
  Each sample file holds 50,000 loans (350,000 in total).
- **Cleaning (applied before sampling, one rule per loan, first failing rule wins):**
  1. credit score missing or outside 300-850 (9999 = not available)
  2. original LTV missing or outside 1-998 (999 = not available)
  3. original DTI missing or outside 0-65 (999 = not available; this also drops all HARP/relief-refinance loans, which have DTI = n/a)
  4. not fixed rate (none in the Standard dataset)
  5. missing or non-positive original UPB / rate / term
  Counts removed per rule are in `results/data_summary.json` -> `cleaning`.
- **Sampling:** after cleaning, an equal quota per vintage (50,000 / 7, remainder to the earliest vintages), chosen deterministically
  by hashing `loan_id` with `seed=42`. Result: exactly 50,000 loans.
- **Panel unit:** one row per loan per calendar quarter in which the loan starts the quarter alive and not yet defaulted.
  Features are the last monthly record of the *previous* quarter (start-of-quarter state); label is "event during this quarter".
- **Default** = first quarter in which the loan reaches 90+ days delinquent (delinquency status >= 3 months; REO acquisition "RA" is
  mapped to 99) OR terminates with zero-balance code 02 (third-party sale), 03 (short sale / charge-off), 09 (REO disposition),
  15 (whole-loan / note sale).
- **Default is absorbing.** A loan that later cures is not re-entered; rows stop after the default quarter. (Simplification: no cure/re-default model.)
- **Prepayment** = zero-balance code 01 (prepaid or matured) in a quarter without a prior default.
- **Other exits** (code 16 reperforming-loan securitisation, 96 defect before credit event) are treated as censored, not as default or prepayment.
- **Quarter timing:** calendar quarters from the monthly `period` (YYYYMM). Freddie's accounting cycle changed over time (user guide);
  we ignore this intra-month timing.
- **Loans that default in their origination quarter** have no start-of-quarter state and never enter the panel (negligible).
- **Delinquency status is not a model feature** (cannot be projected without a roll-rate model). It is kept in the panel only as `start_delq` for reporting.
- **Splits (by outcome quarter):** train <= 2014Q4, validation 2015Q1-2016Q4, out-of-time test 2017Q1-2019Q4.
  2020Q1-2021Q4 are labelled `covid` and excluded from all metrics; 2022 onward is labelled `other` (used for the jump-off state only).

## Phase 2: macro data, Fed scenarios, ARIMA
- **FRED series (5):** UNRATE, MORTGAGE30US, GS10 (quarter average), USSTHPI (FHFA all-transactions HPI, end-of-quarter level), GDPC1 (end-of-quarter level).
- **Fed 2026 scenarios:** Tables 3.A (baseline) and 4.A (severely adverse) of the final 2026 supervisory scenarios PDF, Q1 2026-Q1 2029, jump-off Q4 2025.
  The PDF exposes the data rows as text, not as table cells; they are parsed from the page text and the column order was confirmed against the table header cells.
- **Verified against the PDF text:** unemployment peaks at 10.0% in Q3 2027 (+5.5 pp from the 4.5% jump-off); the Fed HPI trough is 227 in Q4 2027;
  real GDP falls 4.6% from Q4 2025 to the Q2 2027 trough (reproduced to within rounding by compounding the published annualised growth rates).
- **Jump-off HPI proxy:** the Q4 2025 Fed HPI level is not in the tables. The baseline Q1 2026 level (325) is used as proxy, which gives a fall of 30.2%
  (the report says "about 30 percent").
- **HPI chaining:** the Fed HPI is not the FHFA index. The scenario paths use the Fed's quarterly HPI growth applied to the realised FHFA level at 2025Q4.
- **Reasonable-and-supportable period:** the 13 Fed quarters. After Q1 2029 unemployment, mortgage rate and 10y yield revert **linearly over 4 quarters** to their
  1990Q1-2025Q4 mean; HPI growth reverts linearly to its 1990Q1-2025Q4 mean quarterly log growth. Paths run 120 quarters (to 2055Q4).
- **ARIMA:** d from an ADF test (5%) on the pre-evaluation sample, (p,q) <= 3 chosen by AIC on that sample only; the order is then held fixed in an expanding-window
  rolling-origin backtest (first origin 2000Q4, 1-8 quarters ahead) against a random walk and an expanding-mean forecast. The 2020 shock is left in the sample.
  Used (a) to sanity-check the Fed baseline and (b) as support for the mean-reverting extension. It is not the driver of the stress paths.

## Phase 3: features and segmentation
- Features are measured at the start of the quarter. Macro features at outcome quarter q use macro dated q-1 (unemployment level, 4-quarter change in unemployment, HPI year-on-year growth, mortgage rate for the rate incentive).
- Mark-to-market LTV = current UPB / (original value x HPI(q-1) / HPI(origination quarter)), original value = original UPB / (LTV/100), HPI = FHFA USSTHPI (national).
- Age is centred before squaring (AGE_CENTER = 4 years) to avoid collinearity; VIF threshold 10 (all final VIFs <= 9).
- `purpose_refi_other` never occurs in this sample and is dropped by the VIF step (zero variance).
- K-Means on [FICO, LTV, DTI, rate], standardised, fitted on training-period loans only; k chosen by silhouette over k = 3..8 (k = 2 excluded as a trivial split), elbow reported. Segments are used for reporting, not as PD covariates (they are functions of covariates already in the model).

## Phase 4: PD model
- Logistic GLM (Binomial, logit), cluster-robust standard errors by loan, fitted on <= 2014Q4. Champion = GLM unless XGBoost beats its validation AUC by more than 0.02.
- Original LTV is excluded after the first fit: it had the wrong (negative) sign once MTM LTV was included, because MTM LTV already contains it.

## Phase 5: LGD and EAD
- LGD per default event = P(liquidation | default) x severity. 90+ DPD defaults that later prepay, are sold as reperforming loans or have a defect exit lose nothing.
  Defaults still open at the data cutoff are excluded (censored), which likely understates the liquidation probability.
- Severity = Actual Loss / zero-balance removal UPB over liquidated defaults, clipped to [0, 1]; modelled linearly on the house-price change since origination.
  It is not monotone in MTM LTV in this data (mortgage insurance recoveries on high-LTV loans). R-squared is low (about 2%).
- EAD = scheduled start-of-quarter balance (level-payment amortisation from the current balance, rate and remaining term, no curtailment).

## Phase 6-7: ECL and stress
- Prepayment hazard (second logistic) fitted on all non-COVID quarters; it is a projection model and not part of the OOT PD validation.
- ECL is undiscounted; defaults are charged on the start-of-quarter balance.
- 9-quarter horizon = 2026Q1-2028Q1. Sensitivity grid scales the Fed severely adverse paths to the target unemployment peak and HPI trough.
