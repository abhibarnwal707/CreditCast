# Limitations

- **Retail mortgages, not wholesale.** Same methodology (PD, LGD, EAD, CECL, stress testing) but different inputs (ratings, revolvers, low-default portfolios). Not a wholesale portfolio.
- **Sample, not the full dataset:** 50,000 loans drawn from 350,000 sample loans across 7 vintages (2004, 2006, 2007, 2012, 2014, 2016, 2018). Vintages in between are not represented.
- **Few out-of-time defaults.** Validation and OOT windows (2015-2019) are benign; default counts there are small (see `results/data_summary.json` -> `by_split`), so OOT metrics carry wide uncertainty.
- **COVID forbearance (2020-2021):** 90+ day delinquency in that period reflects payment relief, not credit stress. It is excluded from metrics and shown in a separate appendix.
  The 2020 spike also inflates "ever defaulted" rates for the 2016 and 2018 vintages.
- **Default definition:** 90+ DPD or credit-event termination, absorbing; cures and re-defaults, modifications and payment deferrals are not modelled.
- **No delinquency-state (roll-rate) model**, so delinquency status is not a predictor.
- **National macro variables only** (no regional house prices or unemployment).
- **Freddie Mac Standard dataset only** (fully amortising fixed-rate, full-documentation loans; excludes ARMs, FHA/VA, etc.).
- **Survivorship from the vintage design:** pre-crisis vintages are observed from origination; later vintages only after 2012. Calendar-time defaults are therefore a mix of vintages.
- **Jump-off portfolio is small and seasoned.** Only about 5,900 of the 50,000 loans are still alive at 2025Q4, with a mean MTM LTV near 23% after large house-price gains, so absolute loss rates are small (stress losses are a fraction of 1% of UPB). Ratios between scenarios are more informative than levels.
- **LGD extrapolation:** at the jump-off, house-price gains since origination (often +100%) lie outside most of the severity training range, so low-LGD predictions are extrapolated. Severity has a low R-squared.
- **Open defaults excluded from LGD** (575 loans still delinquent at the cutoff).
- **Prepayment:** fitted on all non-COVID quarters and not validated out of time.
- **Undiscounted ECL**, no curtailments in EAD, no cure/re-default dynamics.
