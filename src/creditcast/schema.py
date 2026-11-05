"""Freddie Mac Single-Family Loan-Level Dataset schema.

Source: docs/reference/user_guide.pdf (Release 48, Oct 2026), Glossary section.
Both files are pipe-delimited with NO header row. Column order below is the file order.
Checked against the sample files: origination = 31 fields, monthly performance = 35 fields.
"""

# (name, duckdb type) in file order -------------------------------------------------------------
ORIG_COLUMNS = [
    ("credit_score", "INTEGER"),          # 300-850, 9999 = not available
    ("first_payment_date", "INTEGER"),    # YYYYMM
    ("first_time_homebuyer", "VARCHAR"),
    ("maturity_date", "INTEGER"),         # YYYYMM
    ("msa", "VARCHAR"),
    ("mi_pct", "INTEGER"),                # 999 = not available
    ("n_units", "INTEGER"),               # 99 = not available
    ("occupancy", "VARCHAR"),             # P / I / S / 9
    ("cltv", "DOUBLE"),                   # 999 = not available
    ("dti", "DOUBLE"),                    # 0-65, 999 = not available
    ("orig_upb", "DOUBLE"),
    ("ltv", "DOUBLE"),                    # 999 = not available
    ("orig_rate", "DOUBLE"),
    ("channel", "VARCHAR"),
    ("prepay_penalty", "VARCHAR"),
    ("amort_type", "VARCHAR"),            # FRM / ARM
    ("state", "VARCHAR"),
    ("property_type", "VARCHAR"),
    ("zip3", "VARCHAR"),
    ("loan_id", "VARCHAR"),               # PYYQnXXXXXXX
    ("loan_purpose", "VARCHAR"),          # P / C / N / R / 9
    ("orig_term", "INTEGER"),
    ("n_borrowers", "INTEGER"),
    ("seller", "VARCHAR"),
    ("super_conforming", "VARCHAR"),
    ("pre_harp_loan_id", "VARCHAR"),
    ("special_program", "VARCHAR"),
    ("harp_flag", "VARCHAR"),
    ("valuation_method", "VARCHAR"),
    ("io_flag", "VARCHAR"),
    ("vantage_score", "INTEGER"),
]

PERF_COLUMNS = [
    ("loan_id", "VARCHAR"),
    ("period", "INTEGER"),                # YYYYMM
    ("upb", "DOUBLE"),                    # current actual UPB
    ("delq_status", "VARCHAR"),           # 00,01,02,03.. (months), RA = REO acquisition, XX = n/a
    ("loan_age", "INTEGER"),
    ("remaining_months", "INTEGER"),
    ("defect_settlement_date", "VARCHAR"),
    ("mod_flag", "VARCHAR"),              # Y / P / null
    ("zb_code", "VARCHAR"),               # zero balance code (keep as text: leading zero)
    ("zb_effective_date", "VARCHAR"),
    ("interest_rate", "DOUBLE"),          # current rate
    ("non_int_upb", "DOUBLE"),
    ("ddlpi", "VARCHAR"),
    ("mi_recoveries", "DOUBLE"),          # recoveries are NEGATIVE, expenses/losses POSITIVE
    ("net_sale_proceeds", "DOUBLE"),
    ("non_mi_recoveries", "DOUBLE"),
    ("total_expenses", "DOUBLE"),
    ("legal_costs", "DOUBLE"),
    ("maintenance_costs", "DOUBLE"),
    ("taxes_insurance", "DOUBLE"),
    ("misc_expenses", "DOUBLE"),
    ("actual_loss", "DOUBLE"),            # positive = loss, negative = gain
    ("cum_mod_costs", "DOUBLE"),
    ("step_mod_flag", "VARCHAR"),
    ("payment_deferral_flag", "VARCHAR"),
    ("eltv", "DOUBLE"),                   # AVM estimated LTV, only from Apr 2017, 999 = n/a
    ("zb_removal_upb", "DOUBLE"),
    ("delinquent_accrued_interest", "DOUBLE"),
    ("delinquency_disaster_flag", "VARCHAR"),
    ("borrower_assistance_plan", "VARCHAR"),
    ("current_period_mod_cost", "DOUBLE"),
    ("int_bearing_upb", "DOUBLE"),
    ("mi_cancel_flag", "VARCHAR"),
    ("servicer_name", "VARCHAR"),
    ("bankruptcy_cramdown_costs", "DOUBLE"),
]

# Zero balance codes (user guide, "Zero Balance Code") -------------------------------------------
ZB_PREPAID = "01"                                    # prepaid or matured (voluntary payoff)
ZB_CREDIT_EVENT = ("02", "03", "09", "15")           # third party sale, short sale/charge off, REO, whole loan sale
ZB_OTHER_EXIT = ("16", "96")                         # reperforming loan securitisation, defect prior to credit event
# 02/03/09/15 are the codes for which the guide computes Actual Loss.

DEFAULT_DELQ_MONTHS = 3   # 90+ days delinquent = status >= 03 (months); "RA" mapped to 99
