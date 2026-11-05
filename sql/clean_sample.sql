-- Stage 1: clean origination data, draw the stratified sample, extract monthly performance.
-- Inputs (created by creditcast.data): views orig_raw, perf_raw (all raw sample files) and
-- table params(n_loans, seed).
-- Outputs: tables orig_clean_flags, loans, perf_sample, perf, losses.

-- Every loan gets AT MOST ONE removal reason (the first rule that fails, in this order).
CREATE OR REPLACE TABLE orig_clean_flags AS
SELECT
    *,
    CASE WHEN CAST(substr(loan_id, 2, 2) AS INTEGER) >= 99 THEN 1900 ELSE 2000 END
        + CAST(substr(loan_id, 2, 2) AS INTEGER) AS vintage,
    CASE
        WHEN credit_score IS NULL OR credit_score NOT BETWEEN 300 AND 850 THEN 'r1_credit_score_missing_or_invalid'
        WHEN ltv IS NULL OR ltv NOT BETWEEN 1 AND 998                      THEN 'r2_ltv_missing_or_invalid'
        WHEN dti IS NULL OR dti NOT BETWEEN 0 AND 65                       THEN 'r3_dti_missing_or_invalid'
        WHEN amort_type IS DISTINCT FROM 'FRM'                             THEN 'r4_not_fixed_rate'
        WHEN orig_upb IS NULL OR orig_upb <= 0
          OR orig_rate IS NULL OR orig_rate <= 0
          OR orig_term IS NULL OR orig_term <= 0                           THEN 'r5_missing_upb_rate_or_term'
        ELSE NULL
    END AS removal_rule
FROM orig_raw;

-- Stratified sample: equal quota per origination vintage (remainder goes to the earliest vintages),
-- chosen deterministically by hashing loan_id with the seed.
CREATE OR REPLACE TABLE loans AS
WITH vint AS (
    SELECT vintage, dense_rank() OVER (ORDER BY vintage) AS vrank
    FROM (SELECT DISTINCT vintage FROM orig_clean_flags WHERE removal_rule IS NULL)
),
nv AS (SELECT count(*) AS n FROM vint),
ranked AS (
    SELECT o.*, v.vrank,
           row_number() OVER (
               PARTITION BY o.vintage
               ORDER BY hash(o.loan_id || '-' || CAST(p.seed AS VARCHAR))
           ) AS rn
    FROM orig_clean_flags o
    JOIN vint v USING (vintage)
    CROSS JOIN params p
    WHERE o.removal_rule IS NULL
)
SELECT r.* EXCLUDE (removal_rule, rn, vrank)
FROM ranked r CROSS JOIN params p CROSS JOIN nv
WHERE r.rn <= p.n_loans // nv.n + CASE WHEN r.vrank <= p.n_loans % nv.n THEN 1 ELSE 0 END;

-- Monthly performance rows for the sampled loans only (materialised once).
CREATE OR REPLACE TABLE perf_sample AS
SELECT * FROM perf_raw WHERE loan_id IN (SELECT loan_id FROM loans);

-- Slim monthly table consumed by build_panel.sql.
-- delq: months delinquent (00,01,02,03,...). RA (REO acquisition) -> 99; XX (not available) -> NULL.
CREATE OR REPLACE TABLE perf AS
SELECT
    loan_id,
    period,
    (period // 100) * 4 + ((period % 100) - 1) // 3 AS qidx,
    upb,
    CASE WHEN delq_status = 'RA' THEN 99
         WHEN delq_status = 'XX' THEN NULL
         ELSE TRY_CAST(delq_status AS INTEGER) END AS delq,
    loan_age AS age,
    remaining_months AS rem_months,
    zb_code,
    interest_rate AS rate,
    NULLIF(eltv, 999) AS eltv,
    mod_flag
FROM perf_sample;

-- Loss fields at disposition (used for LGD in Phase 5). Recoveries negative, expenses positive.
CREATE OR REPLACE TABLE losses AS
SELECT loan_id, period AS zb_period, zb_code, zb_removal_upb, net_sale_proceeds,
       mi_recoveries, non_mi_recoveries, total_expenses, delinquent_accrued_interest, actual_loss
FROM perf_sample
WHERE zb_code IN ('02', '03', '09', '15');
