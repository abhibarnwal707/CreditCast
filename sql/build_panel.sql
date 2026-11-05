-- Stage 2: loan-QUARTER panel with default / prepayment labels.
-- Inputs: table loans(loan_id), table perf(loan_id, period, qidx, upb, delq, age, rem_months,
--         zb_code, rate, eltv, mod_flag)  [one row per loan-month].
-- Output: table loan_events, qstate, panel.
--
-- Design (see docs/assumptions.md):
--   * One panel row per loan per calendar quarter q in which the loan starts the quarter ALIVE and
--     NOT YET DEFAULTED (risk set).
--   * Features come from the LAST monthly record of quarter q-1 (= state at start of quarter q): no leakage.
--   * default = 1 if the loan first reaches 90+ days delinquent (delq >= 3, RA = 99) OR terminates with a
--     credit-event zero-balance code (02,03,09,15) during quarter q.
--   * Default is absorbing: later cures are ignored, no rows after the default quarter.
--   * prepay = 1 if the loan terminates with ZB code 01 in quarter q (and did not default first).
--   * exit_other = 1 if it terminates with 16/96 (treated as censored).

CREATE OR REPLACE TABLE loan_events AS
SELECT
    loan_id,
    min(CASE WHEN delq >= 3 OR zb_code IN ('02', '03', '09', '15') THEN qidx END) AS default_q,
    min(CASE WHEN zb_code = '01' THEN qidx END)           AS prepay_q,
    min(CASE WHEN zb_code IN ('16', '96') THEN qidx END)  AS other_exit_q,
    max(qidx)                                             AS last_q
FROM perf
GROUP BY loan_id;

-- State at the end of each quarter = last monthly record in that quarter.
CREATE OR REPLACE TABLE qstate AS
SELECT * EXCLUDE (rn) FROM (
    SELECT *, row_number() OVER (PARTITION BY loan_id, qidx ORDER BY period DESC) AS rn
    FROM perf
) WHERE rn = 1;

CREATE OR REPLACE TABLE panel AS
SELECT
    s.loan_id,
    s.qidx + 1                          AS qidx,         -- the quarter being predicted
    s.upb                               AS start_upb,
    s.age                               AS start_age,
    s.rem_months                        AS start_rem_months,
    s.rate                              AS start_rate,
    s.eltv                              AS start_eltv,
    s.delq                              AS start_delq,   -- NOT a model feature (see CLAUDE.md)
    s.mod_flag                          AS start_mod_flag,
    CASE WHEN e.default_q = s.qidx + 1 THEN 1 ELSE 0 END                                        AS "default",
    CASE WHEN e.prepay_q = s.qidx + 1
              AND (e.default_q IS NULL OR e.default_q > s.qidx + 1) THEN 1 ELSE 0 END           AS prepay,
    CASE WHEN e.other_exit_q = s.qidx + 1
              AND (e.default_q IS NULL OR e.default_q > s.qidx + 1) THEN 1 ELSE 0 END           AS exit_other
FROM qstate s
JOIN loan_events e USING (loan_id)
WHERE s.upb > 0                                         -- still has a balance at start of quarter
  AND s.zb_code IS NULL                                 -- not terminated
  AND s.qidx < e.last_q                                 -- loan is observed in the next quarter
  AND (e.default_q IS NULL OR e.default_q > s.qidx);    -- not already defaulted
