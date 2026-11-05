"""No-look-ahead tests for the feature builder."""
import numpy as np
import pandas as pd

from creditcast.features import CANDIDATES, build_features


def toy():
    # one loan originated 2006Q1 (id F06Q1...), observed in quarters 2007Q1..2007Q3
    loans = pd.DataFrame({"loan_id": ["F06Q1000001"], "vintage": [2006], "credit_score": [700], "ltv": [80.0], "dti": [35.0],
                          "orig_rate": [6.5], "orig_upb": [160000.0], "loan_purpose": ["P"], "occupancy": ["P"]})
    q0 = 2007 * 4
    panel = pd.DataFrame({"loan_id": "F06Q1000001", "qidx": [q0, q0 + 1, q0 + 2], "start_upb": [150000.0, 149000.0, 148000.0],
                          "start_age": [12, 15, 18], "default": 0, "prepay": 0, "exit_other": 0})
    qs = range(2005 * 4, 2009 * 4)
    macro = pd.DataFrame({"unemp": [5.0 + 0.1 * i for i in range(len(qs))], "hpi": [100.0 + i for i in range(len(qs))],
                          "mort": [6.0] * len(qs), "gs10": [4.0] * len(qs)}, index=list(qs))
    return panel, loans, macro


def test_all_candidates_exist():
    panel, loans, macro = toy()
    f = build_features(panel, loans, macro)
    assert set(CANDIDATES) <= set(f.columns)


def test_macro_features_use_previous_quarter_only():
    panel, loans, macro = toy()
    base = build_features(panel, loans, macro)
    # corrupt ALL macro data dated at or after the first outcome quarter: first row must not change
    q_first = int(panel["qidx"].min())
    m2 = macro.copy()
    m2.loc[m2.index >= q_first, ["unemp", "hpi", "mort"]] = 999.0
    pert = build_features(panel, loans, m2)
    cols = ["unemp_lag", "unemp_chg_4q", "hpi_yoy", "mort_rate_lag", "mtm_ltv", "rate_incentive"]
    pd.testing.assert_series_equal(base.loc[0, cols], pert.loc[0, cols])
    # ... while later rows do change (the lag really is one quarter)
    assert not np.isclose(base.loc[1, "unemp_lag"], pert.loc[1, "unemp_lag"])


def test_mark_to_market_ltv_formula():
    panel, loans, macro = toy()
    f = build_features(panel, loans, macro).iloc[0]
    q0 = 2007 * 4
    hpi_lag, hpi_orig = macro.loc[q0 - 1, "hpi"], macro.loc[2006 * 4, "hpi"]
    value0 = 160000.0 / 0.80
    expected = 100 * 150000.0 / (value0 * hpi_lag / hpi_orig)
    assert np.isclose(f["mtm_ltv"], expected)


def test_loan_features_unchanged_by_future_panel_rows():
    panel, loans, macro = toy()
    full = build_features(panel, loans, macro)
    first_only = build_features(panel.iloc[:1], loans, macro)
    cols = [c for c in CANDIDATES]
    pd.testing.assert_series_equal(full.loc[0, cols], first_only.loc[0, cols])
