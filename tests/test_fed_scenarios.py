"""Checks the figures that appear on the resume against the Fed 2026 scenario tables extracted from the PDF."""
import numpy as np
import pandas as pd
import pytest

from creditcast.config import ROOT


@pytest.fixture(scope="module")
def fed():
    return pd.read_csv(ROOT / "data" / "external" / "fed_2026_scenarios.csv")


def sa(fed):
    return fed[fed.scenario == "severely_adverse"].sort_values("quarter").reset_index(drop=True)


def ba(fed):
    return fed[fed.scenario == "baseline"].sort_values("quarter").reset_index(drop=True)


def test_shape_13_quarters_each(fed):
    assert len(sa(fed)) == 13 and len(ba(fed)) == 13
    assert sa(fed).quarter.iloc[0] == "2026Q1" and sa(fed).quarter.iloc[-1] == "2029Q1"


def test_unemployment_peaks_at_10_percent_in_2027q3(fed):
    s = sa(fed)
    assert s.unemployment.max() == 10.0
    assert s.loc[s.unemployment.idxmax(), "quarter"] == "2027Q3"


def test_severe_unemployment_above_baseline_every_quarter(fed):
    assert (sa(fed).unemployment.values > ba(fed).unemployment.values).all()


def test_house_prices_fall_about_30_percent(fed):
    # Q4-2025 HPI level is not in the PDF tables; the baseline Q1-2026 level (325) is the proxy for the jump-off.
    jump = ba(fed).house_price_index.iloc[0]
    trough = sa(fed).house_price_index.min()
    fall = 1 - trough / jump
    assert 0.29 <= fall <= 0.31, fall
    assert sa(fed).loc[sa(fed).house_price_index.idxmin(), "quarter"] == "2027Q4"


def test_real_gdp_falls_about_4_6_percent_to_2027q2(fed):
    # table gives annualised q/q growth; compound Q1-2026..Q2-2027 to get the level change from Q4-2025
    g = sa(fed).set_index("quarter").loc["2026Q1":"2027Q2", "real_gdp_growth"].values / 100
    level_change = np.prod((1 + g) ** 0.25) - 1
    assert -0.048 <= level_change <= -0.044, level_change   # text of the Fed report: 4.6%, tolerance for rounding
