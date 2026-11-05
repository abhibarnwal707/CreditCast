import numpy as np

from creditcast.lgd_ead import amortisation_balances, lgd_function


def test_schedule_amortises_to_zero():
    b = amortisation_balances(200_000, 6.0, 360, 130)   # 120 quarters = 360 months
    assert b[0] == 200_000
    assert b[119] > 0 and b[120] == 0 and b[-1] == 0
    assert np.all(np.diff(b) <= 1e-9)                    # non-increasing


def test_ead_never_exceeds_current_upb():
    for rate in (0.0, 2.5, 7.5):
        b = amortisation_balances(150_000, rate, 240, 100)
        assert b.max() <= 150_000 + 1e-9


def test_zero_rate_is_straight_line():
    b = amortisation_balances(120_000, 0.0, 120, 10)
    assert np.isclose(b[1], 120_000 - 3 * 1000)


def test_no_balance_or_term():
    assert amortisation_balances(0, 5, 100, 5).sum() == 0
    assert amortisation_balances(1000, 5, 0, 5).sum() == 0


def test_lgd_rises_when_house_prices_fall_and_is_bounded():
    m = {"liq_intercept": -3.0, "liq_slope": 0.3, "sev_intercept": 0.5, "sev_slope": -0.03, "sev_min": 0.05}
    h = np.array([0.5, 0.2, 0.0, -0.2, -0.4])           # HPI change since origination, falling
    x = 100 * 0.8 / (1 + h)                              # so MTM LTV rises
    lgd, p, sev = lgd_function(m, x, h)
    assert np.all(np.diff(lgd) > 0)
    assert lgd.min() >= 0 and lgd.max() <= 1
