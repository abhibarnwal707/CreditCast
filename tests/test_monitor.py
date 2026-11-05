import numpy as np

from creditcast.monitor import psi, psi_flag, rolling_quarterly_auc


def test_psi_identical_samples_is_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x.copy()) == 0.0


def test_psi_detects_shift_and_flags():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 20000)
    assert psi_flag(psi(base, rng.normal(0, 1, 20000))) == "stable"
    assert psi_flag(psi(base, rng.normal(1.5, 1, 20000))) == "shifted"
    assert psi(base, rng.normal(0.3, 1, 20000)) > 0


def test_psi_constant_variable_is_zero():
    assert psi(np.ones(100), np.ones(100)) == 0.0
