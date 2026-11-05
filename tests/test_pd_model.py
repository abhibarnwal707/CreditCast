import json

import numpy as np
import pandas as pd
import pytest

from creditcast.config import ROOT
from creditcast.pd_model import EXPECTED_SIGN, ks_stat, metrics, predict_glm

COEF = ROOT / "results" / "pd_glm_coefficients.json"
needs_model = pytest.mark.skipif(not COEF.exists(), reason="run `make train` first")


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_stat(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert 0 <= ks_stat(y, np.array([0.5, 0.5, 0.5, 0.5])) <= 1


def test_metrics_gini_is_2auc_minus_1():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    m = metrics(y, p)
    assert np.isclose(m["gini"], 2 * m["auc"] - 1)


@needs_model
def test_probabilities_between_0_and_1():
    c = json.loads(COEF.read_text())
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f: rng.normal(0, 50, 1000) for f in c["features"]})
    p = predict_glm(c["params"], df)
    assert ((p >= 0) & (p <= 1)).all()


@needs_model
def test_expected_coefficient_signs():
    c = json.loads(COEF.read_text())["params"]
    for var, sign in EXPECTED_SIGN.items():
        assert np.sign(c[var]) == sign, f"{var} has the wrong sign"
