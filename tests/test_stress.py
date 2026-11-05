import json

import numpy as np
import pandas as pd
import pytest

from creditcast.config import ROOT, load_config
from creditcast.paths import load_history
from creditcast.stress import synthetic_scenario

STRESS = ROOT / "results" / "stress.json"
FED = ROOT / "data" / "external" / "fed_2026_scenarios.csv"


@pytest.mark.skipif(not FED.exists(), reason="run macro first")
def test_synthetic_scenario_hits_targets():
    cfg = load_config()
    fed = pd.read_csv(FED)
    hist = load_history(cfg)
    jump = fed[fed.scenario == "baseline"].sort_values("quarter").house_price_index.iloc[0]
    for U, D in [(6, 10), (12, 40)]:
        s = synthetic_scenario(fed, hist, cfg, U, D)
        assert np.isclose(s.unemployment.max(), U)
        assert np.isclose(1 - s.house_price_index.min() / jump, D / 100)


@pytest.mark.skipif(not STRESS.exists(), reason="run stress first")
def test_grid_monotone_and_severe_above_baseline():
    r = json.loads(STRESS.read_text())
    g = np.array(r["sensitivity_grid"]["loss_rate_9q"])
    assert (np.diff(g, axis=0) >= 0).all() and (np.diff(g, axis=1) >= 0).all()
    assert r["scenarios"]["severely_adverse"]["loss_rate_9q"] > r["scenarios"]["baseline"]["loss_rate_9q"]
    assert r["scenarios"]["severely_adverse"]["peak_quarterly_default_rate"] > r["scenarios"]["baseline"]["peak_quarterly_default_rate"]
