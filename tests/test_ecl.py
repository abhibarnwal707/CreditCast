import json

import numpy as np
import pandas as pd
import pytest

from creditcast.config import ROOT, load_config
from creditcast.ecl import amortisation_matrix_long, load_portfolio, prepare_long, project
from creditcast.lgd_ead import amortisation_balances

ARTIFACTS = [ROOT / "results" / n for n in ("pd_glm_coefficients.json", "lgd.json", "prepay_model.json")] + \
            [ROOT / "data" / "interim" / "features.parquet", ROOT / "data" / "external" / "macro_paths.csv"]
needs = pytest.mark.skipif(not all(p.exists() for p in ARTIFACTS), reason="run the pipeline first")


def test_vectorised_amortisation_matches_scalar():
    upb = np.array([200_000.0, 90_000.0, 50_000.0])
    rate = np.array([6.0, 0.0, 3.5])
    rem = np.array([360, 120, 37])
    n_q = np.minimum(np.ceil(rem / 3).astype(int), 120)
    li, t, bal = amortisation_matrix_long(upb, rate, rem, n_q)
    for i in range(3):
        ref = amortisation_balances(upb[i], rate[i], rem[i], n_q[i])
        assert np.allclose(bal[li == i], ref, atol=1e-6)


@pytest.fixture(scope="module")
def runs():
    cfg = load_config()
    pdp = json.loads((ROOT / "results" / "pd_glm_coefficients.json").read_text())["params"]
    lg = json.loads((ROOT / "results" / "lgd.json").read_text())["model"]
    pp = json.loads((ROOT / "results" / "prepay_model.json").read_text())
    port = load_portfolio(cfg)
    L = prepare_long(port, cfg["scenario"]["max_loan_quarters"])
    from creditcast.paths import all_paths
    paths = all_paths(cfg)
    return {s: project(port, L, paths[paths.scenario == s].reset_index(drop=True), pdp, pp, lg, cfg) for s in ("baseline", "severely_adverse")}


@needs
def test_survival_between_0_and_1_and_non_increasing(runs):
    for r in runs.values():
        assert ((r["s"] >= 0) & (r["s"] <= 1)).all()
        d = pd.DataFrame({"li": r["li"], "s": r["s"]})
        assert (d.groupby("li")["s"].diff().dropna() <= 1e-12).all()


@needs
def test_ecl_non_negative_and_ead_not_above_balance(runs):
    for r in runs.values():
        assert (r["ecl"] >= 0).all() and (r["lgd"] >= 0).all() and (r["lgd"] <= 1).all()
        assert ((r["hd"] >= 0) & (r["hd"] <= 1)).all()


@needs
def test_severely_adverse_ecl_at_least_baseline(runs):
    assert runs["severely_adverse"]["ecl"].sum() >= runs["baseline"]["ecl"].sum()
