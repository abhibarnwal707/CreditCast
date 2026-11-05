"""Phase 4: PD model = discrete-time hazard, fitted as a logistic GLM (Binomial family, logit link) on the loan-quarter panel.

Run:  python -m creditcast.pd_model
Outputs: results/pd_metrics.json, results/pd_glm_coefficients.json, results/pd_coefficients.csv,
         results/figures/pd_calibration_validation.png, data/interim/xgb_challenger.json

The OOT test set (2017Q1-2019Q4) is NOT used here (selection uses train/validation only); validate.py adds OOT metrics.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from .config import ROOT, load_config

# readable units for coefficients (odds ratio is per 'scale' raw units)
SCALE = {"credit_score": 100.0, "ltv": 10.0, "dti": 10.0, "orig_rate": 1.0, "age_yrs": 1.0, "age_yrs_sq": 1.0,
         "mtm_ltv": 10.0, "rate_incentive": 1.0, "unemp_lag": 1.0, "unemp_chg_4q": 1.0, "hpi_yoy": 0.1,
         "purpose_cashout": 1.0, "purpose_refi_nocash": 1.0, "occ_investor": 1.0, "occ_second": 1.0}
UNIT = {"credit_score": "+100 points", "ltv": "+10 pts", "dti": "+10 pts", "orig_rate": "+1 pp", "age_yrs": "+1 year",
        "age_yrs_sq": "+1 (centred)", "mtm_ltv": "+10 pts", "rate_incentive": "+1 pp", "unemp_lag": "+1 pp",
        "unemp_chg_4q": "+1 pp", "hpi_yoy": "+10 pp", "purpose_cashout": "yes", "purpose_refi_nocash": "yes",
        "occ_investor": "yes", "occ_second": "yes"}
# expected signs (CLAUDE.md / guide Phase 4): checked on the raw, correctly-interpretable variables
EXPECTED_SIGN = {"credit_score": -1, "mtm_ltv": +1, "unemp_lag": +1, "hpi_yoy": -1}


# Dropped from the PD model after the first fit: original LTV flips sign (negative) once mark-to-market LTV is in the model,
# because MTM LTV already contains it (corr high, VIF 6.8 / 9.0). MTM LTV is the economically meaningful, scenario-sensitive variable.
MODEL_EXCLUDE = {"ltv": "sign flipped negative when mtm_ltv is included (collinear); mtm_ltv subsumes it. Still used for K-Means segmentation."}


def design(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = pd.DataFrame({f: df[f].astype(float) / SCALE[f] for f in features}, index=df.index)
    return sm.add_constant(X, has_constant="add")


def predict_glm(params: dict, df: pd.DataFrame) -> np.ndarray:
    feats = [k for k in params if k != "const"]
    X = design(df, feats)
    z = X["const"] * params["const"] + sum(X[f] * params[f] for f in feats)
    return 1 / (1 + np.exp(-z.values))


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"n": int(len(y)), "defaults": int(np.sum(y)), "auc": float(auc), "gini": float(2 * auc - 1),
            "ks": ks_stat(y, p), "brier": float(brier_score_loss(y, p)),
            "mean_pred": float(np.mean(p)), "mean_actual": float(np.mean(y))}


def calibration_table(y, p, n=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), n, labels=False) + 1
    g = d.groupby("decile").agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"), defaults=("y", "sum"))
    return g.reset_index()


def fit_glm(train: pd.DataFrame, features: list[str]):
    X = design(train, features)
    groups = pd.factorize(train["loan_id"])[0]
    model = sm.GLM(train["default"].astype(float), X, family=sm.families.Binomial())
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


def coefficient_table(res, features):
    rows = []
    for name in res.params.index:
        b, se = res.params[name], res.bse[name]
        row = {"variable": name, "unit": UNIT.get(name, ""), "coef": b, "std_err_cluster": se, "z": res.tvalues[name],
               "p_value": res.pvalues[name], "odds_ratio": float(np.exp(b)),
               "or_ci_low": float(np.exp(b - 1.96 * se)), "or_ci_high": float(np.exp(b + 1.96 * se))}
        sgn = EXPECTED_SIGN.get(name)
        row["expected_sign"] = {1: "+", -1: "-"}.get(sgn, "")
        row["sign_ok"] = None if sgn is None else bool(np.sign(b) == sgn)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    feat = pd.read_parquet(ROOT / "data" / "interim" / "features.parquet")
    selected = [f for f in json.loads((ROOT / "results" / "features.json").read_text())["selected"] if f not in MODEL_EXCLUDE]
    tr, va = feat[feat.split == "train"], feat[feat.split == "validation"]
    print(f"[rows] train {len(tr):,} (defaults {int(tr['default'].sum()):,})  validation {len(va):,} (defaults {int(va['default'].sum()):,})")

    res = fit_glm(tr, selected)
    ct = coefficient_table(res, selected)
    pd.set_option("display.width", 220)
    print(ct[["variable", "unit", "coef", "odds_ratio", "p_value", "expected_sign", "sign_ok"]].round(4).to_string(index=False))
    ct.round(6).to_csv(ROOT / "results" / "pd_coefficients.csv", index=False)
    params = {k: float(v) for k, v in res.params.items()}
    (ROOT / "results" / "pd_glm_coefficients.json").write_text(json.dumps(
        {"features": selected, "scale": {f: SCALE[f] for f in selected}, "params": params, "family": "Binomial/logit",
         "cov_type": "cluster-robust by loan", "fit_period": f"<= {cfg['splits']['train_end']}"}, indent=2))

    out = {"features": selected, "excluded_from_model": MODEL_EXCLUDE, "glm": {}, "xgboost": {}}
    p_tr, p_va = predict_glm(params, tr), predict_glm(params, va)
    out["glm"]["train"], out["glm"]["validation"] = metrics(tr["default"], p_tr), metrics(va["default"], p_va)
    out["glm"]["deviance"] = float(res.deviance)
    out["glm"]["null_deviance"] = float(res.null_deviance)
    out["glm"]["pseudo_r2_deviance"] = float(1 - res.deviance / res.null_deviance)
    out["glm"]["sign_checks"] = {r.variable: bool(r.sign_ok) for r in ct.itertuples() if r.sign_ok is not None}
    out["glm"]["calibration_validation"] = calibration_table(va["default"], p_va).round(6).to_dict("records")
    out["glm"]["calibration_train"] = calibration_table(tr["default"], p_tr).round(6).to_dict("records")

    # XGBoost challenger: same features, early stopping on the validation period
    Xtr, Xva = tr[selected].astype(float), va[selected].astype(float)
    clf = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=20, reg_lambda=5.0, eval_metric="auc", early_stopping_rounds=40,
                            random_state=cfg["seed"], n_jobs=4)
    clf.fit(Xtr, tr["default"], eval_set=[(Xva, va["default"])], verbose=False)
    q_tr, q_va = clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xva)[:, 1]
    out["xgboost"]["train"], out["xgboost"]["validation"] = metrics(tr["default"], q_tr), metrics(va["default"], q_va)
    out["xgboost"]["best_iteration"] = int(clf.best_iteration)
    out["xgboost"]["calibration_validation"] = calibration_table(va["default"], q_va).round(6).to_dict("records")
    clf.save_model(str(ROOT / "data" / "interim" / "xgb_challenger.json"))

    dg = out["xgboost"]["validation"]["auc"] - out["glm"]["validation"]["auc"]
    out["selection"] = {"rule": "champion = GLM unless XGBoost validation AUC exceeds it by > 0.02 (interpretability, regulatory standard); decided on validation only",
                        "xgb_minus_glm_validation_auc": float(dg),
                        "champion": "xgboost" if dg > 0.02 else "glm"}
    (ROOT / "results" / "pd_metrics.json").write_text(json.dumps(out, indent=2))
    for m in ("glm", "xgboost"):
        for s in ("train", "validation"):
            r = out[m][s]
            print(f"[{m:<7}] {s:<10} AUC {r['auc']:.4f}  Gini {r['gini']:.4f}  KS {r['ks']:.4f}  Brier {r['brier']:.5f}  "
                  f"mean pred {r['mean_pred']:.4%} vs actual {r['mean_actual']:.4%}")
    print(f"[select] XGB - GLM validation AUC = {dg:+.4f} -> champion {out['selection']['champion']}")

    cal = pd.DataFrame(out["glm"]["calibration_validation"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(cal["predicted"] * 100, cal["actual"] * 100, "o-", color="#1f4e79", label="GLM, validation 2015-2016")
    mx = max(cal["predicted"].max(), cal["actual"].max()) * 100
    ax.plot([0, mx], [0, mx], "k--", lw=1)
    ax.set_xlabel("predicted default rate per quarter (%)")
    ax.set_ylabel("actual default rate per quarter (%)")
    ax.set_title("PD calibration by decile")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "pd_calibration_validation.png", dpi=150)
    print("[done] results/pd_metrics.json, pd_glm_coefficients.json, pd_coefficients.csv")


if __name__ == "__main__":
    main()
