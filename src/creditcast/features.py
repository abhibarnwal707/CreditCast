"""Phase 3: loan-quarter features (measured at the START of each quarter) and K-Means segmentation.

Run:  python -m creditcast.features  -> data/interim/features.parquet, results/segments.json, results/features.json

No-leakage design:
  * loan features come from the previous quarter's last monthly record (panel start_* columns) and origination data;
  * macro features at outcome quarter q use macro values dated q-1 (information available at the start of q);
  * scaler and K-Means are fitted on training-period loans only.
"""
import json

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import ROOT, assign_split, idx_to_q, load_config, q_to_idx

SEG_VARS = ["credit_score", "ltv", "dti", "orig_rate"]
AGE_CENTER = 4.0   # years, approx. mean loan age in the training period (fixed constant, not data-fitted)


def macro_by_qidx() -> pd.DataFrame:
    f = pd.read_csv(ROOT / "data" / "external" / "fred_quarterly.csv", index_col="quarter")
    f.index = pd.PeriodIndex(f.index, freq="Q")
    m = pd.DataFrame({"unemp": f["UNRATE"], "hpi": f["USSTHPI"], "mort": f["MORTGAGE30US"], "gs10": f["GS10"]})
    m.index = [p.year * 4 + p.quarter - 1 for p in m.index]
    return m


def loan_orig_qidx(loan_id: pd.Series) -> pd.Series:
    yy = loan_id.str[1:3].astype(int)
    year = np.where(yy >= 99, 1900 + yy, 2000 + yy)
    return pd.Series(year * 4 + loan_id.str[4].astype(int) - 1, index=loan_id.index)


def lag_lookup(series: pd.Series, qidx: pd.Series, lag: int) -> np.ndarray:
    return series.reindex(qidx.values - lag).values


def build_features(panel: pd.DataFrame, loans: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Pure function (unit-tested for look-ahead): features for each panel row."""
    L = loans[["loan_id", "vintage", "credit_score", "ltv", "dti", "orig_rate", "orig_upb", "loan_purpose", "occupancy"]].copy()
    L["orig_qidx"] = loan_orig_qidx(L["loan_id"])
    df = panel.merge(L, on="loan_id", how="left", validate="m:1")
    q = df["qidx"]
    unemp, hpi, mort = macro["unemp"], macro["hpi"], macro["mort"]
    df["unemp_lag"] = lag_lookup(unemp, q, 1)
    df["unemp_chg_4q"] = df["unemp_lag"] - lag_lookup(unemp, q, 5)
    hpi_lag = lag_lookup(hpi, q, 1)
    df["hpi_yoy"] = hpi_lag / lag_lookup(hpi, q, 5) - 1
    df["mort_rate_lag"] = lag_lookup(mort, q, 1)
    hpi_orig = hpi.reindex(df["orig_qidx"].values).values
    orig_value = df["orig_upb"] / (df["ltv"] / 100.0)
    df["mtm_ltv"] = 100.0 * df["start_upb"] / (orig_value * hpi_lag / hpi_orig)
    df["hpi_chg_orig"] = hpi_lag / hpi_orig - 1          # house-price change since origination (for LGD severity)
    df["rate_incentive"] = df["orig_rate"] - df["mort_rate_lag"]
    df["age_yrs"] = df["start_age"] / 12.0
    df["age_yrs_sq"] = (df["age_yrs"] - AGE_CENTER) ** 2   # centred so age and age^2 are not collinear
    for k, name in (("C", "purpose_cashout"), ("N", "purpose_refi_nocash"), ("R", "purpose_refi_other")):
        df[name] = (df["loan_purpose"] == k).astype(int)
    df["occ_investor"] = (df["occupancy"] == "I").astype(int)
    df["occ_second"] = (df["occupancy"] == "S").astype(int)
    return df


CANDIDATES = ["credit_score", "ltv", "dti", "orig_rate", "age_yrs", "age_yrs_sq", "mtm_ltv", "rate_incentive",
              "unemp_lag", "unemp_chg_4q", "hpi_yoy", "purpose_cashout", "purpose_refi_nocash", "purpose_refi_other",
              "occ_investor", "occ_second"]


def vif_select(df: pd.DataFrame, cols: list[str], thresh: float = 10.0):
    cols = list(cols)
    log = []
    for c in list(cols):
        if df[c].astype(float).std() == 0:          # constant in the training period: cannot be estimated
            log.append({"dropped": c, "reason": "zero variance in training period"})
            cols.remove(c)
    X = df[cols].astype(float)
    X = (X - X.mean()) / X.std()
    while True:
        Xc = np.column_stack([np.ones(len(X)), X[cols].values])
        v = pd.Series([variance_inflation_factor(Xc, i + 1) for i in range(len(cols))], index=cols)
        if v.max() <= thresh:
            return cols, v, log
        worst = v.idxmax()
        log.append({"dropped": worst, "vif": float(v.max())})
        cols.remove(worst)


def fit_segments(feat: pd.DataFrame, cfg: dict) -> tuple[dict, np.ndarray]:
    train_loans = feat.loc[feat["split"] == "train", ["loan_id"] + SEG_VARS].drop_duplicates("loan_id").sort_values("loan_id").reset_index(drop=True)
    scaler = StandardScaler().fit(train_loans[SEG_VARS])
    Z = scaler.transform(train_loans[SEG_VARS])
    rng = np.random.default_rng(cfg["seed"])
    sil_idx = rng.choice(len(Z), size=min(10000, len(Z)), replace=False)
    scores = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg["seed"]).fit(Z)
        scores[k] = {"inertia": float(km.inertia_), "silhouette": float(silhouette_score(Z[sil_idx], km.labels_[sil_idx]))}
        print(f"[kmeans] k={k} inertia={km.inertia_:,.0f} silhouette={scores[k]['silhouette']:.3f}")
    # choose k by silhouette among k>=3 (k=2 is a trivial good/bad split); elbow reported for reference
    best_k = max((k for k in scores if k >= 3), key=lambda k: scores[k]["silhouette"])
    km = KMeans(n_clusters=best_k, n_init=10, random_state=cfg["seed"]).fit(Z)
    order = np.argsort(km.cluster_centers_[:, 0])   # relabel so segment 0 = lowest FICO centroid
    remap = {old: new for new, old in enumerate(order)}
    allL = feat[["loan_id"] + SEG_VARS].drop_duplicates("loan_id")
    seg = pd.Series(km.predict(scaler.transform(allL[SEG_VARS]))).map(remap).values
    seg_map = dict(zip(allL["loan_id"], seg))
    info = {"k": int(best_k), "silhouette": scores[best_k]["silhouette"], "selection_rule": "max silhouette over k=3..8 (k=2 excluded as a trivial split); elbow (inertia) reported",
            "k_scan": scores, "fit_on": "training-period loans only", "n_train_loans": int(len(train_loans)),
            "scaler_mean": dict(zip(SEG_VARS, scaler.mean_.tolist())), "scaler_scale": dict(zip(SEG_VARS, scaler.scale_.tolist()))}
    return info, feat["loan_id"].map(seg_map).values


def main() -> None:
    cfg = load_config()
    con = duckdb.connect()
    panel = con.execute(f"SELECT * FROM read_parquet('{ROOT}/data/interim/panel.parquet')").df()
    loans = con.execute(f"SELECT * FROM read_parquet('{ROOT}/data/interim/loans.parquet')").df()
    feat = build_features(panel, loans, macro_by_qidx())
    feat["split"] = assign_split(feat["qidx"], cfg).values
    print(f"[rows] loan-quarters: {len(feat):,}  defaults: {int(feat['default'].sum()):,}")
    na = feat[CANDIDATES].isna().sum()
    print("[rows] feature NaNs:", {k: int(v) for k, v in na.items() if v})
    before = len(feat)
    feat = feat.dropna(subset=CANDIDATES).reset_index(drop=True)
    print(f"[rows] dropped {before - len(feat):,} rows with missing features; remaining {len(feat):,}, defaults {int(feat['default'].sum()):,}")

    info, seg = fit_segments(feat, cfg)
    feat["segment"] = seg
    train = feat[feat["split"] == "train"]
    prof = []
    for s, g in feat.groupby("segment"):
        gl = g.drop_duplicates("loan_id")
        gt = train[train["segment"] == s]
        prof.append({"segment": int(s), "loans": int(len(gl)), "mean_fico": float(gl["credit_score"].mean()), "mean_ltv": float(gl["ltv"].mean()),
                     "mean_dti": float(gl["dti"].mean()), "mean_orig_rate": float(gl["orig_rate"].mean()),
                     "train_loan_quarters": int(len(gt)), "train_defaults": int(gt["default"].sum()),
                     "train_default_rate_per_quarter": float(gt["default"].mean()) if len(gt) else None})
    info["segment_profiles"] = prof
    print(pd.DataFrame(prof).round(3).to_string(index=False))
    (ROOT / "results" / "segments.json").write_text(json.dumps(info, indent=2))

    keep, vifs, log = vif_select(train, CANDIDATES)
    print("[vif] dropped:", log, "\n[vif] final VIF:\n", vifs.round(2).to_string())
    (ROOT / "results" / "features.json").write_text(json.dumps(
        {"candidates": CANDIDATES, "selected": keep, "vif_dropped": log, "final_vif": vifs.round(3).to_dict(),
         "note": "VIF on training period, standardised; threshold 10"}, indent=2))
    feat.to_parquet(ROOT / "data" / "interim" / "features.parquet", index=False)
    print("[done] data/interim/features.parquet, results/segments.json, results/features.json")


if __name__ == "__main__":
    main()
