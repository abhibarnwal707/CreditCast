"""Phase 2b: ARIMA macro forecasting with a rolling-origin (expanding window) backtest.

Run:  python -m creditcast.arima        -> results/arima.json
Series: unemployment rate (level), FHFA HPI growth (annualised q/q log growth), 30y mortgage rate (level).
Order selection: AIC grid on the PRE-evaluation sample only (no look-ahead); d from an ADF test on that sample.
"""
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from .config import ROOT, load_config, q_to_idx

HORIZONS = range(1, 9)
warnings.filterwarnings("ignore")


def load_series(cfg: dict) -> dict[str, pd.Series]:
    f = pd.read_csv(ROOT / "data" / "external" / "fred_quarterly.csv", index_col="quarter")
    f.index = pd.PeriodIndex(f.index, freq="Q")
    f = f.loc[: cfg["macro"]["history_end"]]
    hpi_g = (400 * np.log(f["USSTHPI"]).diff()).dropna()
    return {
        "unemployment": f["UNRATE"].dropna(),
        "hpi_growth": hpi_g,
        "mortgage_rate": f["MORTGAGE30US"].dropna(),
    }


def choose_d(y: pd.Series) -> tuple[int, float, float]:
    p0 = adfuller(y.values, autolag="AIC")[1]
    if p0 < 0.05:
        return 0, p0, float("nan")
    p1 = adfuller(np.diff(y.values), autolag="AIC")[1]
    return 1, p0, p1


def make_model(y, order):
    trend = "c" if order[1] == 0 else "n"
    return ARIMA(np.asarray(y), order=order, trend=trend)


def select_order(y: pd.Series, d: int, max_p: int, max_q: int) -> tuple[tuple[int, int, int], float]:
    best = (None, np.inf)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                aic = make_model(y, (p, d, q)).fit().aic
            except Exception:
                continue
            if aic < best[1]:
                best = ((p, d, q), aic)
    return best


def rolling_backtest(y: pd.Series, order, first_origin: str) -> dict:
    idx0 = y.index.get_loc(pd.Period(first_origin, freq="Q"))
    errs = {m: {h: [] for h in HORIZONS} for m in ("arima", "random_walk", "mean")}
    for i in range(idx0, len(y) - 1):
        train = y.iloc[: i + 1]
        H = min(max(HORIZONS), len(y) - 1 - i)
        try:
            fc = make_model(train, order).fit().forecast(H)
        except Exception:
            continue
        mu = train.mean()
        for h in range(1, H + 1):
            actual = y.iloc[i + h]
            errs["arima"][h].append(fc[h - 1] - actual)
            errs["random_walk"][h].append(train.iloc[-1] - actual)
            errs["mean"][h].append(mu - actual)
    rmse = {m: {h: float(np.sqrt(np.mean(np.square(e)))) for h, e in d.items() if e} for m, d in errs.items()}
    avg = {m: float(np.mean(list(r.values()))) for m, r in rmse.items()}
    return {
        "n_origins": len(errs["arima"][1]),
        "rmse_by_horizon": rmse,
        "avg_rmse_1_to_8q": avg,
        "improvement_vs_random_walk_pct": float((1 - avg["arima"] / avg["random_walk"]) * 100),
        "improvement_vs_mean_pct": float((1 - avg["arima"] / avg["mean"]) * 100),
    }


def main() -> None:
    cfg = load_config()
    m = cfg["macro"]
    fed = pd.read_csv(ROOT / "data" / "external" / "fed_2026_scenarios.csv")
    base = fed[fed["scenario"] == "baseline"].set_index("quarter")
    out = {"eval_first_origin": m["arima_eval_first_origin"], "history_end": m["history_end"], "series": {}}
    for name, y in load_series(cfg).items():
        pre = y.loc[: pd.Period(m["arima_eval_first_origin"], freq="Q") - 1]
        d, p0, p1 = choose_d(pre)
        order, aic = select_order(pre, d, m["arima_max_p"], m["arima_max_q"])
        bt = rolling_backtest(y, order, m["arima_eval_first_origin"])
        # sanity check vs Fed baseline: full-sample fit, 13-quarter forecast
        fc = make_model(y, order).fit().forecast(13)
        fed_col = {"unemployment": "unemployment", "mortgage_rate": "mortgage_rate"}.get(name)
        cmp = None
        if fed_col:
            fv = base[fed_col].values
            cmp = {"arima_forecast": [round(float(x), 2) for x in fc], "fed_baseline": [float(x) for x in fv],
                   "mean_abs_diff": float(np.mean(np.abs(fc - fv)))}
        else:  # hpi growth: Fed baseline annualised q/q growth from index levels
            lv = base["house_price_index"].values
            fv = 400 * np.diff(np.log(lv))
            cmp = {"arima_forecast_q2_2026_to_q1_2029": [round(float(x), 2) for x in fc[1:]],
                   "fed_baseline": [round(float(x), 2) for x in fv],
                   "mean_abs_diff": float(np.mean(np.abs(fc[1:] - fv)))}
        out["series"][name] = {"adf_p_level": float(p0), "adf_p_diff": None if np.isnan(p1) else float(p1), "d": d,
                               "order": list(order), "aic_pre_sample": float(aic), "backtest": bt,
                               "sanity_vs_fed_baseline": cmp, "long_run_mean": float(y.loc[m["long_run_start"]: ].mean())}
        print(f"[arima] {name:<14} d={d} order={order}  RMSE(1-8q avg) arima={bt['avg_rmse_1_to_8q']['arima']:.3f} "
              f"rw={bt['avg_rmse_1_to_8q']['random_walk']:.3f} mean={bt['avg_rmse_1_to_8q']['mean']:.3f}  "
              f"vs RW {bt['improvement_vs_random_walk_pct']:+.1f}%  vs mean {bt['improvement_vs_mean_pct']:+.1f}%  origins={bt['n_origins']}")
    (ROOT / "results" / "arima.json").write_text(json.dumps(out, indent=2))
    print("[done] results/arima.json")


if __name__ == "__main__":
    main()
