"""Phase 7: Fed 2026 stress test (baseline vs severely adverse) and a sensitivity grid.

9-quarter planning horizon = 2026Q1 .. 2028Q1 (t = 1..9). Loss rate = cumulative ECL over the horizon / starting UPB.
Sensitivity grid: peak unemployment U in {6, 8, 10, 12}% x peak house-price decline D in {10, 20, 30, 40}%.
Synthetic scenario: the Fed severely adverse paths scaled so that the unemployment increase over the jump-off reaches U and the HPI trough is D below the jump-off;
the mortgage rate and 10y yield follow the Fed severely adverse path; after Q1 2029 the usual mean reversion applies.

Run:  python -m creditcast.stress -> results/stress.json, results/figures/stress_*.png
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ROOT, idx_to_q, load_config, q_to_idx
from .ecl import load_portfolio, prepare_long, project, quarterly_profile
from .paths import all_paths, load_history, scenario_path

U_GRID = [6, 8, 10, 12]
D_GRID = [10, 20, 30, 40]
HORIZON = 9


def synthetic_scenario(fed: pd.DataFrame, hist: pd.DataFrame, cfg: dict, peak_u: float, hpi_decline_pct: float) -> pd.DataFrame:
    sa = fed[fed["scenario"] == "severely_adverse"].sort_values("quarter").set_index("quarter")
    ba = fed[fed["scenario"] == "baseline"].sort_values("quarter").set_index("quarter")
    u0 = float(hist.loc[cfg["scenario"]["jump_off"], "UNRATE"])
    k_u = (peak_u - u0) / (sa["unemployment"].max() - u0)
    s = sa.copy()
    s["unemployment"] = u0 + (sa["unemployment"] - u0) * k_u
    jump_level = float(ba["house_price_index"].iloc[0])            # same jump-off proxy as paths.py
    trough = float(sa["house_price_index"].min())
    k_h = np.log(1 - hpi_decline_pct / 100.0) / np.log(trough / jump_level)
    s["house_price_index"] = jump_level * np.exp(k_h * np.log(sa["house_price_index"] / jump_level))
    return s


def run(port, L, path, pdp, pp, lgd, cfg):
    r = project(port, L, path, pdp, pp, lgd, cfg)
    return r, quarterly_profile(port, r, 40)


def main() -> None:
    cfg = load_config()
    pdp = json.loads((ROOT / "results" / "pd_glm_coefficients.json").read_text())["params"]
    pp = json.loads((ROOT / "results" / "prepay_model.json").read_text())
    lgd = json.loads((ROOT / "results" / "lgd.json").read_text())["model"]
    port = load_portfolio(cfg)
    L = prepare_long(port, cfg["scenario"]["max_loan_quarters"])
    paths = all_paths(cfg)
    first_q = q_to_idx(cfg["scenario"]["jump_off"]) + 1
    out = {"horizon_quarters": HORIZON, "horizon": f"{idx_to_q(first_q)} to {idx_to_q(first_q + HORIZON - 1)}",
           "loans": int(len(port)), "starting_upb": float(port["start_upb"].sum()), "scenarios": {}}
    profs = {}
    for scen in ("baseline", "severely_adverse"):
        r, prof = run(port, L, paths[paths["scenario"] == scen].reset_index(drop=True), pdp, pp, lgd, cfg)
        profs[scen] = prof
        p9 = prof[prof["t"] <= HORIZON]
        peak = p9.loc[p9["quarterly_default_rate"].idxmax()]
        out["scenarios"][scen] = {
            "loss_rate_9q": float(p9["quarterly_loss_rate"].sum()),
            "loss_dollars_9q": float(p9["loss"].sum()),
            "cum_default_rate_9q": float(p9["quarterly_default_rate"].sum()),
            "peak_quarterly_default_rate": float(peak["quarterly_default_rate"]),
            "peak_quarter": idx_to_q(first_q + int(peak["t"]) - 1),
            "loss_path_by_quarter": {idx_to_q(first_q + int(t) - 1): float(v) for t, v in zip(p9["t"], p9["quarterly_loss_rate"])},
            "default_rate_path_by_quarter": {idx_to_q(first_q + int(t) - 1): float(v) for t, v in zip(p9["t"], p9["quarterly_default_rate"])},
        }
        s = out["scenarios"][scen]
        print(f"[stress] {scen:<17} 9q loss rate {s['loss_rate_9q']:.4%} (${s['loss_dollars_9q']:,.0f}); 9q cum default rate {s['cum_default_rate_9q']:.3%}; "
              f"peak quarterly default rate {s['peak_quarterly_default_rate']:.3%} in {s['peak_quarter']}")
    b, a = out["scenarios"]["baseline"], out["scenarios"]["severely_adverse"]
    out["ratio_severely_adverse_to_baseline_loss_9q"] = a["loss_rate_9q"] / b["loss_rate_9q"]
    out["ratio_peak_default_rate"] = a["peak_quarterly_default_rate"] / b["peak_quarterly_default_rate"]
    print(f"[stress] ratio severely adverse / baseline (9q loss): {out['ratio_severely_adverse_to_baseline_loss_9q']:.2f}x")

    # sensitivity grid
    fed = pd.read_csv(ROOT / "data" / "external" / "fed_2026_scenarios.csv")
    hist = load_history(cfg)
    grid = np.zeros((len(U_GRID), len(D_GRID)))
    for i, U in enumerate(U_GRID):
        for j, D in enumerate(D_GRID):
            sc = synthetic_scenario(fed, hist, cfg, U, D)
            path = scenario_path(sc, hist, cfg)
            _, prof = run(port, L, path, pdp, pp, lgd, cfg)
            grid[i, j] = prof.loc[prof["t"] <= HORIZON, "quarterly_loss_rate"].sum()
    gdf = pd.DataFrame(grid * 100, index=[f"U={u}%" for u in U_GRID], columns=[f"HPI -{d}%" for d in D_GRID])
    print("[grid] 9-quarter loss rate (% of starting UPB):\n", gdf.round(4).to_string())
    mono_u = bool((np.diff(grid, axis=0) >= -1e-12).all())
    mono_d = bool((np.diff(grid, axis=1) >= -1e-12).all())
    out["sensitivity_grid"] = {"unemployment_peak_pct": U_GRID, "hpi_decline_pct": D_GRID, "loss_rate_9q": grid.tolist(),
                               "monotone_in_unemployment": mono_u, "monotone_in_hpi_decline": mono_d,
                               "grid_construction": "Fed severely adverse paths scaled to the target unemployment peak and HPI trough"}
    print(f"[grid] monotone in unemployment: {mono_u}; monotone in house-price decline: {mono_d}")
    assert mono_u and mono_d, "more stress must mean more loss in every cell"
    (ROOT / "results" / "stress.json").write_text(json.dumps(out, indent=2))

    figdir = ROOT / "results" / "figures"
    fig, ax = plt.subplots(figsize=(6, 4.2))
    im = ax.imshow(grid * 100, cmap="YlOrRd", aspect="auto", origin="lower")
    ax.set_xticks(range(len(D_GRID)), [f"-{d}%" for d in D_GRID])
    ax.set_yticks(range(len(U_GRID)), [f"{u}%" for u in U_GRID])
    ax.set_xlabel("peak house-price decline")
    ax.set_ylabel("peak unemployment rate")
    for i in range(len(U_GRID)):
        for j in range(len(D_GRID)):
            ax.text(j, i, f"{grid[i, j] * 100:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("9-quarter loss rate (% of starting UPB)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(figdir / "stress_sensitivity_heatmap.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for scen, c in (("baseline", "#1f4e79"), ("severely_adverse", "#c0504d")):
        s = out["scenarios"][scen]["loss_path_by_quarter"]
        ax.plot(list(s.keys()), np.array(list(s.values())) * 100, marker="o", color=c, label=scen.replace("_", " "))
    ax.set_ylabel("quarterly loss rate (% of starting UPB)")
    ax.set_title("Fed 2026 scenarios: projected quarterly loss path")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "stress_loss_path.png", dpi=150)
    print("[done] results/stress.json + figures")


if __name__ == "__main__":
    main()
