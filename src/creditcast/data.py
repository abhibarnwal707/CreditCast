"""Phase 1: load Freddie Mac sample files into DuckDB, clean, sample, build the loan-quarter panel.

Run:  python -m creditcast.data
Outputs: data/interim/{loans,panel,losses}.parquet, results/data_summary.json,
         results/figures/default_rate_by_quarter.png
"""
import json
import re
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import ROOT, assign_split, idx_to_q, load_config, q_to_idx
from .schema import ORIG_COLUMNS, PERF_COLUMNS


def split_statements(sql: str) -> list[str]:
    sql = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    return [s.strip() for s in sql.split(";") if s.strip()]


def run_sql_file(con, path: Path) -> None:
    for stmt in split_statements(Path(path).read_text()):
        con.execute(stmt)


def _columns_arg(cols) -> str:
    return "{" + ", ".join(f"'{n}': '{t}'" for n, t in cols) + "}"


def register_raw(con, raw_dir: Path) -> None:
    for kind, cols in (("orig", ORIG_COLUMNS), ("perf", PERF_COLUMNS)):
        files = sorted(raw_dir.glob(f"sample_*/sample_{kind}_*.txt"))
        assert files, f"no {kind} files under {raw_dir}"
        flist = "[" + ", ".join(f"'{f}'" for f in files) + "]"
        con.execute(
            f"CREATE OR REPLACE VIEW {kind}_raw AS SELECT * FROM read_csv({flist}, delim='|', header=false, "
            f"columns={_columns_arg(cols)}, quote='')"
        )


def build(cfg: dict) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(ROOT / "data" / "interim" / "creditcast.duckdb"))
    register_raw(con, ROOT / "data" / "raw")
    con.execute("CREATE OR REPLACE TABLE params AS SELECT ? AS n_loans, ? AS seed", [cfg["n_loans"], cfg["seed"]])
    run_sql_file(con, ROOT / "sql" / "clean_sample.sql")
    print("[clean_sample] done")
    run_sql_file(con, ROOT / "sql" / "build_panel.sql")
    print("[build_panel] done")
    return con


def summarise(con, cfg: dict) -> dict:
    out = {}
    clean = con.execute(
        "SELECT coalesce(removal_rule, 'kept') AS rule, count(*) AS n FROM orig_clean_flags GROUP BY 1 ORDER BY 1"
    ).df()
    raw_loans = int(clean["n"].sum())
    out["cleaning"] = {
        "raw_loans": raw_loans,
        "removed_by_rule": {r: int(n) for r, n in zip(clean["rule"], clean["n"]) if r != "kept"},
        "valid_after_cleaning": int(clean.loc[clean["rule"] == "kept", "n"].sum()),
    }
    print(f"[rows] raw origination loans: {raw_loans:,}")
    for r, n in out["cleaning"]["removed_by_rule"].items():
        print(f"       removed {n:>7,}  {r}")
    print(f"       valid after cleaning: {out['cleaning']['valid_after_cleaning']:,}")

    loans = con.execute("SELECT * FROM loans").df()
    panel = con.execute('SELECT * FROM panel').df()
    n_perf = con.execute("SELECT count(*) FROM perf").fetchone()[0]
    print(f"[rows] sampled loans: {len(loans):,}   monthly perf rows (sample): {n_perf:,}   loan-quarters: {len(panel):,}")

    panel["split"] = assign_split(panel["qidx"], cfg).values
    out["loans"] = int(len(loans))
    out["monthly_rows"] = int(n_perf)
    out["loan_quarters"] = int(len(panel))
    out["default_events"] = int(panel["default"].sum())
    out["prepayment_events"] = int(panel["prepay"].sum())
    out["other_exit_events"] = int(panel["exit_other"].sum())
    out["vintage_range"] = [int(loans["vintage"].min()), int(loans["vintage"].max())]
    out["loans_per_vintage"] = {int(k): int(v) for k, v in loans.groupby("vintage").size().items()}
    out["observation_window"] = [idx_to_q(int(panel["qidx"].min())), idx_to_q(int(panel["qidx"].max()))]
    by_split = panel.groupby("split").agg(loan_quarters=("default", "size"), defaults=("default", "sum"))
    by_split["default_rate"] = by_split["defaults"] / by_split["loan_quarters"]
    out["by_split"] = {k: {"loan_quarters": int(r.loan_quarters), "defaults": int(r.defaults),
                           "default_rate_per_quarter": float(r.default_rate)} for k, r in by_split.iterrows()}

    m = panel.merge(loans[["loan_id", "vintage"]], on="loan_id")
    ever = m.groupby("loan_id").agg(vintage=("vintage", "first"), d=("default", "max"), p=("prepay", "max"))
    v = ever.groupby("vintage").agg(loans=("d", "size"), defaulted=("d", "sum"), prepaid=("p", "sum"))
    v["ever_default_rate"] = v["defaulted"] / v["loans"]
    out["default_rate_by_origination_year"] = {
        int(k): {"loans_in_panel": int(r.loans), "defaulted": int(r.defaulted), "ever_default_rate": float(r.ever_default_rate)}
        for k, r in v.iterrows()
    }
    losses = con.execute("SELECT count(*) FROM losses").fetchone()[0]
    out["loss_records_credit_event"] = int(losses)

    print(f"[rows] default events: {out['default_events']:,}   prepayments: {out['prepayment_events']:,}")
    for k, r in out["by_split"].items():
        print(f"       split {k:<10} loan-quarters {r['loan_quarters']:>9,}  defaults {r['defaults']:>6,}")
    print(f"       ever-default rate by vintage: " +
          ", ".join(f"{k}: {r['ever_default_rate']:.1%}" for k, r in out["default_rate_by_origination_year"].items()))
    if out["by_split"].get("train", {}).get("defaults", 0) < 500:
        print("WARNING: fewer than 500 training defaults - add more pre-crisis vintages (guide, Phase 1).")
    return out, panel, loans


def plot_default_rate(panel: pd.DataFrame, cfg: dict, path: Path) -> None:
    g = panel.groupby("qidx").agg(n=("default", "size"), d=("default", "sum"))
    g["rate"] = g["d"] / g["n"]
    g = g[g["n"] >= 500]  # avoid noisy tails
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(g.index / 4, g["rate"] * 100, lw=1.8, color="#1f4e79")
    c0, c1 = (q_to_idx(x) for x in cfg["splits"]["covid_exclude"])
    ax.axvspan(c0 / 4, (c1 + 1) / 4, color="grey", alpha=0.25, label="COVID forbearance (excluded)")
    for key, color in (("train_end", "#2ca02c"), ("val_end", "#ff7f0e")):
        ax.axvline((q_to_idx(cfg["splits"][key]) + 1) / 4, color=color, ls="--", lw=1, label=key.replace("_", " "))
    ax.set_xlabel("calendar quarter")
    ax.set_ylabel("default rate per quarter (%)")
    ax.set_title("CreditCast sample: quarterly default rate (90+ DPD or credit-event termination)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    peak = g["rate"].idxmax()
    print(f"[plot] saved {path}; peak quarterly default rate {g['rate'].max():.3%} in {idx_to_q(int(peak))}")


def main() -> None:
    cfg = load_config()
    (ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)
    con = build(cfg)
    summary, panel, loans = summarise(con, cfg)
    for t in ("loans", "panel", "losses"):
        order = {"loans": "loan_id", "panel": "loan_id, qidx", "losses": "loan_id"}[t]   # deterministic row order
        con.execute(f"COPY (SELECT * FROM {t} ORDER BY {order}) TO '{ROOT / 'data' / 'interim' / (t + '.parquet')}' (FORMAT PARQUET)")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_summary.json").write_text(json.dumps(summary, indent=2))
    plot_default_rate(panel, cfg, ROOT / "results" / "figures" / "default_rate_by_quarter.png")
    print("[done] results/data_summary.json written")


if __name__ == "__main__":
    main()
