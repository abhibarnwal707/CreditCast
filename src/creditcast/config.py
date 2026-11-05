"""Config + calendar-quarter helpers shared across the project."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def q_to_idx(label: str) -> int:
    """'2014Q4' -> integer quarter index (year*4 + quarter-1)."""
    y, q = label.upper().split("Q")
    return int(y) * 4 + int(q) - 1


def idx_to_q(idx: int) -> str:
    return f"{idx // 4}Q{idx % 4 + 1}"


def assign_split(qidx, cfg: dict):
    """Split label for outcome-quarter index (pandas Series): train / validation / oot / covid / other."""
    import numpy as np
    import pandas as pd

    s = cfg["splits"]
    q = pd.Series(qidx)
    c0, c1 = (q_to_idx(x) for x in s["covid_exclude"])
    conds = [
        (q >= c0) & (q <= c1),
        q <= q_to_idx(s["train_end"]),
        (q >= q_to_idx(s["val_start"])) & (q <= q_to_idx(s["val_end"])),
        (q >= q_to_idx(s["oot_start"])) & (q <= q_to_idx(s["oot_end"])),
    ]
    return pd.Series(np.select(conds, ["covid", "train", "validation", "oot"], default="other"), index=q.index)
