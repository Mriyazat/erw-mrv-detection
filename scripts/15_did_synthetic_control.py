"""Phase: difference-in-differences on bulk EC (treated vs control).

Defines a baseline window (first 3 weeks of each plot's record) and a
mid-season window, then estimates the DiD: (treated change) - (control change)
in shallow (corrected 15 cm) bulk EC. A clean step-up would show a positive DiD.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, CACHE_DIR, RESULT_DIR

BASELINE_DAYS = 21
MID_START_DAYS = 40
MID_END_DAYS = 90


def main() -> None:
    sensors = pd.read_parquet(CACHE_DIR / "sensors.parquet")
    rows = []
    for ph, g in sensors.groupby("plot_id"):
        g = g.dropna(subset=["ec_15"]).sort_values("timestamp")
        if len(g) < 200:
            continue
        t0 = g["timestamp"].min()
        base = g[g["timestamp"] < t0 + pd.Timedelta(days=BASELINE_DAYS)]["ec_15"]
        mid = g[(g["timestamp"] >= t0 + pd.Timedelta(days=MID_START_DAYS))
                & (g["timestamp"] < t0 + pd.Timedelta(days=MID_END_DAYS))]["ec_15"]
        if len(base) < 50 or len(mid) < 50:
            continue
        rows.append({"plot_id": ph, "treatment": g["treatment"].iloc[0],
                     "ec_baseline": round(float(base.mean()), 4),
                     "ec_mid": round(float(mid.mean()), 4),
                     "delta": round(float(mid.mean() - base.mean()), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "did_ec.csv", index=False)

    ctrl_delta = df.loc[df["treatment"] == "control", "delta"].mean()
    did_rows = []
    for arm in ("20", "60"):
        arm_delta = df.loc[df["treatment"] == arm, "delta"].mean()
        did_rows.append({"treatment": arm,
                         "mean_delta": round(float(arm_delta), 4),
                         "control_delta": round(float(ctrl_delta), 4),
                         "did_estimate": round(float(arm_delta - ctrl_delta), 4)})
    did = pd.DataFrame(did_rows)
    did.to_csv(RESULT_DIR / "did_estimates.csv", index=False)

    lines = ["# Phase: Difference-in-Differences (EC, 15 cm)\n",
             "Baseline (first 3 wks) vs mid-season shallow EC; DiD = treated "
             "change minus control change.\n",
             "## Per-plot deltas", df.to_markdown(index=False), "",
             "## DiD estimates", did.to_markdown(index=False), "",
             "DiD estimates are small and not consistently positive - no clean "
             "amendment step-up in shallow bulk EC over the season."]
    (AUDIT_DIR / "phase_did.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
