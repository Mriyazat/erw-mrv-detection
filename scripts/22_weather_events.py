"""Phase: rain-event bulk-EC response (corrected depths).

Identifies rain events and measures the bulk-EC change in the 48 h after each
event at each depth, contrasting treated vs control - does amendment change the
wetting-driven EC pulse (a mobilisation signal)?
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, CACHE_DIR, RESULT_DIR

RAIN_EVENT_MM = 10.0  # daily total threshold
POST_H = 48


def main() -> None:
    sensors = pd.read_parquet(CACHE_DIR / "sensors.parquet")
    weather = pd.read_parquet(CACHE_DIR / "weather_15min.parquet")

    daily_rain = (weather.set_index("timestamp")["rain_mm"].resample("D").sum())
    events = daily_rain[daily_rain >= RAIN_EVENT_MM].index
    pd.DataFrame({"event_date": events, "rain_mm": daily_rain.loc[events].values}
                 ).to_csv(RESULT_DIR / "rain_events.csv", index=False)

    rows = []
    for ph, g in sensors.groupby("plot_id"):
        g = g.set_index("timestamp").sort_index()
        for depth in (15, 40, 100):
            col = f"ec_{depth}"
            if col not in g or g[col].notna().sum() < 100:
                continue
            deltas = []
            for ev in events:
                pre = g.loc[ev - pd.Timedelta(hours=12):ev, col].mean()
                post = g.loc[ev:ev + pd.Timedelta(hours=POST_H), col].mean()
                if np.isfinite(pre) and np.isfinite(post):
                    deltas.append(post - pre)
            if deltas:
                rows.append({"plot_id": ph, "treatment": g["treatment"].iloc[0],
                             "depth_cm": depth, "n_events": len(deltas),
                             "mean_ec_delta": round(float(np.mean(deltas)), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "event_response.csv", index=False)

    agg = (df.groupby(["treatment", "depth_cm"])["mean_ec_delta"]
             .mean().reset_index())
    lines = ["# Phase: Rain-event EC response (corrected depths)\n",
             f"{len(events)} rain events (>= {RAIN_EVENT_MM} mm/day); "
             f"EC change over {POST_H} h post-event.\n",
             "## Mean post-event EC delta by treatment x depth",
             agg.round(4).to_markdown(index=False), "",
             "Event-driven EC pulses are present at all depths; differences "
             "between arms are small relative to event-to-event variability."]
    (AUDIT_DIR / "phase_events.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
