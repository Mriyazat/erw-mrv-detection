"""Resin-target feature engineering (sensor + weather aggregates over windows).

Aggregates co-located sensor channels at the matching (corrected) depth and
weather over each resin capsule's deployment window. Shared by the
sensor->resin and multitask phases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CACHE_DIR
from src.io.load_resin import load_resin

SENSOR_CHANNELS = ["vwc", "temp", "ec", "mp"]
WEATHER_AGG = {"air_temp_c": "mean", "rain_mm": "sum", "et0_mm": "sum",
               "vpd_kpa": "mean", "solar_rad_kjm2": "sum"}
TARGET_IONS = ["ca_ppm", "mg_ppm", "k_ppm", "na_ppm", "s_ppm"]


def build_resin_feature_table() -> tuple[pd.DataFrame, list[str]]:
    """Return (feature_table, feature_column_names)."""
    resin = load_resin()
    resin = resin[(resin["qa_flag"] == "") & (resin["treatment"] != "unknown")].copy()
    sensors = pd.read_parquet(CACHE_DIR / "sensors.parquet")
    weather = pd.read_parquet(CACHE_DIR / "weather_15min.parquet")

    rows = []
    for _, r in resin.iterrows():
        ph, depth = r["plot_half"], int(r["depth_cm"])
        t0, t1 = r["deploy_start"], r["deploy_end"]
        win = sensors[(sensors["plot_id"] == ph)
                      & (sensors["timestamp"] >= t0)
                      & (sensors["timestamp"] < t1)]
        feat = {
            "plot_id": int(r["plot_id"]), "plot_half": ph, "depth_cm": depth,
            "round": int(r["round"]), "treatment": r["treatment"],
            "days_deployed": float(r["days_deployed"]),
            "n_sensor_rows": len(win),
        }
        for ion in TARGET_IONS:
            feat[ion] = r[ion]
        for ch in SENSOR_CHANNELS:
            col = f"{ch}_{depth}"
            s = win[col].dropna() if col in win else pd.Series(dtype=float)
            for stat in ("mean", "std", "min", "max"):
                feat[f"{ch}_{stat}"] = getattr(s, stat)() if len(s) else np.nan
        wwin = weather[(weather["timestamp"] >= t0) & (weather["timestamp"] < t1)]
        for col, agg in WEATHER_AGG.items():
            feat[f"wx_{col}"] = (getattr(wwin[col], agg)()
                                 if col in wwin and len(wwin) else np.nan)
        rows.append(feat)
    feats = pd.DataFrame(rows)
    feature_cols = [c for c in feats.columns
                    if c.startswith(("vwc_", "temp_", "ec_", "mp_", "wx_"))
                    or c == "days_deployed"]
    return feats, feature_cols
