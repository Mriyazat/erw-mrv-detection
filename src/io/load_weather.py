"""Simcoe Research Station weather loader.

Reads the 15-min Raw Data sheet from each weather workbook, harmonises
columns, computes VPD (FAO Tetens) and a solar-share ET0 proxy.

VERIFICATION HARDENING: merge_sensor_weather records the explicit weather
cutoff and how many sensor rows fall outside weather coverage, returning
that policy info rather than silently dropping ~18.6% of rows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, WEATHER_DIR

logger = logging.getLogger(__name__)

WEATHER_FILES = [
    WEATHER_DIR / "weather_2025.xlsx",
    WEATHER_DIR / "weather_2026.xlsx",
]
SHEET_15MIN = "15 Min Raw Data"

COLUMN_MAP = {
    0:  "timestamp", 1: "air_temp_c", 2: "rh_pct", 3: "rain_mm",
    4:  "leaf_wetness", 5: "solar_rad_kjm2", 6: "wind_spd_kmh",
    7:  "wind_dir_deg", 8: "grass_temp_c", 9: "soil_temp_5_c",
    10: "soil_temp_10_c", 11: "soil_temp_15_c", 12: "soil_temp_30_c",
}

CACHE_PATH = CACHE_DIR / "weather_15min.parquet"


def _load_15min_sheet(filepath: Path) -> pd.DataFrame:
    raw = pd.read_excel(filepath, sheet_name=SHEET_15MIN, header=None, skiprows=2)
    if raw.shape[1] < len(COLUMN_MAP):
        raise ValueError(f"{filepath}: expected >={len(COLUMN_MAP)} cols, "
                         f"got {raw.shape[1]}")
    raw = raw.iloc[:, : len(COLUMN_MAP)].copy()
    raw.columns = [COLUMN_MAP[i] for i in range(len(COLUMN_MAP))]

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    for col in raw.columns:
        if col != "timestamp":
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.replace({-999: np.nan, -9999: np.nan})
    raw["rain_mm"] = raw["rain_mm"].clip(lower=0)
    raw["rh_pct"] = raw["rh_pct"].clip(lower=0, upper=100)
    raw.loc[(raw["wind_dir_deg"] < 0) | (raw["wind_dir_deg"] > 360),
            "wind_dir_deg"] = np.nan
    return raw


def compute_vpd(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    ea = es * rh_pct.clip(lower=0, upper=100) / 100.0
    return (es - ea).clip(lower=0)


def compute_et0_15min(weather_15: pd.DataFrame) -> pd.Series:
    df = weather_15.copy().set_index("timestamp")
    daily = df.resample("D").agg(
        solar_kj=("solar_rad_kjm2", "sum"),
        tmean=("air_temp_c", "mean"),
        tmax=("air_temp_c", "max"),
        tmin=("air_temp_c", "min"),
    )
    daily["et0_mm_day"] = 0.408 * (daily["solar_kj"] / 1000.0)
    df["solar_share"] = (
        df["solar_rad_kjm2"]
        / df.groupby(df.index.date)["solar_rad_kjm2"].transform("sum").replace(0, np.nan)
    )
    df["et0_mm_15min"] = df["solar_share"] * df.index.normalize().map(
        daily["et0_mm_day"].to_dict()
    )
    return df["et0_mm_15min"].fillna(0).reset_index(drop=True)


def load_weather_15min(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and CACHE_PATH.exists():
        logger.info("Loading cached 15-min weather from %s", CACHE_PATH)
        return pd.read_parquet(CACHE_PATH)

    frames = []
    for fp in WEATHER_FILES:
        if not fp.exists():
            logger.warning("Weather file missing: %s", fp)
            continue
        logger.info("Loading %s", fp.name)
        df = _load_15min_sheet(fp)
        logger.info("  -> %d rows  [%s -> %s]",
                    len(df), df["timestamp"].min(), df["timestamp"].max())
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No weather files found")

    combined = (pd.concat(frames, ignore_index=True)
                  .sort_values("timestamp")
                  .drop_duplicates("timestamp", keep="last")
                  .reset_index(drop=True))
    combined["vpd_kpa"] = compute_vpd(combined["air_temp_c"], combined["rh_pct"])
    combined["et0_mm"]  = compute_et0_15min(combined)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(CACHE_PATH, index=False)
    logger.info("Cached %d weather rows to %s", len(combined), CACHE_PATH)
    return combined


def merge_sensor_weather(
    sensor_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    drop_unmatched: bool = True,
    tolerance: str = "7min",
    key_channel: str = "air_temp_c",
) -> tuple[pd.DataFrame, dict]:
    """Left-join 15-min weather onto sensor rows (nearest-asof).

    Returns (merged_df, policy) where `policy` records the weather cutoff and
    exactly how many sensor rows fell outside weather coverage - so the
    ~18.6% drop is an explicit, logged decision rather than silent loss.
    """
    s = sensor_df.sort_values("timestamp")
    w = weather_df.sort_values("timestamp")
    weather_max = w["timestamp"].max()
    weather_min = w["timestamp"].min()
    n_sensor = len(s)
    n_after_cutoff = int((s["timestamp"] > weather_max).sum())

    merged = pd.merge_asof(s, w, on="timestamp", direction="nearest",
                           tolerance=pd.Timedelta(tolerance))
    coverage = merged[w.columns.drop("timestamp")].notna().mean().mean()
    n_unmatched = int(merged[key_channel].isna().sum())
    policy = {
        "weather_min": weather_min, "weather_max": weather_max,
        "n_sensor_rows": n_sensor,
        "n_rows_after_weather_cutoff": n_after_cutoff,
        "n_unmatched_total": n_unmatched,
        "frac_dropped": round(n_unmatched / max(n_sensor, 1), 4),
        "match_coverage": round(float(coverage), 4),
        "drop_unmatched": drop_unmatched,
    }
    logger.info("Weather coverage to %s; %d/%d sensor rows past cutoff "
                "(%.1f%% of all rows unmatched)", weather_max,
                n_after_cutoff, n_sensor, 100 * policy["frac_dropped"])
    if drop_unmatched:
        merged = merged.dropna(subset=[key_channel]).reset_index(drop=True)
    return merged, policy


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    w = load_weather_15min(use_cache=False)
    print(f"\nShape: {w.shape}")
    print(f"Time range: {w.timestamp.min()} -> {w.timestamp.max()}")
