"""Build all parquet caches from raw xlsx (idempotent).

Produces in outputs/cache/:
  sensors.parquet, sensors_audit.parquet, resin.parquet, weather_15min.parquet,
  chamber.parquet, plot_metadata.parquet, sensor_weather.parquet
and records the weather-join policy (explicit cutoff, no silent drop) in
outputs/cache/weather_join_policy.json.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import CACHE_DIR
from src.io.load_sensors import load_all_sensors
from src.io.load_resin import load_resin
from src.io.load_weather import load_weather_15min, merge_sensor_weather
from src.io.load_chamber import load_chamber
from src.io.load_plot_metadata import load_plot_metadata

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("build_caches")


def main(rebuild: bool = True) -> None:
    use_cache = not rebuild

    log.info("[1/6] sensors")
    sensors, audit = load_all_sensors(use_cache=use_cache)

    log.info("[2/6] resin")
    resin = load_resin(use_cache=use_cache)

    log.info("[3/6] weather")
    weather = load_weather_15min(use_cache=use_cache)

    log.info("[4/6] chamber")
    chamber = load_chamber(use_cache=use_cache)

    log.info("[5/6] plot metadata")
    meta = load_plot_metadata(use_cache=use_cache)

    log.info("[6/6] sensor x weather join (explicit cutoff policy)")
    merged, policy = merge_sensor_weather(sensors, weather, drop_unmatched=True)
    merged.to_parquet(CACHE_DIR / "sensor_weather.parquet", index=False)
    policy_out = {k: (str(v) if isinstance(v, pd.Timestamp) else v)
                  for k, v in policy.items()}
    (CACHE_DIR / "weather_join_policy.json").write_text(
        json.dumps(policy_out, indent=2))

    print("\n=== Cache build summary ===")
    print(f"  sensors          {sensors.shape}")
    print(f"  sensors_audit    {audit.shape}  "
          f"(skipped configs: {int((~audit['kept']).sum()) if 'kept' in audit else 0})")
    print(f"  resin            {resin.shape}")
    print(f"  weather_15min    {weather.shape}")
    print(f"  chamber          {chamber.shape}")
    print(f"  plot_metadata    {meta.shape}")
    print(f"  sensor_weather   {merged.shape}")
    print(f"\n  weather join policy: {json.dumps(policy_out, indent=2)}")


if __name__ == "__main__":
    rebuild = "--use-cache" not in sys.argv
    main(rebuild=rebuild)
