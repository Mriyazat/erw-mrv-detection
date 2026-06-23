"""Smoke test: imports, config sanity, optional-dependency report."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config


def main() -> None:
    print("erw smoke test")
    print("-" * 40)
    print(f"PROJECT_ROOT      {config.PROJECT_ROOT}")
    print(f"IS_HPC            {config.IS_HPC}")
    print(f"RANDOM_SEED       {config.RANDOM_SEED}")
    print(f"canonical dose    {config.CANONICAL_DOSE_UNIT}  {config.DOSE_THA}")
    print(f"canonical effect  {config.CANONICAL_EFFECT_SIZE}")
    print(f"CV group col      {config.CV_GROUP_COL}")
    assert len(config.TREATMENT_MAP) == 12
    assert config.DOSE_KGM2["60"] == 6.0
    assert config.DOSE_THA["60"] == 60.0

    print("\nData files present:")
    for name, path in [
        ("sensors/plot_3E_2025", config.PLOT_FILES_2025["3E"]),
        ("resin R1R2", config.RESIN_FILES["R1_R2"]),
        ("weather 2025", config.WEATHER_DIR / "weather_2025.xlsx"),
        ("chamber", config.CHAMBER_DIR / "chamber.xlsx"),
    ]:
        print(f"  {'OK ' if path.exists() else 'MISSING'} {name}: {path}")

    print("\nOptional dependencies:")
    for m in ["lightgbm", "xgboost", "catboost", "numpyro", "jax",
              "ruptures", "shap", "statsmodels"]:
        try:
            mod = importlib.import_module(m)
            print(f"  OK  {m} {getattr(mod, '__version__', '')}")
        except Exception as e:
            print(f"  --  {m} MISSING ({type(e).__name__})")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
