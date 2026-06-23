"""Phase: sensor -> resin ML under leave-one-PLOT-out CV (honest baseline).

For each resin capsule we aggregate the co-located sensor channels at the
MATCHING (corrected) depth over the capsule's deployment window, plus weather
aggregates, and try to predict the captured ion (Ca, Mg). Skill is reported
RELATIVE to a leave-one-plot-out mean predictor - the honest bar for n=12 plots.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.config import AUDIT_DIR, RANDOM_SEED, RESULT_DIR
from src.ml.cv import mean_predictor_baseline, run_grouped_cv
from src.ml.features import build_resin_feature_table

logging.getLogger("lightgbm").setLevel(logging.ERROR)


def lgbm():
    return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                         min_child_samples=5, subsample=0.8,
                         colsample_bytree=0.8, random_state=RANDOM_SEED,
                         verbosity=-1)


def main() -> None:
    feats, feature_cols = build_resin_feature_table()
    feats.to_csv(RESULT_DIR / "sensor_resin_features.csv", index=False)

    results = []
    for target in ("ca_ppm", "mg_ppm"):
        d = feats.dropna(subset=[target]).reset_index(drop=True)
        X, y, g = d[feature_cols], d[target], d["plot_id"]
        cv = run_grouped_cv(X, y, g, lgbm, "lightgbm", target)
        base = mean_predictor_baseline(y, g)
        results.append({
            "target": target, "model": "lightgbm", "n": cv.n,
            "r2_oof": round(cv.r2_oof, 3), "mae_oof": round(cv.mae_oof, 3),
            "baseline_r2": round(base["r2_oof"], 3),
            "baseline_mae": round(base["mae_oof"], 3),
            "mae_skill_vs_baseline": round(1 - cv.mae_oof / base["mae_oof"], 3),
            "beats_baseline": cv.mae_oof < base["mae_oof"],
        })
    res = pd.DataFrame(results)
    res.to_csv(RESULT_DIR / "sensor_resin_cv.csv", index=False)

    lines = ["# Phase: Sensor -> Resin ML (LOPO)\n",
             f"Feature table: {len(feats)} capsules x {len(feature_cols)} "
             "features (corrected-depth sensor aggregates + weather).\n",
             "## Leave-one-plot-out skill vs mean predictor",
             res.to_markdown(index=False), "",
             "Honest read: with 12 plots and a weak aqueous signal, the model "
             "should be judged ONLY against the leave-one-plot-out mean predictor "
             "(`mae_skill_vs_baseline` > 0 means it genuinely beats the mean)."]
    (AUDIT_DIR / "phase_sensor_resin.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
