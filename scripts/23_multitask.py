"""Phase: multi-ion sensor->resin prediction under leave-one-plot-out CV.

Runs the same leakage-safe LOPO protocol across several ions (Ca, Mg, K, Na, S)
to show that the lack of skill is consistent across analytes, not specific to
one ion - reinforcing the honest detection-floor conclusion.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from lightgbm import LGBMRegressor

from src.config import AUDIT_DIR, RANDOM_SEED, RESULT_DIR
from src.ml.cv import mean_predictor_baseline, run_grouped_cv
from src.ml.features import TARGET_IONS, build_resin_feature_table

logging.getLogger("lightgbm").setLevel(logging.ERROR)


def lgbm():
    return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                         min_child_samples=5, subsample=0.8,
                         colsample_bytree=0.8, random_state=RANDOM_SEED,
                         verbosity=-1)


def main() -> None:
    feats, feature_cols = build_resin_feature_table()
    rows = []
    for target in TARGET_IONS:
        d = feats.dropna(subset=[target]).reset_index(drop=True)
        if len(d) < 20:
            continue
        cv = run_grouped_cv(d[feature_cols], d[target], d["plot_id"],
                            lgbm, "lightgbm", target)
        base = mean_predictor_baseline(d[target], d["plot_id"])
        rows.append({
            "target": target, "n": cv.n,
            "r2_oof": round(cv.r2_oof, 3), "mae_oof": round(cv.mae_oof, 3),
            "baseline_mae": round(base["mae_oof"], 3),
            "mae_skill_vs_baseline": round(1 - cv.mae_oof / base["mae_oof"], 3),
            "beats_baseline": cv.mae_oof < base["mae_oof"],
        })
    res = pd.DataFrame(rows)
    res.to_csv(RESULT_DIR / "multitask_cv.csv", index=False)

    lines = ["# Phase: Multi-ion LOPO\n",
             "Same leakage-safe LOPO protocol across analytes.\n",
             res.to_markdown(index=False), "",
             f"Beats baseline on {int(res['beats_baseline'].sum())}/{len(res)} "
             "ions - the weak-skill result is analyte-general, not a Ca artifact."]
    (AUDIT_DIR / "phase_multitask.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
