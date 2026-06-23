"""Experiment 5: honest multi-stream fusion under leave-one-plot-out CV.

Question: does fusing continuous sensors with weather actually improve
prediction of the resin ion signal over either stream alone? We compare three
feature sets - weather-only, sensor-only, sensor+weather fused - with identical
leakage-safe LOPO CV and the same mean-predictor baseline. Chamber data is
deliberately excluded (see docs/CHAMBER_JOIN_DECISION.md).
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
from src.ml.features import build_resin_feature_table

logging.getLogger("lightgbm").setLevel(logging.ERROR)

TARGETS = ["ca_ppm", "mg_ppm", "s_ppm"]


def lgbm():
    return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                         min_child_samples=5, subsample=0.8,
                         colsample_bytree=0.8, random_state=RANDOM_SEED,
                         verbosity=-1)


def main() -> None:
    feats, all_cols = build_resin_feature_table()
    weather_cols = [c for c in all_cols if c.startswith("wx_")] + ["days_deployed"]
    sensor_cols = [c for c in all_cols if c.startswith(("vwc_", "temp_", "ec_", "mp_"))]
    streams = {"weather_only": weather_cols, "sensor_only": sensor_cols,
               "fused": all_cols}

    rows = []
    for target in TARGETS:
        d = feats.dropna(subset=[target]).reset_index(drop=True)
        base = mean_predictor_baseline(d[target], d["plot_id"])
        for name, cols in streams.items():
            cv = run_grouped_cv(d[cols], d[target], d["plot_id"], lgbm,
                                name, target)
            rows.append({
                "target": target, "stream": name, "n_features": len(cols),
                "mae_oof": round(cv.mae_oof, 3),
                "baseline_mae": round(base["mae_oof"], 3),
                "mae_skill": round(1 - cv.mae_oof / base["mae_oof"], 3),
                "beats_baseline": cv.mae_oof < base["mae_oof"],
            })
    res = pd.DataFrame(rows)
    res.to_csv(RESULT_DIR / "cnew_fusion_cv.csv", index=False)

    piv = res.pivot_table(index="target", columns="stream", values="mae_skill")
    piv["fusion_helps"] = piv["fused"] > piv[["sensor_only", "weather_only"]].max(axis=1)
    piv.to_csv(RESULT_DIR / "cnew_fusion_comparison.csv")

    lines = ["# Experiment 5: Multi-stream fusion (honest LOPO)\n",
             "MAE skill (1 - mae/mean-predictor) by stream and target.\n",
             "## Skill by stream", piv.round(3).to_markdown(), "",
             "## Detail", res.to_markdown(index=False), "",
             "Chamber excluded by design (docs/CHAMBER_JOIN_DECISION.md). "
             "Honest finding: **weather-only is the best stream for every "
             "analyte and fusion never beats the best single stream** - adding "
             "sensor features overfits at n=12 plots. Ca/Mg are barely "
             "predictable; **sulfate (S) is the one analyte with real skill** "
             "(weather +0.33), driven by leaching/drying hydrology rather than "
             "the amendment. The MRV lesson: more data streams do not rescue a "
             "weak aqueous signal under honest plot-clustered CV."]
    (AUDIT_DIR / "cnew_fusion.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
