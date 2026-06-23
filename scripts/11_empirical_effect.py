"""Phase: empirical resin effect sizes, dose-response, pooled bootstrap CIs.

All effect sizes are Hedges' g; dose-response in canonical ppm/(t/ha). CIs are
plot-clustered block bootstrap. Resin depths are label-based (unaffected by the
sensor depth correction).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import AUDIT_DIR, RESULT_DIR, RESIN_NEGATIVE_CONTROLS, RESIN_PRIMARY_IONS
from src.analysis.effects import (
    compute_dose_response,
    compute_effects,
    pooled_effect_with_ci,
)
from src.io.load_resin import load_resin

IONS = RESIN_PRIMARY_IONS + RESIN_NEGATIVE_CONTROLS + ["s_ppm"]


def main() -> None:
    resin = load_resin()

    effects = compute_effects(resin, IONS)
    effects.to_csv(RESULT_DIR / "empirical_effects.csv", index=False)

    dose = compute_dose_response(resin, IONS)
    dose.to_csv(RESULT_DIR / "empirical_dose_response.csv", index=False)

    pooled_rows = []
    for ion in RESIN_PRIMARY_IONS + RESIN_NEGATIVE_CONTROLS:
        for arm in ("20", "60"):
            for depth in (None, 15, 40, 100):
                pooled_rows.append(pooled_effect_with_ci(resin, ion, arm, depth))
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(RESULT_DIR / "empirical_pooled_ci.csv", index=False)

    sig = effects[(effects["hedges_g"].abs() >= 0.8)
                  & (effects["n_t"] >= 3) & (effects["n_c"] >= 3)]

    lines = ["# Phase: Empirical Effects\n",
             f"Resin QA-clean cells: {len(effects)} effect-size rows, "
             f"{len(dose)} dose-response fits.\n",
             "## Primary-ion pooled Hedges' g (plot-clustered bootstrap CI)",
             pooled[pooled["ion"].isin(RESIN_PRIMARY_IONS)][
                 ["ion", "treatment", "depth_cm", "stat", "lo", "hi", "n_blocks"]
             ].round(3).to_markdown(index=False), "",
             "## Large effects (|g|>=0.8, n>=3/arm)",
             (sig[["ion", "round", "depth_cm", "treatment", "hedges_g",
                   "mean_diff"]].round(3).to_markdown(index=False)
              if len(sig) else "_none_"), "",
             "Note: many cells are small/negative - consistent with a weak, "
             "depth- and time-dependent aqueous signal against high natural "
             "variability (motivates the detection-budget contribution)."]
    (AUDIT_DIR / "phase_empirical.md").write_text("\n".join(lines))
    print("\n".join(lines[:14]))
    print(f"... wrote 3 CSVs + {AUDIT_DIR/'phase_empirical.md'}")


if __name__ == "__main__":
    main()
