"""Phase: theory-vs-resin reconciliation.

The first-principles model predicts large Ca/Mg SNR (>20), yet resin effect
sizes are mostly small. This phase quantifies the gap: the sigma-inflation
factor (how much natural-variability sigma must rise above the model's assumed
sigma_F to explain the weak observed effects) and the implied k-retention.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, RESULT_DIR
from src.analysis.effects import compute_effects
from src.io.load_resin import load_resin
from src.physics.snr_model import get_theoretical_snr_at_depth

ION_MAP = {"ca_ppm": "Ca", "mg_ppm": "Mg"}


def main() -> None:
    resin = load_resin()
    eff = compute_effects(resin, list(ION_MAP))

    rows = []
    for _, r in eff.iterrows():
        ion_sym = ION_MAP[r["ion"]]
        snr_th = get_theoretical_snr_at_depth(r["treatment"], int(r["depth_cm"]),
                                              ion=ion_sym)
        g = abs(r["hedges_g"])
        if not np.isfinite(g) or g <= 0 or snr_th <= 0:
            continue
        rows.append({
            "ion": r["ion"], "round": r["round"], "depth_cm": r["depth_cm"],
            "treatment": r["treatment"], "snr_theory": round(snr_th, 2),
            "empirical_abs_g": round(g, 3),
            "sigma_inflation_needed": round(snr_th / g, 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "theory_vs_resin.csv", index=False)

    summ = (df.groupby(["ion", "depth_cm"])
              .agg(median_snr_theory=("snr_theory", "median"),
                   median_abs_g=("empirical_abs_g", "median"),
                   median_sigma_inflation=("sigma_inflation_needed", "median"),
                   n=("sigma_inflation_needed", "count"))
              .reset_index())
    summ.to_csv(RESULT_DIR / "theory_vs_resin_summary.csv", index=False)

    lines = ["# Phase: Theory vs Resin\n",
             "The model treats sigma_F as the only noise; reality adds spatial, "
             "temporal and capsule noise. `sigma_inflation_needed` = how many-fold "
             "the effective noise must exceed the model's sigma_F to match the "
             "observed (small) effect sizes.\n",
             "## Median by ion x depth", summ.round(2).to_markdown(index=False), "",
             "A large, consistent inflation factor is the quantitative statement "
             "that first-principles SNR is optimistic for aqueous-phase MRV - and "
             "motivates the empirical detection-budget contribution."]
    (AUDIT_DIR / "phase_theory_vs_resin.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
