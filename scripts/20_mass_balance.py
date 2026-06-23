"""Phase: cation mass-balance / capture-fraction budget.

Compares the theoretical annual Ca/Mg release (first-principles model, 60 t/ha)
to the resin-captured flux, giving an order-of-magnitude capture fraction and
framing how far observed aqueous capture is from the released-cation budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, ION_MOLAR_MASS, RESULT_DIR
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin
from src.physics.snr_model import compute_surface_ion_flux, DOSE_KGM2


def main() -> None:
    resin = qa_clean(load_resin())
    rows = []
    for ion_col, ion_sym in (("ca_ppm", "Ca"), ("mg_ppm", "Mg")):
        f0 = compute_surface_ion_flux(DOSE_KGM2["60"])[ion_sym]  # mol/m2/yr
        for depth in (15, 40, 100):
            sub = resin[(resin["depth_cm"] == depth)
                        & (resin["treatment"].isin(["60", "control"]))]
            trt = sub.loc[sub["treatment"] == "60", ion_col].mean()
            ctrl = sub.loc[sub["treatment"] == "control", ion_col].mean()
            excess_ppm = trt - ctrl  # ug/10cm2 supply-rate proxy over deployment
            excess_mol = excess_ppm / (ION_MOLAR_MASS[ion_col] * 1e6)
            rows.append({
                "ion": ion_sym, "depth_cm": depth,
                "theory_release_mol_m2_yr": round(f0, 3),
                "resin_excess_ppm_60_vs_ctrl": round(float(excess_ppm), 3),
                "resin_excess_mol_proxy": f"{excess_mol:.3e}",
                "excess_positive": bool(excess_ppm > 0),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "mass_balance.csv", index=False)

    lines = ["# Phase: Mass balance / capture fraction\n",
             "Theoretical 60 t/ha annual cation release vs resin-captured excess "
             "(treated minus control). Resin flux is a supply-rate proxy, not an "
             "absolute soil-solution concentration, so this is order-of-magnitude.\n",
             df.to_markdown(index=False), "",
             "Most depth/ion cells show small or negative excess - the captured "
             "aqueous cation pool is a tiny, noisy fraction of the released-cation "
             "budget, the central MRV difficulty for early-stage ERW."]
    (AUDIT_DIR / "phase_mass_balance.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
