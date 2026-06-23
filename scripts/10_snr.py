"""Phase: theoretical first-principles SNR table + sigma sweep."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import AUDIT_DIR, RESULT_DIR
from src.physics.snr_model import compute_full_snr_table


def main() -> None:
    base = compute_full_snr_table()
    base.to_csv(RESULT_DIR / "snr_table.csv", index=False)

    sweeps = []
    for mult in (0.5, 1.0, 2.0, 5.0, 10.0):
        t = compute_full_snr_table(sigma_multiplier=mult)
        t["sigma_mult"] = mult
        sweeps.append(t)
    sweep = pd.concat(sweeps, ignore_index=True)
    sweep.to_csv(RESULT_DIR / "snr_sigma_sweep.csv", index=False)

    piv = base.pivot_table(values="SNR", index=["treatment", "depth_cm"],
                           columns="ion")
    zmax = (base[base["treatment"] == "60"]
            .pivot_table(values="z_max_snr3_m", index="depth_cm", columns="ion"))

    lines = ["# Phase: Theoretical SNR\n",
             "First-principles signal-to-noise for the 50:50 wollastonite + "
             "diopside amendment. SNR = depth-attenuated annual ion flux / "
             "natural flux sigma. SNR>=3 ~ detectable.\n",
             "## SNR by treatment x depth x ion", piv.round(2).to_markdown(), "",
             "## Max detectable depth (m) at 60 t/ha, SNR>=3",
             zmax.round(2).to_markdown(), "",
             "Ca is the strongest channel; Mg weaker (lower flux, slower diopside); "
             "Si strong in theory but NOT measurable by the resin panel."]
    (AUDIT_DIR / "phase_snr.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
