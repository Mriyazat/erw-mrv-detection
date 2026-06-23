"""Experiment 3: aqueous-phase detection-budget across ions x depths x seasons.

Builds the per-cell variability table from the resin control arm (by ion,
depth, and round=season) and runs the reusable detection-budget calculator to
produce a design surface: power at the current N=4, MDE, and the replication
needed for 80% power. This is the deliverable tool that generalises the
one-off power phase into a planning instrument for future ERW trials.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, FIGURE_DIR, RESULT_DIR, RESIN_PRIMARY_IONS
from src.analysis.detection_budget import detection_budget, mde_for_power
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin

IONS = RESIN_PRIMARY_IONS + ["k_ppm", "na_ppm", "s_ppm"]
SEASON = {1: "Jul", 2: "Aug", 3: "Sep-Oct"}


def main() -> None:
    resin = qa_clean(load_resin())
    cells = []
    for ion in IONS:
        for depth in (15, 40, 100):
            for rnd in (1, 2, 3):
                ctrl = resin[(resin["treatment"] == "control")
                             & (resin["depth_cm"] == depth)
                             & (resin["round"] == rnd)][ion].dropna()
                if len(ctrl) >= 3 and ctrl.std(ddof=1) > 0:
                    cells.append({"ion": ion, "depth_cm": depth, "season": SEASON[rnd],
                                  "control_mean": round(float(ctrl.mean()), 3),
                                  "control_sd": round(float(ctrl.std(ddof=1)), 3)})
    cell_df = pd.DataFrame(cells)

    budget = detection_budget(cell_df, n_per_arm=4)
    budget.to_csv(RESULT_DIR / "cnew_detection_budget.csv", index=False)

    # replication curve: N needed for 80% power vs effect size
    eff = np.linspace(0.3, 3.0, 28)
    from src.analysis.detection_budget import n_for_power
    repl = pd.DataFrame({"effect_sd": eff,
                         "n_per_arm_for_80pct": [n_for_power(e) for e in eff]})
    repl.to_csv(RESULT_DIR / "cnew_detection_replication_curve.csv", index=False)

    mde4 = mde_for_power(4)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eff, repl["n_per_arm_for_80pct"], "-o", ms=3)
    ax.axhline(4, color="r", ls="--", label="current design (N=4)")
    ax.axvline(mde4, color="g", ls=":", label=f"MDE at N=4 ({mde4:.2f} SD)")
    ax.set_xlabel("True effect (control-SD units)")
    ax.set_ylabel("N plot-halves/arm for 80% power")
    ax.set_title("Experiment 3: ERW aqueous detection budget")
    ax.set_ylim(0, 40)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cnew_detection_budget.png", dpi=300)
    plt.close(fig)

    summary = (budget[budget["ion"].isin(RESIN_PRIMARY_IONS)
                      & (budget["target_effect_sd"] == 1.0)]
               .groupby(["ion", "depth_cm"])
               .agg(mean_power_1sd=("power", "mean"),
                    mde_sd=("mde_sd_at_design", "first"))
               .reset_index())

    lines = ["# Experiment 3: Aqueous-phase detection budget (reusable tool)\n",
             f"Per-cell control variability across {len(cell_df)} "
             "ion x depth x season cells -> design surface.\n",
             f"## MDE at current design (N=4): **{mde4:.2f} control-SD** "
             "(two-sided t, alpha=0.05, 80% power)\n",
             "## Mean power to detect a 1-SD shift, primary ions",
             summary.round(3).to_markdown(index=False), "",
             "## Replication needed (excerpt)",
             repl.iloc[::6].round(2).to_markdown(index=False), "",
             "The tool maps any (ion, depth, season, assumed effect) to power and "
             "required replication - directly reusable to size future ERW MRV "
             "campaigns. At N=4 only effects >~1.6 SD are reliably detectable.",
             "", f"Figure: {FIGURE_DIR/'cnew_detection_budget.png'}"]
    (AUDIT_DIR / "cnew_detection_budget.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
