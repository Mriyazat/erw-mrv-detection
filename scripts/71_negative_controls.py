"""Negative-control test: do amendment-derived ions behave unlike inert tracers?

A treatment "effect" on Ca/Mg is only credible if ions that the amendment does
NOT release (K, Na, NO3-N, NH4-N - biologically/geologically unrelated to
wollastonite + diopside) show no comparable dose response. This phase contrasts
the pooled treated(60)-vs-control Hedges' g (plot-clustered bootstrap CI) for:
  * products      : Ca, Mg          (released by the silicate amendment)
  * reactive-other : S              (the one FDR-surviving analyte; flagged)
  * inert controls : K, Na, NO3-N, NH4-N

A forest plot lets a reviewer see at a glance whether the product ions separate
from the inert controls or sit in the same null band.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import AUDIT_DIR, FIGURE_DIR, RESULT_DIR
from src.analysis.effects import pooled_effect_with_ci
from src.io.load_resin import load_resin

GROUPS = {
    "ca_ppm": "product", "mg_ppm": "product",
    "s_ppm": "reactive-other",
    "k_ppm": "inert-control", "na_ppm": "inert-control",
    "no3_n_ppm": "inert-control", "nh4_n_ppm": "inert-control",
}
LABEL = {"ca_ppm": "Ca", "mg_ppm": "Mg", "s_ppm": "S", "k_ppm": "K",
         "na_ppm": "Na", "no3_n_ppm": "NO3-N", "nh4_n_ppm": "NH4-N"}
GROUP_COLOR = {"product": "#1b7837", "reactive-other": "#998ec3",
               "inert-control": "#b2182b"}
GROUP_ORDER = ["product", "reactive-other", "inert-control"]


def main() -> None:
    resin = load_resin()

    rows = []
    for ion, grp in GROUPS.items():
        r = pooled_effect_with_ci(resin, ion, "60", depth_cm=None, n_boot=4000)
        rows.append({"ion": LABEL[ion], "group": grp, "hedges_g": r["stat"],
                     "ci_lo": r["lo"], "ci_hi": r["hi"], "se": r["se"],
                     "ci_excludes_0": bool(r["lo"] > 0 or r["hi"] < 0),
                     "n_blocks": r["n_blocks"]})
    df = pd.DataFrame(rows)
    df["_g"] = df["group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    df = df.sort_values(["_g", "hedges_g"]).drop(columns="_g").reset_index(drop=True)
    df.to_csv(RESULT_DIR / "negative_controls.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for y, row in enumerate(df.itertuples()):
        c = GROUP_COLOR[row.group]
        ax.plot([row.ci_lo, row.ci_hi], [y, y], "-", color=c, lw=2, alpha=0.85)
        ax.plot(row.hedges_g, y, "o", color=c, ms=7,
                mec="k" if row.ci_excludes_0 else c, mew=1.2)
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["ion"])
    ax.set_xlabel("Pooled Hedges' g, treated(60 t/ha) - control "
                  "(plot-clustered bootstrap 95% CI)")
    ax.set_title("Negative-control contrast: amendment products vs inert tracers")
    handles = [plt.Line2D([0], [0], color=GROUP_COLOR[g], lw=3,
                          label=g) for g in GROUP_ORDER]
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "negative_controls.png", dpi=130)
    plt.close(fig)

    prod = df[df["group"] == "product"]["hedges_g"]
    inert = df[df["group"] == "inert-control"]["hedges_g"]
    lines = ["# Phase: Negative-control contrast\n",
             "Pooled treated(60)-vs-control Hedges' g per ion (depths/rounds "
             "pooled, plot-clustered bootstrap 95% CI).\n",
             "## Pooled effect by ion",
             df[["ion", "group", "hedges_g", "ci_lo", "ci_hi",
                 "ci_excludes_0", "n_blocks"]].round(3).to_markdown(index=False),
             "",
             f"product mean g = {prod.mean():+.3f}; "
             f"inert-control mean g = {inert.mean():+.3f}.", "",
             "Reading: if product ions (Ca/Mg) do NOT separate from the inert "
             "controls (K/Na/NO3/NH4), the aqueous resin signal is not a clean "
             "amendment fingerprint at this dose/season - consistent with the "
             "FDR result (no effect cell survives) and the detection-budget "
             "argument. S is shown separately because it is the only FDR-"
             "surviving dose response and is not a silicate dissolution product.",
             "", f"Figure: {FIGURE_DIR/'negative_controls.png'}"]
    (AUDIT_DIR / "phase_negative_controls.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
