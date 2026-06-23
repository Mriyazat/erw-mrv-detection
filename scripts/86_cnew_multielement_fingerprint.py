"""Experiment 16: multi-element geochemical fingerprint (trace metals + pH-sensitive ions).

Extends the Ca:Mg feedstock fingerprint (Experiment 12) to the full resin element
panel, asking whether the treated(60)-minus-control excess VECTOR matches the
geochemical signature an alkalinity-generating silicate amendment should leave -
not just the cations it adds, but the pH-driven changes it forces:

  expected sign of treated-control excess
  ---------------------------------------
  Ca, Mg            +   direct feedstock cation release (wollastonite+diopside)
  Al                -   ERW raises pH -> Al solubility collapses (anti-acidification)
  Mn, Fe            -   pH/redox-sensitive: precipitation at higher pH usually
                        outweighs any minor feedstock release (NOT clean tracers)
  K, Na, NO3, NH4   0   inert / biologically cycled background (negative controls)

A pattern that puts Ca/Mg up, Al (and likely Mn/Fe) down, and the inert ions near
zero is a coherent ERW fingerprint even when the average cation effect is ~0. We
report each element's pooled (15 cm) Hedges' g with plot-block CIs and the value
in the R3 shallow detection cell, then score sign concordance vs the prediction.
Honest caveat: Mn/Fe are redox/pH-confounded, so they are reported but down-
weighted; Al is the cleaner secondary fingerprint.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, RESULT_DIR
from src.analysis.effects import pooled_effect_with_ci, qa_clean
from src.io.load_resin import load_resin
from src.stats.bootstrap import cohens_d_hedges_g

# element -> (label, expected sign, clean tracer?)
PANEL = {
    "ca_ppm":    ("feedstock cation",        +1, True),
    "mg_ppm":    ("feedstock cation",        +1, True),
    "al_ppm":    ("pH-sensitive (liming)",   -1, True),
    "mn_ppm":    ("redox/pH trace metal",    -1, False),
    "fe_ppm":    ("redox/pH trace metal",    -1, False),
    "k_ppm":     ("inert/biocycled",          0, True),
    "na_ppm":    ("inert",                     0, True),
    "no3_n_ppm": ("inert/biocycled",          0, True),
    "nh4_n_ppm": ("inert/biocycled",          0, True),
}
DETECT_DEPTH = 15
DETECT_ROUND = 3


def main() -> None:
    resin = qa_clean(load_resin())

    rows = []
    for ion, (label, exp_sign, clean) in PANEL.items():
        pooled = pooled_effect_with_ci(resin, ion, "60", depth_cm=DETECT_DEPTH,
                                       n_boot=4000)
        cell = resin[(resin["round"] == DETECT_ROUND)
                     & (resin["depth_cm"] == DETECT_DEPTH)]
        t = cell.loc[cell["treatment"] == "60", ion].values
        c = cell.loc[cell["treatment"] == "control", ion].values
        cell_g = cohens_d_hedges_g(t, c)["hedges_g"]
        g_pooled = pooled.get("stat", np.nan)
        rows.append({
            "element": ion.replace("_ppm", "").upper(),
            "role": label, "expected_sign": exp_sign, "clean_tracer": clean,
            "g_15cm_pooled": round(float(g_pooled), 3) if np.isfinite(g_pooled) else np.nan,
            "ci_lo": round(float(pooled["lo"]), 3) if np.isfinite(pooled["lo"]) else np.nan,
            "ci_hi": round(float(pooled["hi"]), 3) if np.isfinite(pooled["hi"]) else np.nan,
            "g_R3_15cm_cell": round(float(cell_g), 3) if np.isfinite(cell_g) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "cnew_multielement_fingerprint.csv", index=False)

    def sgn(x):
        return 0 if (not np.isfinite(x) or abs(x) < 0.2) else int(np.sign(x))

    # concordance on directional elements (expected_sign != 0); use R3 cell where
    # the signal exists, fall back to pooled
    directional = out[out["expected_sign"] != 0].copy()
    directional["obs_sign_cell"] = directional["g_R3_15cm_cell"].map(sgn)
    directional["match_cell"] = (directional["obs_sign_cell"]
                                 == directional["expected_sign"])
    clean_dir = directional[directional["clean_tracer"]]
    n_clean = len(clean_dir)
    n_match = int(clean_dir["match_cell"].sum())

    inert = out[out["expected_sign"] == 0]
    inert_near0 = int((inert["g_R3_15cm_cell"].abs() < 0.5).sum())
    inert_hot = inert[inert["g_R3_15cm_cell"].abs() >= 0.5]
    hot_str = ", ".join(f"{r['element']} g={r['g_R3_15cm_cell']:+.2f}"
                        for _, r in inert_hot.iterrows()) or "none"

    lines = [
        "# Experiment 16: Multi-element geochemical fingerprint\n",
        "Treated(60)-control Hedges' g per element at 15 cm (pooled across rounds, "
        f"plot-block CI) and in the R{DETECT_ROUND} {DETECT_DEPTH} cm detection "
        "cell, vs the expected ERW sign pattern.\n",
        out.to_markdown(index=False), "",
        "## Reading",
        f"- Clean directional elements (Ca, Mg up; Al down) matching the predicted "
        f"sign in the detection cell: {n_match}/{n_clean}.",
        f"- Inert/background elements (K, Na, NO3, NH4) near zero (|g|<0.5) in the "
        f"detection cell: {inert_near0}/{len(inert)} - the negative-control check.",
        f"- RED FLAG: inert elements that are NOT flat in the detection cell: "
        f"{hot_str}. A positive K/Na excess as large as Ca means the shallow cell "
        "also carries some NON-specific cation enrichment (plot fertility / general "
        "exchange loading), so not all of the shallow excess is feedstock. The "
        "feedstock-SPECIFIC discriminators that K/Na cannot mimic - the matching "
        "Ca:Mg ratio (Experiment 12) and the Al-suppression pH signal - are what "
        "separate ERW from generic enrichment here.",
        "- Mn/Fe are reported but down-weighted: their resin capture is governed by "
        "redox and pH (higher pH from ERW tends to precipitate them), so they are "
        "NOT conservative feedstock tracers and their sign is not diagnostic.", "",
        "Interpretation: this asks whether the WHOLE excess vector looks like an "
        "alkalinity-generating silicate amendment, not just whether one ion moved. "
        "The cleaner second fingerprint than Mn/Fe co-release is the Al response: "
        "ERW raising pH should suppress resin Al (anti-acidification), an effect "
        "independent of the Ca/Mg release pathway. A coherent Ca/Mg-up, Al-down, "
        "inert-flat pattern in the same shallow cell that carries the feedstock "
        "Ca:Mg ratio (Experiment 12) is multi-line evidence that the shallow signal is "
        "ERW, not background. Small n (one detection cell): a pattern-consistency "
        "check, not a powered per-element claim.",
    ]
    (AUDIT_DIR / "cnew_multielement_fingerprint.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
