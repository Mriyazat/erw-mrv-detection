"""Experiment 12: feedstock dissolution-stoichiometry fingerprint (is the signal really ERW?).

A near-zero pooled cation effect is only half the story - the deeper MRV question
is whether the cation excess that DOES appear carries the chemical signature of
the applied feedstock (50:50 wollastonite CaSiO3 + diopside CaMgSi2O6) rather
than background soil cation cycling. Multi-element ratios are a recognised ERW
fingerprinting approach; here we test the Ca:Mg stoichiometry of the treated-
minus-control resin excess against two feedstock end-members derived from
config.SNR_MODEL:

  * STOCK ratio  - molar Ca:Mg if both minerals fully dissolve (composition).
  * RATE ratio   - molar Ca:Mg of the instantaneous RELEASE flux, weighting each
                   mineral by its dissolution rate (wollastonite ~10x faster), so
                   early-season release should be strongly Ca-dominated and drift
                   toward the stock ratio over time.

A treated excess that is Ca-enriched relative to the control background Ca:Mg, in
the window between the rate and stock ratios, is a positive feedstock fingerprint.
A treated excess matching background is not. n is small (signal concentrates in
the R3 shallow cell), so this is a fingerprint test on the cells where a signal
exists, not a powered whole-trial claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, ION_MOLAR_MASS, RESULT_DIR, SNR_MODEL
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin
from src.stats.bootstrap import plot_block_bootstrap

DEPTHS = [15, 40, 100]
ROUND_LABEL = {1: "R1 (Jul)", 2: "R2 (Aug)", 3: "R3 (Sep-Oct)"}


def feedstock_ratios() -> dict:
    """Theoretical Ca:Mg molar ratios from the 50:50 wollastonite+diopside mix."""
    mw_w = SNR_MODEL["mw_wollastonite_g_mol"]
    mw_d = SNR_MODEL["mw_diopside_g_mol"]
    fw = SNR_MODEL["f_rate_wollastonite"]
    fd = SNR_MODEL["f_rate_diopside"]
    # per 100 g of 50:50 mix
    mol_w = 50.0 / mw_w          # wollastonite: 1 Ca each
    mol_d = 50.0 / mw_d          # diopside: 1 Ca + 1 Mg each
    ca_stock = mol_w + mol_d
    mg_stock = mol_d
    ca_rate = mol_w * fw + mol_d * fd
    mg_rate = mol_d * fd
    return {"ca_mg_stock": ca_stock / mg_stock,
            "ca_mg_rate": ca_rate / mg_rate}


def molar(series: pd.Series, ion: str) -> pd.Series:
    return series.fillna(0) / ION_MOLAR_MASS[ion]


def excess_ratio(df: pd.DataFrame) -> float:
    """Ca:Mg molar ratio of the treated(60)-minus-control mean excess."""
    t = df[df["treatment"] == "60"]
    c = df[df["treatment"] == "control"]
    dca = molar(t["ca_ppm"], "ca_ppm").mean() - molar(c["ca_ppm"], "ca_ppm").mean()
    dmg = molar(t["mg_ppm"], "mg_ppm").mean() - molar(c["mg_ppm"], "mg_ppm").mean()
    if dmg <= 0 or dca <= 0:
        return np.nan
    return dca / dmg


def main() -> None:
    fr = feedstock_ratios()
    resin = qa_clean(load_resin())
    resin = resin[resin["treatment"].isin(["control", "60"])].copy()

    # control background Ca:Mg (molar), per depth
    bg_rows = []
    for depth in DEPTHS:
        c = resin[(resin["treatment"] == "control") & (resin["depth_cm"] == depth)]
        bg = (molar(c["ca_ppm"], "ca_ppm").mean()
              / max(molar(c["mg_ppm"], "mg_ppm").mean(), 1e-9))
        bg_rows.append({"depth_cm": depth, "control_ca_mg_molar": round(bg, 2)})
    bg = pd.DataFrame(bg_rows)

    rows = []
    for rnd in (1, 2, 3):
        for depth in DEPTHS:
            sub = resin[(resin["round"] == rnd) & (resin["depth_cm"] == depth)]
            t = sub[sub["treatment"] == "60"]
            c = sub[sub["treatment"] == "control"]
            if len(t) < 2 or len(c) < 2:
                continue
            dca = (molar(t["ca_ppm"], "ca_ppm").mean()
                   - molar(c["ca_ppm"], "ca_ppm").mean())
            dmg = (molar(t["mg_ppm"], "mg_ppm").mean()
                   - molar(c["mg_ppm"], "mg_ppm").mean())
            boot = plot_block_bootstrap(sub, excess_ratio, block_col="plot_half",
                                        n_resamples=3000)
            ctrl_bg = float(bg.loc[bg["depth_cm"] == depth,
                                   "control_ca_mg_molar"].iloc[0])
            ratio = dca / dmg if (dca > 0 and dmg > 0) else np.nan
            rows.append({
                "round": rnd, "round_label": ROUND_LABEL[rnd], "depth_cm": depth,
                "excess_ca_mmol": round(dca * 1000, 3),
                "excess_mg_mmol": round(dmg * 1000, 3),
                "excess_ca_mg_molar": round(ratio, 2) if np.isfinite(ratio) else np.nan,
                "ratio_lo": round(boot["lo"], 2) if np.isfinite(boot["lo"]) else np.nan,
                "ratio_hi": round(boot["hi"], 2) if np.isfinite(boot["hi"]) else np.nan,
                "control_bg_ca_mg": ctrl_bg,
                "ca_enriched_vs_bg": bool(np.isfinite(ratio) and ratio > ctrl_bg),
                "both_excess_positive": bool(dca > 0 and dmg > 0),
            })
    prof = pd.DataFrame(rows)
    prof.to_csv(RESULT_DIR / "cnew_feedstock_fingerprint.csv", index=False)

    # cells with a genuine positive Ca+Mg excess (where the ratio is meaningful)
    pos = prof[prof["both_excess_positive"]]
    n_pos = len(pos)
    n_enriched = int(pos["ca_enriched_vs_bg"].sum()) if n_pos else 0

    lines = [
        "# Experiment 12: Feedstock dissolution-stoichiometry fingerprint\n",
        "Ca:Mg molar ratio of the treated(60)-minus-control resin excess vs the "
        "wollastonite+diopside feedstock end-members and the control background.\n",
        "## Feedstock end-members (from config.SNR_MODEL)",
        f"- STOCK Ca:Mg (full dissolution, composition): **{fr['ca_mg_stock']:.2f}**",
        f"- RATE Ca:Mg (dissolution-rate-weighted release flux): "
        f"**{fr['ca_mg_rate']:.1f}** (wollastonite-dominated, Ca-rich early)", "",
        "## Control background Ca:Mg by depth",
        bg.to_markdown(index=False), "",
        "## Treated-control excess stoichiometry by round x depth",
        prof.to_markdown(index=False), "",
        "## Reading",
        f"- Cells with a genuine positive Ca AND Mg excess: {n_pos}; of these, "
        f"{n_enriched} are Ca-enriched relative to the local control background "
        "(the direction a wollastonite-dominated feedstock predicts).",
        "- A positive fingerprint = excess Ca:Mg above background and within the "
        f"[stock {fr['ca_mg_stock']:.1f}, rate {fr['ca_mg_rate']:.0f}] feedstock "
        "window; values at/below background indicate the excess is background "
        "cation cycling, not feedstock dissolution.", "",
        "Interpretation: this turns a near-zero average effect into a falsifiable "
        "chemistry test - WHERE a cation excess appears (notably the R3 shallow "
        "CDR-lag cell), does it carry the feedstock's Ca-dominated signature? A "
        "yes strengthens attribution of the shallow signal to ERW; a no warns that "
        "even the visible excess may be background. Small n (signal concentrates in "
        "few cells): a fingerprint check on the cells where a signal exists, not a "
        "powered whole-trial claim. Si is absent (2M-HCl resin elution does not "
        "liberate silicate Si), so this is a cation-only fingerprint.",
    ]
    (AUDIT_DIR / "cnew_feedstock_fingerprint.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
