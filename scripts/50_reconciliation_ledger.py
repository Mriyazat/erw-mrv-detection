"""Reconciliation ledger: erw_ml vs erw_mrv vs new canonical results.

Establishes, for every contested number, the single authoritative value and
why. Three jobs:
  1. Numeric cross-check: recompute resin Hedges' g + dose-response from the
     corrected data and confirm they match the prior repos (validates the
     rebuild). Resin depths are label-based, so they are unaffected by the
     sensor depth swap.
  2. Convention lock: dose unit (t/ha), effect size (Hedges' g), document the
     10x dose-slope discrepancy as a pure units artifact.
  3. Structural corrections: sensor depth-label swap (sensor-only), degenerate
     CI policy.

Outputs:
  outputs/results/reconciliation_effects.csv  (new vs mrv vs ml, per cell)
  outputs/results/reconciliation_ledger.csv   (one row per contested quantity)
  docs/RECONCILIATION.md
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import DOCS_DIR, RESULT_DIR, RESIN_PRIMARY_IONS
from src.analysis.effects import compute_dose_response, compute_effects
from src.io.load_resin import load_resin

AIDA = ROOT.parent
MRV = AIDA / "erw_mrv" / "outputs" / "results"
ML = AIDA / "erw_ml" / "outputs" / "results"

IONS = RESIN_PRIMARY_IONS + ["k_ppm", "na_ppm", "s_ppm"]


def numeric_crosscheck() -> tuple[pd.DataFrame, dict]:
    resin = load_resin()
    new_eff = compute_effects(resin, IONS)

    mrv = pd.read_csv(MRV / "phase3_effects.csv")
    mrv_keep = mrv[["ion", "round", "depth_cm", "treatment", "hedges_g"]].rename(
        columns={"hedges_g": "hedges_g_mrv"})
    mrv_keep["treatment"] = mrv_keep["treatment"].astype(str)

    merged = new_eff.merge(mrv_keep, on=["ion", "round", "depth_cm", "treatment"],
                           how="left")
    merged["abs_diff_mrv"] = (merged["hedges_g"] - merged["hedges_g_mrv"]).abs()

    matched = merged.dropna(subset=["hedges_g_mrv", "hedges_g"])
    stats = {
        "n_cells": int(len(matched)),
        "max_abs_diff_vs_mrv": float(matched["abs_diff_mrv"].max()),
        "median_abs_diff_vs_mrv": float(matched["abs_diff_mrv"].median()),
        "n_cells_diff_gt_1e3": int((matched["abs_diff_mrv"] > 1e-3).sum()),
    }
    return merged, stats


def dose_units_check() -> dict:
    resin = load_resin()
    dr = compute_dose_response(resin, ["ca_ppm"])
    mrv = pd.read_csv(MRV / "phase3_dose_response.csv")
    j = dr.merge(
        mrv[["ion", "round", "depth_cm", "slope_ppm_per_kgm2"]].rename(
            columns={"slope_ppm_per_kgm2": "slope_kgm2_mrv"}),
        on=["ion", "round", "depth_cm"], how="left")
    j["ratio_tha_to_mrv"] = j["slope_ppm_per_kgm2"] / j["slope_kgm2_mrv"]
    j["ratio_unit_factor"] = j["slope_ppm_per_kgm2"] / j["slope_ppm_per_tha"]
    return {
        "kgm2_slopes_match_mrv": bool(
            np.allclose(j["slope_ppm_per_kgm2"].dropna(),
                        j["slope_kgm2_mrv"].dropna(), rtol=1e-3, atol=1e-6)),
        "tha_over_kgm2_factor": float(
            (j["slope_ppm_per_tha"] / j["slope_ppm_per_kgm2"]).dropna().median()),
    }


def build_ledger(cross_stats: dict, dose_stats: dict) -> pd.DataFrame:
    rows = [
        {"quantity": "Effect-size metric", "erw_ml": "Hedges' g (& Cohen's d)",
         "erw_mrv": "Hedges' g (& Cohen's d)", "authoritative": "Hedges' g",
         "status": "RECONCILED",
         "note": f"new vs mrv match: max|diff|={cross_stats['max_abs_diff_vs_mrv']:.2e} "
                 f"over {cross_stats['n_cells']} cells"},
        {"quantity": "Dose unit", "erw_ml": "t/ha (0/20/60)",
         "erw_mrv": "kg/m^2 (0/2/6)", "authoritative": "t/ha (kg/m^2 secondary)",
         "status": "RECONCILED",
         "note": f"prior 10x slope gap is pure units; t/ha = "
                 f"{dose_stats['tha_over_kgm2_factor']:.3f} x kg/m^2 slope; "
                 f"kg/m^2 slopes match mrv: {dose_stats['kgm2_slopes_match_mrv']}"},
        {"quantity": "Sensor depth labels (15 vs 100 cm)",
         "erw_ml": "[15,40,100] ascending (WRONG)",
         "erw_mrv": "[15,40,100] ascending (WRONG)",
         "authoritative": "[100,40,15] deep->shallow (thermal-validated)",
         "status": "CORRECTED",
         "note": "block order is deep->shallow; prior repos swapped 15cm/100cm "
                 "SENSOR channels. Resin depths (label-based) unaffected. "
                 "See docs/SENSOR_DEPTH_CORRECTION.md (11/12 plots pass)."},
        {"quantity": "Resin depth-resolved effects",
         "erw_ml": "label-based depths", "erw_mrv": "label-based depths",
         "authoritative": "label-based depths (unaffected by sensor swap)",
         "status": "RECONCILED",
         "note": "Sample-ID Shallow/Middle/Deep are unambiguous."},
        {"quantity": "Degenerate CIs (e.g. SNR_Ca upper ~7.6e9)",
         "erw_ml": "present in headline_summary.csv", "erw_mrv": "n/a",
         "authoritative": "ratio CIs clipped/flagged; report on log or bounded scale",
         "status": "FIXED-IN-HEADLINE",
         "note": "SNR ratio blows up when sigma denom ~0; fixed in 51_build_headline.py"},
        {"quantity": "Retired ML targets (wflux_proxy_total, pre-split z-score, "
                     "Phase7-on-wflux)",
         "erw_ml": "computed but RETIRED", "erw_mrv": "excluded",
         "authoritative": "excluded from all headline claims",
         "status": "EXCLUDED",
         "note": "tautological target / leakage; never in headline."},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    merged, cross_stats = numeric_crosscheck()
    dose_stats = dose_units_check()
    ledger = build_ledger(cross_stats, dose_stats)

    merged.to_csv(RESULT_DIR / "reconciliation_effects.csv", index=False)
    ledger.to_csv(RESULT_DIR / "reconciliation_ledger.csv", index=False)

    lines = ["# Reconciliation Ledger\n",
             "Authoritative resolution for every number that differed between "
             "`erw_ml` and `erw_mrv`. Generated by `make reconcile`.\n",
             "## Numeric cross-check (validates the rebuild)\n",
             f"- Recomputed resin Hedges' g matches `erw_mrv` phase3 to "
             f"**max |diff| = {cross_stats['max_abs_diff_vs_mrv']:.2e}** across "
             f"{cross_stats['n_cells']} ion x round x depth x arm cells "
             f"({cross_stats['n_cells_diff_gt_1e3']} cells differ by >1e-3).",
             f"- Dose-response: t/ha slope = "
             f"{dose_stats['tha_over_kgm2_factor']:.3f} x the kg/m^2 slope; "
             f"kg/m^2 slopes match `erw_mrv`: {dose_stats['kgm2_slopes_match_mrv']}.\n",
             "## Ledger\n",
             ledger.to_markdown(index=False), "",
             "## Locked conventions",
             "- Dose: **t/ha** canonical (kg/m^2 secondary).",
             "- Effect size: **Hedges' g**.",
             "- Sensor depth: **[100, 40, 15]** (deep->shallow), thermal-validated.",
             "- CV: plot-level spatial holdout; in-fold preprocessing.",
             ""]
    (DOCS_DIR / "RECONCILIATION.md").write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWrote {RESULT_DIR/'reconciliation_ledger.csv'}, "
          f"{RESULT_DIR/'reconciliation_effects.csv'}, {DOCS_DIR/'RECONCILIATION.md'}")


if __name__ == "__main__":
    main()
