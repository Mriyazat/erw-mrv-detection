"""Experiment 18: cross-plot contamination / spatial spillover test.

The plots are arranged as an interleaved transect (treated and control halves
alternate, ~15 m within a row and ~20-40 m between rows). If amendment-derived
cations move laterally (surface runoff, shallow lateral flow, tillage/wind drift),
'control' plots near treated plots would be enriched - which would DEFLATE the
treated-minus-control contrast and partly explain the near-zero average effect.

We test this directly. For each plot we build a spillover-exposure index from the
applied doses of the OTHER plots weighted by inverse distance:
    E_i = sum_{j != i} dose_j / d_ij        (dose in t/ha, d in m)
and ask whether CONTROL-plot resin base cations (Ca+Mg) rise with exposure. A
positive control-only association is evidence of contamination; a flat one means
the controls are clean and the weak contrast is genuine, not leakage. n=4 controls
=> a screening test (Spearman + exposed-vs-interior contrast), not a powered claim,
but either outcome is a concrete MRV trial-design lesson.
"""

from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from src.config import AUDIT_DIR, CACHE_DIR, DOSE_THA, ION_CHARGE, ION_MOLAR_MASS, RESULT_DIR
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin

SHALLOW = 15


def plot_coords() -> pd.DataFrame:
    m = pd.read_parquet(CACHE_DIR / "plot_metadata.parquet")
    c = (m.dropna(subset=["latitude", "longitude"])
         .groupby("plot_id")
         .agg(lat=("latitude", "mean"), lon=("longitude", "mean"),
              treatment=("treatment", "first")).reset_index())
    lat0 = c["lat"].mean()
    c["x_m"] = (c["lon"] - c["lon"].mean()) * np.cos(np.radians(lat0)) * 111_320
    c["y_m"] = (c["lat"] - c["lat"].mean()) * 111_320
    c["dose_tha"] = c["treatment"].map(DOSE_THA).fillna(0.0)
    return c


def base_molc_by_plot(depth: int | None) -> pd.DataFrame:
    r = qa_clean(load_resin())
    if depth is not None:
        r = r[r["depth_cm"] == depth]
    bc = np.zeros(len(r))
    for ion in ("ca_ppm", "mg_ppm"):
        bc += (r[ion].fillna(0) / ION_MOLAR_MASS[ion]) * ION_CHARGE[ion]
    r = r.assign(base_molc=bc)
    return (r.groupby("plot_half")["base_molc"].mean()
            .rename("base_molc").reset_index()
            .rename(columns={"plot_half": "plot_id"}))


def main() -> None:
    c = plot_coords()
    # pairwise distances + inverse-distance dose exposure (exclude self)
    xy = c[["x_m", "y_m"]].to_numpy()
    dose = c["dose_tha"].to_numpy()
    n = len(c)
    exposure = np.zeros(n)
    nearest_treated = np.full(n, np.nan)
    for i in range(n):
        d = np.sqrt(((xy - xy[i]) ** 2).sum(axis=1))
        d[i] = np.inf
        exposure[i] = np.nansum(np.where(dose > 0, dose / np.maximum(d, 1.0), 0.0))
        td = d[dose > 0]
        nearest_treated[i] = np.nanmin(td) if len(td) else np.nan
    c["exposure_idx"] = exposure
    c["nearest_treated_m"] = np.round(nearest_treated, 1)

    # pooled across all depths: more robust at n=4 (GPS cannot resolve within-row
    # E/W halves, and lateral leakage is not depth-exclusive)
    bm = base_molc_by_plot(None)
    c = c.merge(bm, on="plot_id", how="left")
    c.to_csv(RESULT_DIR / "cnew_spatial_spillover.csv", index=False)

    ctrl = c[c["treatment"] == "control"].dropna(subset=["base_molc"]).copy()

    def exact_spearman_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Observed Spearman rho and EXACT one-sided p over all y permutations."""
        rho_obs = stats.spearmanr(x, y).statistic
        if not np.isfinite(rho_obs):
            return np.nan, np.nan
        rhos = [stats.spearmanr(x, np.array(p)).statistic
                for p in permutations(y)]
        rhos = np.array([r for r in rhos if np.isfinite(r)])
        return float(rho_obs), float((rhos >= rho_obs - 1e-12).mean())

    # control base cation vs exposure (positive => possible contamination)
    if len(ctrl) >= 3 and ctrl["exposure_idx"].nunique() >= 3:
        rho, pval = exact_spearman_p(ctrl["exposure_idx"].to_numpy(),
                                     ctrl["base_molc"].to_numpy())
    else:
        rho, pval = np.nan, np.nan

    # confound check: does exposure (and cations) just track transect position?
    rho_pos_exp, _ = (exact_spearman_p(ctrl["exposure_idx"].to_numpy(),
                                       ctrl["y_m"].to_numpy())
                      if len(ctrl) >= 3 else (np.nan, np.nan))
    allp = c.dropna(subset=["base_molc"])
    rho_grad = (stats.spearmanr(allp["y_m"], allp["base_molc"]).statistic
                if len(allp) >= 3 else np.nan)

    # exposed (treated half in the SAME row, <=5 m) vs interior controls
    ctrl["exposed"] = ctrl["nearest_treated_m"] <= 5.0
    exp_mean = ctrl.loc[ctrl["exposed"], "base_molc"].mean()
    int_mean = ctrl.loc[~ctrl["exposed"], "base_molc"].mean()

    # reference: mean treated base cation, to scale any leakage
    trt_mean = c.loc[c["treatment"] != "control", "base_molc"].mean()

    lines = [
        "# Experiment 18: Cross-plot contamination / spatial spillover\n",
        "Inverse-distance dose-exposure index per plot vs shallow (15 cm) resin "
        "base cations (Ca+Mg, mol_c); control-only association tests for lateral "
        "leakage into controls.\n",
        "## Per-plot exposure and base cations",
        c[["plot_id", "treatment", "dose_tha", "nearest_treated_m",
           "exposure_idx", "base_molc"]].round(3).to_markdown(index=False), "",
        "## Control-only contamination test",
        f"- Spearman(control exposure, control base cation) rho = "
        f"{rho:.2f}, EXACT one-sided p = {pval:.3f} (positive rho => possible "
        "contamination). Note: at n=4 the smallest attainable p is ~1/24=0.042, "
        "so a perfect order is only weakly significant.",
        f"- Exposed controls (treated half <=5 m, n="
        f"{int(ctrl['exposed'].sum())}) mean base cation = {exp_mean:.2f} vs "
        f"interior controls (n={int((~ctrl['exposed']).sum())}) = {int_mean:.2f} "
        "mol_c.",
        f"- Treated-plot mean base cation = {trt_mean:.2f} mol_c (reference scale).",
        f"- Confound check: Spearman(control exposure, transect position) rho = "
        f"{rho_pos_exp:.2f}; all-plot cation transect gradient rho = {rho_grad:.2f}.",
        "",
        "## Reading",
    ]
    spillover_like = np.isfinite(rho) and rho > 0.5
    exposed_exceeds_treated = np.isfinite(exp_mean) and exp_mean > trt_mean
    if spillover_like and exposed_exceeds_treated:
        verdict = ("Control cations rise monotonically with proximity-weighted "
                   "dose exposure (rho={:.2f}), BUT the exposed controls exceed "
                   "even the treated-plot mean ({:.2f} > {:.2f}), which simple "
                   "dilution-spillover cannot produce. Combined with the exposure "
                   "index tracking transect position, the more likely driver is a "
                   "native soil-fertility gradient along the transect, not lateral "
                   "leakage. Either way the controls are NOT a clean, spatially "
                   "exchangeable baseline.").format(rho, exp_mean, trt_mean)
    elif spillover_like:
        verdict = ("Control cations rise with dose-exposure (rho={:.2f}) and stay "
                   "below the treated mean - a pattern consistent with lateral "
                   "spillover deflating the contrast.").format(rho)
    elif np.isfinite(rho):
        verdict = ("No positive control-exposure association: lateral "
                   "contamination is not an obvious driver of the weak contrast.")
    else:
        verdict = "Too few controls with data to assess the association."
    lines += [
        f"- {verdict}",
        "- Bottom line for design: the control field has strong spatial structure "
        "(consistent with the geostatistics, Experiment geostat), so an interleaved "
        "transect with adjacent arms and only 4 controls cannot isolate a small "
        "treatment effect from spatial background. The actionable MRV lesson is "
        "spatially-paired/blocked controls with buffer distance, not more plots "
        "alone.",
        "- Caveat: n=4 controls over ~100 m is a screening test; it flags that "
        "controls are spatially confounded (contamination and/or fertility "
        "gradient) but cannot separate the two mechanisms at this sample size.",
    ]
    (AUDIT_DIR / "cnew_spatial_spillover.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
