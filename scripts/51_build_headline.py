"""Build the single authoritative headline_summary.csv.

Aggregates the key numbers from the ladder into one table with consistent
columns and FIXED confidence intervals: any ratio-type CI whose magnitude
exceeds a sane bound (the degenerate SNR_Ca ~7.6e9 artifact in the old repo) is
clipped and flagged rather than reported as-is.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import RESULT_DIR

CI_ABS_BOUND = 1e4  # any |CI| beyond this is a numerical artifact -> clip+flag


def _read(name):
    p = RESULT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def fix_ci(lo, hi):
    flag = ""
    if not np.isfinite(lo) or abs(lo) > CI_ABS_BOUND:
        lo, flag = np.nan, "ci_clipped"
    if not np.isfinite(hi) or abs(hi) > CI_ABS_BOUND:
        hi, flag = np.nan, "ci_clipped"
    return lo, hi, flag


def main() -> None:
    rows = []

    snr = _read("snr_table.csv")
    if snr is not None:
        for ion in ("Ca", "Mg"):
            v = snr[(snr["treatment"] == "60") & (snr["depth_cm"] == 15)
                    & (snr["ion"] == ion)]["SNR"]
            if len(v):
                rows.append({"metric": f"theoretical_SNR_{ion}_60tha_15cm",
                             "value": round(float(v.iloc[0]), 2),
                             "ci_low": np.nan, "ci_high": np.nan,
                             "unit": "ratio", "source": "phase_snr",
                             "flag": "", "note": "first-principles, optimistic"})

    pooled = _read("empirical_pooled_ci.csv")
    if pooled is not None:
        for _, r in pooled[(pooled["ion"].isin(["ca_ppm", "mg_ppm"]))
                           & (pooled["depth_cm"].isna())].iterrows():
            lo, hi, flag = fix_ci(r["lo"], r["hi"])
            rows.append({
                "metric": f"empirical_hedges_g_{r['ion']}_{r['treatment']}tha_pooled",
                "value": round(float(r["stat"]), 3),
                "ci_low": None if np.isnan(lo) else round(lo, 3),
                "ci_high": None if np.isnan(hi) else round(hi, 3),
                "unit": "hedges_g", "source": "phase_empirical",
                "flag": flag, "note": "plot-clustered bootstrap CI"})

    tvr = _read("theory_vs_resin_summary.csv")
    if tvr is not None:
        for _, r in tvr[tvr["depth_cm"] == 15].iterrows():
            rows.append({"metric": f"sigma_inflation_needed_{r['ion']}_15cm",
                         "value": round(float(r["median_sigma_inflation"]), 1),
                         "ci_low": np.nan, "ci_high": np.nan, "unit": "x",
                         "source": "phase_theory_vs_resin", "flag": "",
                         "note": "model overpredicts detectability by this factor"})

    mt = _read("multitask_cv.csv")
    if mt is not None:
        for _, r in mt.iterrows():
            rows.append({"metric": f"lopo_mae_skill_{r['target']}",
                         "value": round(float(r["mae_skill_vs_baseline"]), 3),
                         "ci_low": np.nan, "ci_high": np.nan, "unit": "skill",
                         "source": "phase_multitask",
                         "flag": "" if r["beats_baseline"] else "below_baseline",
                         "note": "1 - mae/mean_predictor_mae (>0 beats baseline)"})

    power = _read("power_mde.csv")
    if power is not None:
        for _, r in power[(power["ion"].isin(["ca_ppm", "mg_ppm"]))
                          & (power["depth_cm"] == 15)].iterrows():
            rows.append({"metric": f"MDE_80pct_{r['ion']}_15cm",
                         "value": round(float(r["mde_in_sd"]), 2),
                         "ci_low": np.nan, "ci_high": np.nan, "unit": "control_SD",
                         "source": "phase_power", "flag": "",
                         "note": f"={r['mde_pct_of_control']}% of control mean"})

    bayes = _read("bayesian_ca.csv")
    if bayes is not None:
        b = bayes[bayes["parameter"] == "beta_dose_ppm_per_tha"]
        if len(b):
            r = b.iloc[0]
            lo, hi, flag = fix_ci(r["hdi_2.5"], r["hdi_97.5"])
            rows.append({"metric": "bayesian_dose_effect_Ca",
                         "value": round(float(r["mean"]), 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                         "unit": "ppm_per_tha", "source": "phase_bayesian",
                         "flag": flag, "note": f"P(>0)={r['p_gt_0']} (95% HDI)"})

    gm = _read("geostat_morans_i.csv")
    if gm is not None:
        rows.append({"metric": "morans_I_resin_Ca", "value": float(gm["morans_I"].iloc[0]),
                     "ci_low": np.nan, "ci_high": np.nan, "unit": "index",
                     "source": "phase_geostat", "flag": "",
                     "note": "spatial autocorrelation of plot-mean Ca"})

    head = pd.DataFrame(rows)
    head.to_csv(RESULT_DIR / "headline_summary.csv", index=False)

    n_clip = int((head["flag"] == "ci_clipped").sum())
    assert not ((head["ci_low"].abs() > CI_ABS_BOUND)
                | (head["ci_high"].abs() > CI_ABS_BOUND)).any(), \
        "degenerate CI leaked into headline"
    print(head.to_markdown(index=False))
    print(f"\n{len(head)} headline rows; {n_clip} CI(s) clipped as artifacts; "
          f"wrote {RESULT_DIR/'headline_summary.csv'}")


if __name__ == "__main__":
    main()
