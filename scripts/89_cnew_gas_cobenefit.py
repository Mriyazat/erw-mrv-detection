"""Experiment 19: multi-gas co-benefit screen (CO2, CH4, N2O, NH3) from the chamber data.

A pathway we had not analysed: raising soil pH via silicate weathering is expected
to shift the soil greenhouse-gas budget - most notably to SUPPRESS N2O (pH-
sensitive denitrification favours full reduction to N2 at higher pH), a headline
ERW co-benefit in the current literature - and to modulate CH4 and CO2. We screen
the eosAC autochamber fluxes (spring 2026) for a treatment contrast.

Hard limit: the chamber covers only 4 plots (2 treated: 6W=60, 7E=20; 2 control:
6E, 7W), so this is a directional snapshot, not a powered test - the exact
permutation p floors at 1/6 = 0.167. We report QA-filtered per-plot mean fluxes,
the treated-vs-control contrast and Hedges' g per gas, with the expected sign, and
state plainly that anything here is hypothesis-generating.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, CACHE_DIR, RESULT_DIR
from src.stats.bootstrap import cohens_d_hedges_g

# gas -> (flux column, qa column, expected treated-vs-control sign, mechanism)
GASES = {
    "N2O": ("flux_n2o_exp", "flux_n2o_qa_ok", -1,
            "higher pH favours N2O->N2 reduction (denitrification co-benefit)"),
    "CH4": ("flux_ch4_exp", "flux_ch4_qa_ok",  0,
            "ambiguous: pH/moisture shift oxidation vs production"),
    "CO2": ("flux_co2_exp", "flux_co2_qa_ok",  0,
            "ambiguous: carbonate buffering vs respiration changes"),
    "NH3": ("flux_nh3_exp", "flux_nh3_qa_ok", +1,
            "higher pH shifts NH4+/NH3 equilibrium toward volatile NH3"),
}


def exact_perm_p(values: np.ndarray, is_t: np.ndarray, sign: int) -> float:
    """Exact one-sided p in the EXPECTED direction; two-sided fallback if sign=0."""
    g_obs = cohens_d_hedges_g(values[is_t], values[~is_t])["hedges_g"]
    if not np.isfinite(g_obs):
        return np.nan
    n, k = len(values), int(is_t.sum())
    gs = []
    for combo in combinations(range(n), k):
        m = np.zeros(n, dtype=bool)
        m[list(combo)] = True
        g = cohens_d_hedges_g(values[m], values[~m])["hedges_g"]
        if np.isfinite(g):
            gs.append(g)
    gs = np.array(gs)
    if sign < 0:
        return float((gs <= g_obs + 1e-12).mean())
    if sign > 0:
        return float((gs >= g_obs - 1e-12).mean())
    return float((np.abs(gs) >= abs(g_obs) - 1e-12).mean())


def main() -> None:
    ch = pd.read_parquet(CACHE_DIR / "chamber.parquet")
    ch = ch[ch["valid"]].copy()

    rows = []
    perplot_all = []
    for gas, (fcol, qcol, sign, mech) in GASES.items():
        sub = ch[ch[qcol]].dropna(subset=[fcol, "plot_id", "treatment"])
        pp = (sub.groupby(["plot_id", "treatment"])[fcol]
              .mean().reset_index().rename(columns={fcol: "flux"}))
        pp["gas"] = gas
        perplot_all.append(pp)
        is_t = pp["treatment"].isin(["20", "60"]).to_numpy()
        if is_t.sum() < 1 or (~is_t).sum() < 1:
            continue
        vals = pp["flux"].to_numpy()
        res = cohens_d_hedges_g(vals[is_t], vals[~is_t])
        p = (exact_perm_p(vals, is_t, sign)
             if is_t.sum() >= 1 and (~is_t).sum() >= 1 and len(vals) >= 4 else np.nan)
        rows.append({
            "gas": gas, "n_plots": len(pp),
            "n_treated": int(is_t.sum()), "n_control": int((~is_t).sum()),
            "n_qa_obs": len(sub),
            "treated_mean_flux": round(float(vals[is_t].mean()), 5),
            "control_mean_flux": round(float(vals[~is_t].mean()), 5),
            "contrast": round(res["mean_diff"], 5),
            "hedges_g": round(res["hedges_g"], 3),
            "expected_sign": sign,
            "obs_matches_expected": bool(
                sign == 0 or np.sign(res["mean_diff"]) == np.sign(sign)),
            "exact_perm_p": round(p, 3) if np.isfinite(p) else np.nan,
            "mechanism": mech,
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "cnew_gas_cobenefit.csv", index=False)
    pd.concat(perplot_all, ignore_index=True).to_csv(
        RESULT_DIR / "cnew_gas_cobenefit_perplot.csv", index=False)

    n2o = out[out["gas"] == "N2O"]
    n2o_dir = (bool(n2o["obs_matches_expected"].iloc[0]) if len(n2o) else False)

    lines = [
        "# Experiment 19: Multi-gas co-benefit screen (chamber)\n",
        "eosAC autochamber fluxes (spring 2026), QA-filtered, treated(20/60) vs "
        "control. 4 plots only (2/arm) - directional snapshot, exact permutation p "
        "floors at 1/6=0.167.\n",
        out.drop(columns=["mechanism"]).to_markdown(index=False), "",
        "## Mechanisms / expected signs",
        *[f"- **{g}** (expected {'-' if v[2]<0 else '+' if v[2]>0 else '0'}): {v[3]}"
          for g, v in GASES.items()],
        "",
        "## Reading",
        f"- N2O is the co-benefit of interest: observed contrast "
        f"{'matches' if n2o_dir else 'does NOT match'} the expected suppression "
        "direction" + (f" (g={n2o['hedges_g'].iloc[0]:+.2f}, "
                       f"p={n2o['exact_perm_p'].iloc[0]})." if len(n2o) else "."),
        "- With 2 plots/arm nothing can be significant (p>=0.167); signs and effect "
        "sizes are hypothesis-generating only.", "",
        "Interpretation: this opens the gas-phase / co-benefit dimension of the "
        "dataset for the first time. A treated N2O suppression in the expected "
        "direction would motivate a properly replicated flux campaign as the "
        "highest-value next measurement (N2O has ~273x the GWP of CO2, so a "
        "co-benefit can dominate the climate case for ERW); a null or wrong-sign "
        "snapshot says the spring chamber data cannot support a co-benefit claim. "
        "Either way it is framed as a pilot that scopes a future experiment, not a "
        "result - the chamber's 4-plot coverage is the binding constraint.",
    ]
    (AUDIT_DIR / "cnew_gas_cobenefit.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
