"""Experiment 14: is the CDR-lag depth fingerprint statistically real? (randomization test)

The headline mechanism (Experiment 1) is a SHAPE claim: the treated base-cation excess
is retained shallow and the deep layer sits at/below control, so the excess
DECREASES with depth. Per-cell bootstrap CIs (scripts/60) are wide at n=12; here
we test the depth pattern as a single quantity with a treatment-label permutation
test that respects the plot-level randomisation.

Two statistics on the treated(60)-minus-control base-cation (Ca+Mg, mol_c) excess:
  * retention index  = mean_round( excess[15 cm] - excess[100 cm] )   (> 0 => shallow-retained)
  * depth slope      = OLS slope of excess vs depth                   (< 0 => decreasing with depth)

Null: treatment is unrelated to the depth pattern. We permute the plot->treatment
labels (treatment is a plot-level attribute) many times and recompute, giving an
exact-style p-value. Plot-block bootstrap CIs accompany the point estimates.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, ION_CHARGE, ION_MOLAR_MASS, RANDOM_SEED, RESULT_DIR
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin

DEPTHS = [15, 40, 100]
N_PERM = 10000


def base_molc(df: pd.DataFrame) -> pd.Series:
    out = np.zeros(len(df))
    for ion in ("ca_ppm", "mg_ppm"):
        out = out + (df[ion].fillna(0) / ION_MOLAR_MASS[ion]) * ION_CHARGE[ion]
    return pd.Series(out, index=df.index)


def _stats(cells: dict, treated_mask: np.ndarray) -> tuple[float, float]:
    """retention index & depth slope from precomputed per-plot cell means.

    `cells[(round, depth)] = (plot_idx_array, value_array)`; treated_mask is a
    boolean over the global plot index. Pure numpy for fast permutation.
    """
    rounds = sorted({r for (r, _) in cells})
    per_round, pts_d, pts_e = [], [], []
    for rnd in rounds:
        ex = {}
        for depth in DEPTHS:
            key = (rnd, depth)
            if key not in cells:
                continue
            idx, val = cells[key]
            tm = treated_mask[idx]
            if tm.all() or (~tm).all():
                continue
            ex[depth] = val[tm].mean() - val[~tm].mean()
            pts_d.append(depth); pts_e.append(ex[depth])
        if 15 in ex and 100 in ex:
            per_round.append(ex[15] - ex[100])
    ri = float(np.mean(per_round)) if per_round else np.nan
    slope = (float(np.polyfit(pts_d, pts_e, 1)[0])
             if len(set(pts_d)) >= 2 else np.nan)
    return ri, slope


def _build_cells(pm: pd.DataFrame, pidx: dict) -> dict:
    cells = {}
    for (rnd, depth), g in pm.groupby(["round", "depth_cm"]):
        idx = g["plot_half"].map(pidx).to_numpy()
        cells[(rnd, depth)] = (idx, g["base_molc"].to_numpy())
    return cells


def main() -> None:
    resin = qa_clean(load_resin())
    resin = resin[resin["treatment"].isin(["control", "60"])].copy()
    resin["base_molc"] = base_molc(resin)

    # precompute per (plot_half, round, depth) mean -> small table
    pm = (resin.groupby(["plot_half", "round", "depth_cm"])["base_molc"]
          .mean().reset_index())
    plots = resin[["plot_half", "treatment"]].drop_duplicates()
    plot_ids = list(plots["plot_half"])
    pidx = {p: i for i, p in enumerate(plot_ids)}
    treated0 = np.array([plots.set_index("plot_half").loc[p, "treatment"] == "60"
                         for p in plot_ids])

    cells = _build_cells(pm, pidx)
    obs_ri, obs_slope = _stats(cells, treated0)

    rng = np.random.default_rng(RANDOM_SEED)
    perm_ri = np.empty(N_PERM)
    perm_slope = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_ri[i], perm_slope[i] = _stats(cells, rng.permutation(treated0))

    perm_ri = perm_ri[~np.isnan(perm_ri)]
    perm_slope = perm_slope[~np.isnan(perm_slope)]
    # one-sided: retention index > 0 (shallow-retained), slope < 0 (decreasing)
    p_ri = float((perm_ri >= obs_ri).mean())
    p_slope = float((perm_slope <= obs_slope).mean())

    # per-plot cell list for fast plot-block bootstrap
    plot_cells = {i: [] for i in range(len(plot_ids))}
    for _, row in pm.iterrows():
        plot_cells[pidx[row["plot_half"]]].append(
            (int(row["round"]), int(row["depth_cm"]), float(row["base_molc"])))

    boot_ri, boot_slope = [], []
    for _ in range(4000):
        draw = rng.choice(len(plot_ids), size=len(plot_ids), replace=True)
        bcells: dict = {}
        bmask = np.empty(len(draw), dtype=bool)
        for k, j in enumerate(draw):
            bmask[k] = treated0[j]
            for (rnd, depth, val) in plot_cells[j]:
                key = (rnd, depth)
                bcells.setdefault(key, ([], []))
                bcells[key][0].append(k)
                bcells[key][1].append(val)
        bcells = {key: (np.array(ix), np.array(vv)) for key, (ix, vv) in bcells.items()}
        ri, sl = _stats(bcells, bmask)
        if np.isfinite(ri):
            boot_ri.append(ri)
        if np.isfinite(sl):
            boot_slope.append(sl)
    ri_lo, ri_hi = np.quantile(boot_ri, [0.025, 0.975])
    sl_lo, sl_hi = np.quantile(boot_slope, [0.025, 0.975])

    out = pd.DataFrame([
        {"statistic": "retention_index (excess_15 - excess_100, mol_c)",
         "observed": round(obs_ri, 4), "ci_lo": round(float(ri_lo), 4),
         "ci_hi": round(float(ri_hi), 4), "perm_p_one_sided": round(p_ri, 4),
         "direction_tested": "> 0 (shallow-retained)"},
        {"statistic": "depth_slope (excess vs depth, mol_c per cm)",
         "observed": round(obs_slope, 6), "ci_lo": round(float(sl_lo), 6),
         "ci_hi": round(float(sl_hi), 6), "perm_p_one_sided": round(p_slope, 4),
         "direction_tested": "< 0 (decreasing with depth)"},
    ])
    out.to_csv(RESULT_DIR / "cnew_cdr_lag_significance.csv", index=False)

    sig_ri = p_ri < 0.05
    sig_slope = p_slope < 0.05
    lines = [
        "# Experiment 14: Randomization test of the CDR-lag depth fingerprint\n",
        f"Treatment-label permutation test ({N_PERM:,} permutations, plot-level "
        "labels) on the treated(60)-control base-cation (Ca+Mg) excess; plot-block "
        "bootstrap CIs (4,000).\n",
        out.to_markdown(index=False), "",
        "## Reading",
        f"- Retention index = {obs_ri:.3f} mol_c "
        f"[{ri_lo:.3f}, {ri_hi:.3f}], permutation p = {p_ri:.3f} "
        f"({'significant' if sig_ri else 'not significant'} at 0.05).",
        f"- Depth slope = {obs_slope:.4f} mol_c/cm "
        f"[{sl_lo:.4f}, {sl_hi:.4f}], permutation p = {p_slope:.3f} "
        f"({'significant' if sig_slope else 'not significant'} at 0.05).", "",
        "Interpretation: a positive retention index and negative depth slope are "
        "the quantitative signature of shallow retention / deep depletion. The "
        "permutation p-values say how often label-reshuffling alone reproduces a "
        "pattern this strong - turning the per-cell bootstrap picture into a single "
        "design-respecting significance statement for the CDR-lag SHAPE. With n=12 "
        "plots and 3 rounds this is the best-powered test the data allow; treat a "
        "non-significant p as 'suggestive shape, underpowered', not 'absent'.",
    ]
    (AUDIT_DIR / "cnew_cdr_lag_significance.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
