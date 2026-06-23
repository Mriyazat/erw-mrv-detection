"""Experiment 8: two-stage sampling design - add plots or add capsules?

The MRV reviews stress that beating soil heterogeneity needs high-density
sampling, but give little guidance on HOW to allocate effort. We decompose the
natural (control-only) resin variance into between-plot and within-plot
(capsule) components and use the classic two-stage formula to map the minimum
detectable effect (MDE) over a grid of (plots/arm x capsules/plot):

    SE(arm mean) = sqrt( sigma_b^2 / P  +  sigma_w^2 / (P * M) )
    MDE          = (z_{1-alpha/2} + z_{power}) * SE * sqrt(2)      (two arms)

This tells a field team whether the next dollar buys more plots or more capsules
per plot - an actionable design output, not just "sample more".
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (AUDIT_DIR, FIGURE_DIR, ION_CHARGE, ION_MOLAR_MASS,
                        RESULT_DIR)
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin

ALPHA, POWER = 0.05, 0.80
P_GRID = [2, 4, 6, 8, 12, 24]      # plots per arm
M_GRID = [1, 2, 3, 4, 6]           # capsules per plot
CATIONS = ["ca_ppm", "mg_ppm", "k_ppm", "na_ppm", "nh4_n_ppm"]
ANIONS = ["no3_n_ppm", "s_ppm", "p_ppm"]


def molc(s: pd.Series, ion: str) -> pd.Series:
    return (s.fillna(0) / ION_MOLAR_MASS[ion]) * abs(ION_CHARGE[ion])


def variance_components(df: pd.DataFrame, col: str) -> dict:
    """Between-plot and within-plot variance from control replicates."""
    g = df.groupby("plot_half")[col]
    plot_means = g.mean()
    within = g.var(ddof=1).dropna()
    sigma_b2 = float(plot_means.var(ddof=1))
    sigma_w2 = float(within.mean()) if len(within) else 0.0
    return {"sigma_b2": sigma_b2, "sigma_w2": sigma_w2,
            "icc": sigma_b2 / (sigma_b2 + sigma_w2)
            if (sigma_b2 + sigma_w2) > 0 else np.nan,
            "grand_sd": float(np.sqrt(sigma_b2 + sigma_w2))}


def mde_grid(vc: dict) -> pd.DataFrame:
    z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)
    rows = []
    for P in P_GRID:
        for M in M_GRID:
            se = np.sqrt(vc["sigma_b2"] / P + vc["sigma_w2"] / (P * M))
            mde = z * se * np.sqrt(2)
            rows.append({"plots_per_arm": P, "capsules_per_plot": M,
                         "mde_abs": mde,
                         "mde_in_grand_sd": mde / vc["grand_sd"]
                         if vc["grand_sd"] > 0 else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    resin = qa_clean(load_resin())
    resin["net_alk"] = (sum(molc(resin[i], i) for i in CATIONS)
                        - sum(molc(resin[i], i) for i in ANIONS))
    ctrl = resin[resin["treatment"] == "control"].copy()

    targets = {"ca_ppm": "Ca (ppm)", "net_alk": "net charge balance (mol_c)"}
    vc_rows, grids = [], {}
    for col, lab in targets.items():
        vc = variance_components(ctrl.dropna(subset=[col]), col)
        vc_rows.append({"target": lab, **{k: round(v, 4) for k, v in vc.items()}})
        grids[col] = mde_grid(vc)
        grids[col].to_csv(RESULT_DIR / f"cnew_sampling_mde_{col}.csv", index=False)
    vc_df = pd.DataFrame(vc_rows)
    vc_df.to_csv(RESULT_DIR / "cnew_sampling_variance_components.csv", index=False)

    # Heatmap of MDE (in grand-SD units) for Ca.
    g = grids["ca_ppm"].pivot(index="plots_per_arm", columns="capsules_per_plot",
                              values="mde_in_grand_sd")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    im = ax.imshow(g.values, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(M_GRID))); ax.set_xticklabels(M_GRID)
    ax.set_yticks(range(len(P_GRID))); ax.set_yticklabels(P_GRID)
    ax.set_xlabel("Capsules per plot")
    ax.set_ylabel("Plots per arm")
    ax.set_title("MDE (in grand-SD units) for Ca - two-stage design")
    for i in range(len(P_GRID)):
        for j in range(len(M_GRID)):
            ax.text(j, i, f"{g.values[i, j]:.2f}", ha="center", va="center",
                    color="w", fontsize=8)
    fig.colorbar(im, label="MDE / grand SD")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cnew_sampling_design.png", dpi=130)
    plt.close(fig)

    icc_ca = vc_df.loc[vc_df["target"].str.startswith("Ca"), "icc"].iloc[0]
    # marginal value: MDE drop from doubling plots vs doubling capsules at P=4,M=2
    base = grids["ca_ppm"]
    def mde_at(P, M):
        return float(base[(base.plots_per_arm == P)
                          & (base.capsules_per_plot == M)]["mde_in_grand_sd"].iloc[0])
    d_plots = mde_at(4, 2) - mde_at(8, 2)
    d_caps = mde_at(4, 2) - mde_at(4, 4)

    lines = ["# Experiment 8: Two-stage sampling design (plots vs capsules)\n",
             "Variance components from CONTROL replicates; MDE via the two-stage "
             "formula (alpha=0.05, power=0.80).\n",
             "## Variance components", vc_df.to_markdown(index=False), "",
             f"Intraclass correlation (Ca) ICC = {icc_ca:.2f}: only ~{icc_ca*100:.0f}% "
             "of natural variance is BETWEEN plots; ~"
             f"{(1-icc_ca)*100:.0f}% is WITHIN-plot (capsule-scale). This is the "
             "opposite of the usual assumption and means capsule replication is "
             "unusually valuable - fine-scale soil heterogeneity, not plot-to-plot "
             "differences, dominates the noise.", "",
             "## MDE grid for Ca (in grand-SD units)",
             grids["ca_ppm"].round(3).to_markdown(index=False), "",
             f"Per added unit, more plots still wins (it shrinks BOTH variance "
             f"terms): from P=4,M=2 doubling plots cuts MDE by {d_plots:.2f} SD vs "
             f"{d_caps:.2f} SD for doubling capsules. But because ICC is low, extra "
             "capsules recover most of that gain, so when capsules are much cheaper "
             "than establishing plots the cost-optimal design adds capsules first. "
             "Either way, hitting a 1-SD MDE needs ~8-12 plots/arm with >=2 "
             "capsules - converting 'sample more' into a concrete allocation rule.",
             "", f"Figure: {FIGURE_DIR/'cnew_sampling_design.png'}"]
    (AUDIT_DIR / "cnew_sampling_design.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
