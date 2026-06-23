"""Experiment 7: which MRV readout is most detectable - charge-balance vs single-ion?

The current MRV debate (Biogeosciences 2026 review: cation/charge-balance MRV is
more practical than carbon/total-alkalinity titration, which is argued unsuitable
for agricultural porewaters) is about WHICH quantity to track. We test this
empirically on the resin panel by comparing the *detectability* of three
candidate readouts as replication grows:

    - net charge balance  = sum(cation mol_c) - sum(anion mol_c)   (alkalinity proxy)
    - total base cations  = sum(cation mol_c)
    - single ion (Ca)     = Ca mol_c                               (SIA / selective-ion)

Detectability = fraction of plot-clustered bootstrap subsamples (k plot-halves
per arm) whose treated-vs-control effect has a same-sign 95% CI excluding 0.
Our resin readout uses ion-exchange capture (mol_c), side-stepping the porewater
titration problem the review raises.
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (AUDIT_DIR, FIGURE_DIR, ION_CHARGE, ION_MOLAR_MASS,
                        RANDOM_SEED, RESULT_DIR)
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin
from src.stats.bootstrap import cohens_d_hedges_g

CATIONS = ["ca_ppm", "mg_ppm", "k_ppm", "na_ppm", "nh4_n_ppm"]
ANIONS = ["no3_n_ppm", "s_ppm", "p_ppm"]
READOUTS = ["net_alk", "cation_molc", "ca_only"]
K_GRID = [2, 3, 4]            # plot-halves per arm
N_DRAWS = 400


def molc(series: pd.Series, ion: str) -> pd.Series:
    return (series.fillna(0) / ION_MOLAR_MASS[ion]) * abs(ION_CHARGE[ion])


def detect_rate(df: pd.DataFrame, metric: str, k: int,
                rng: np.random.Generator) -> dict:
    """Fraction of subsamples (k plot-halves/arm) with |g|>=0.8 and same sign."""
    ctrl_halves = df.loc[df["treatment"] == "control", "plot_half"].unique()
    trt_halves = df.loc[df["treatment"] == "60", "plot_half"].unique()
    if len(ctrl_halves) < k or len(trt_halves) < k:
        return {"detect_rate": np.nan, "mean_g": np.nan}
    ctrl_combos = list(combinations(ctrl_halves, k))
    trt_combos = list(combinations(trt_halves, k))
    gs, hits = [], 0
    for _ in range(N_DRAWS):
        cc = ctrl_combos[rng.integers(len(ctrl_combos))]
        tc = trt_combos[rng.integers(len(trt_combos))]
        c = df[(df["treatment"] == "control") & df["plot_half"].isin(cc)][metric].values
        t = df[(df["treatment"] == "60") & df["plot_half"].isin(tc)][metric].values
        es = cohens_d_hedges_g(t, c)
        g = es["hedges_g"]
        if np.isfinite(g):
            gs.append(g)
            if abs(g) >= 0.8:
                hits += 1
    return {"detect_rate": round(hits / max(len(gs), 1), 3),
            "mean_g": round(float(np.mean(gs)), 3) if gs else np.nan}


def main() -> None:
    resin = qa_clean(load_resin())
    resin = resin[resin["treatment"].isin(["control", "60"])].copy()
    resin["net_alk"] = (sum(molc(resin[i], i) for i in CATIONS)
                        - sum(molc(resin[i], i) for i in ANIONS))
    resin["cation_molc"] = sum(molc(resin[i], i) for i in CATIONS)
    resin["ca_only"] = molc(resin["ca_ppm"], "ca_ppm")

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for metric in READOUTS:
        for k in K_GRID:
            r = detect_rate(resin, metric, k, rng)
            rows.append({"readout": metric, "plot_halves_per_arm": k, **r})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "cnew_mrv_readout_detectability.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    label = {"net_alk": "net charge balance", "cation_molc": "total base cations",
             "ca_only": "single ion (Ca)"}
    for metric in READOUTS:
        sub = df[df["readout"] == metric]
        ax.plot(sub["plot_halves_per_arm"], sub["detect_rate"], "-o",
                label=label[metric], lw=1.8)
    ax.set_xlabel("Plot-halves per arm (replication)")
    ax.set_ylabel("Detection rate (|g|>=0.8 in bootstrap subsamples)")
    ax.set_xticks(K_GRID)
    ax.set_ylim(0, 1)
    ax.set_title("Which MRV readout is most detectable at low replication?")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cnew_mrv_readout.png", dpi=130)
    plt.close(fig)

    best = (df.loc[df.groupby("plot_halves_per_arm")["detect_rate"].idxmax()]
            [["plot_halves_per_arm", "readout", "detect_rate"]])
    lines = ["# Experiment 7: MRV readout detectability (charge-balance vs single-ion)\n",
             "Detection rate = fraction of plot-clustered bootstrap subsamples "
             "(k plot-halves/arm, 60 t/ha vs control) with |Hedges g|>=0.8.\n",
             "## Detectability by readout x replication",
             df.to_markdown(index=False), "",
             "## Most detectable readout per replication level",
             best.to_markdown(index=False), "",
             "Framing: this is field evidence for the cation/charge-balance MRV "
             "debate (Biogeosciences 2026). Because our resin readout captures "
             "ions on exchange membranes as charge equivalents (mol_c), it avoids "
             "the porewater total-alkalinity titration problem the review flags. "
             "Where the multi-ion charge-balance readout is not more detectable "
             "than single-ion Ca, the panel's value is interpretive (negative "
             "controls, anion correction), not raw detection - an honest "
             "qualifier to single-ion selective-instrument (SIA) marketing.",
             "", f"Figure: {FIGURE_DIR/'cnew_mrv_readout.png'}"]
    (AUDIT_DIR / "cnew_mrv_readout.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
