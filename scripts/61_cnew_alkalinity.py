"""Experiment 2: charge-balance / alkalinity budget from the multi-ion resin panel.

Single-ion ERW MRV (e.g. selective Ca analysis, the SIA / Everest "Pulsar"
paradigm) tracks one cation. Here we use the FULL UNIBEST panel to build a
charge-balance proxy for net base alkalinity:

    net_alkalinity_proxy = sum(cation mol_c) - sum(strong-anion mol_c)

and compare its treatment-detection effect size against the Ca-only signal.
The claim: a multi-ion charge-balance readout is a more robust, less
ion-specific detector of the aqueous ERW signal than any single cation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, ION_CHARGE, ION_MOLAR_MASS, RESULT_DIR
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin
from src.stats.bootstrap import cohens_d_hedges_g, plot_block_bootstrap

CATIONS = ["ca_ppm", "mg_ppm", "k_ppm", "na_ppm", "nh4_n_ppm"]
ANIONS = ["no3_n_ppm", "s_ppm", "p_ppm"]


def molc(series: pd.Series, ion: str) -> pd.Series:
    return (series.fillna(0) / ION_MOLAR_MASS[ion]) * abs(ION_CHARGE[ion])


def main() -> None:
    resin = qa_clean(load_resin())
    resin = resin[resin["treatment"].isin(["control", "20", "60"])].copy()

    resin["cation_molc"] = sum(molc(resin[i], i) for i in CATIONS)
    resin["anion_molc"] = sum(molc(resin[i], i) for i in ANIONS)
    resin["net_alkalinity"] = resin["cation_molc"] - resin["anion_molc"]
    resin["ca_only_molc"] = molc(resin["ca_ppm"], "ca_ppm")

    # Effect sizes: net-alkalinity detector vs Ca-only detector.
    rows = []
    for metric in ("net_alkalinity", "ca_only_molc", "cation_molc"):
        for arm in ("20", "60"):
            for depth in (None, 15, 40, 100):
                sub = resin if depth is None else resin[resin["depth_cm"] == depth]
                t = sub.loc[sub["treatment"] == arm, metric].values
                c = sub.loc[sub["treatment"] == "control", metric].values
                es = cohens_d_hedges_g(t, c)

                def stat(d, _m=metric, _a=arm):
                    tt = d.loc[d["treatment"] == _a, _m].values
                    cc = d.loc[d["treatment"] == "control", _m].values
                    return cohens_d_hedges_g(tt, cc)["hedges_g"]

                pool = sub[sub["treatment"].isin([arm, "control"])]
                boot = plot_block_bootstrap(pool, stat, block_col="plot_half",
                                            n_resamples=1500)
                rows.append({
                    "metric": metric, "treatment": arm,
                    "depth_cm": "all" if depth is None else depth,
                    "hedges_g": round(es["hedges_g"], 3),
                    "ci_lo": round(boot["lo"], 3), "ci_hi": round(boot["hi"], 3),
                    "ci_excludes_0": bool(boot["lo"] * boot["hi"] > 0),
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "cnew_alkalinity_effects.csv", index=False)

    # head-to-head: |g| of net-alkalinity vs Ca-only, pooled depths
    comp = (df[df["depth_cm"] == "all"]
            .pivot_table(index="treatment", columns="metric", values="hedges_g"))
    comp["net_beats_ca_only"] = comp["net_alkalinity"].abs() > comp["ca_only_molc"].abs()
    comp.to_csv(RESULT_DIR / "cnew_alkalinity_vs_single_ion.csv")

    lines = ["# Experiment 2: Charge-balance / alkalinity budget\n",
             "Net base-alkalinity proxy = sum(cation mol_c) - sum(anion mol_c), "
             "from the full resin panel. Compared head-to-head against single-ion "
             "(Ca-only) detection.\n",
             "## Pooled effect size: net-alkalinity vs Ca-only vs total-cation",
             comp.round(3).to_markdown(), "",
             "## Full effect-size table (with plot-clustered CIs)",
             df.to_markdown(index=False), "",
             "Where `net_beats_ca_only` is True, the multi-ion charge-balance "
             "readout gives a larger-magnitude treatment signal than the single "
             "cation that selective-ion instruments (SIA / Everest Pulsar) target "
             "- arguing for panel-based alkalinity MRV over single-ion detection."]
    (AUDIT_DIR / "cnew_alkalinity.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
