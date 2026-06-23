"""Experiment 1: cation-exchange buffering & CDR lag from the depth x time profile.

Hypothesis (Kanzaki 2025; Beerling 2025): released Ca/Mg is initially retained
on the soil exchange complex in the upper profile, so the aqueous base-cation
excess appears shallow-first and propagates downward slowly - the deep layers
can even sit at/below control early on. That retention is a *lag* between
weathering and the deep alkalinity export that ultimately defines CDR.

We quantify, per round (time) and depth, the treated-minus-control base-cation
excess (Ca + Mg, mol_c equivalents), a shallow->deep retention index, and how
the profile evolves R1 (Jul) -> R3 (Oct).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, FIGURE_DIR, ION_CHARGE, ION_MOLAR_MASS, RESULT_DIR
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin
from src.stats.bootstrap import plot_block_bootstrap

DEPTHS = [15, 40, 100]
ROUND_LABEL = {1: "R1 (Jul)", 2: "R2 (Aug)", 3: "R3 (Sep-Oct)"}


def base_cation_molc(df: pd.DataFrame) -> pd.Series:
    """Ca + Mg in mol_c (charge-equivalents) per capsule."""
    out = np.zeros(len(df))
    for ion in ("ca_ppm", "mg_ppm"):
        out = out + (df[ion].fillna(0) / ION_MOLAR_MASS[ion]) * ION_CHARGE[ion]
    return pd.Series(out, index=df.index)


def main() -> None:
    resin = qa_clean(load_resin())
    resin = resin[resin["treatment"].isin(["control", "60"])].copy()
    resin["base_molc"] = base_cation_molc(resin)

    rows = []
    for rnd in (1, 2, 3):
        for depth in DEPTHS:
            sub = resin[(resin["round"] == rnd) & (resin["depth_cm"] == depth)]
            trt = sub.loc[sub["treatment"] == "60", "base_molc"]
            ctrl = sub.loc[sub["treatment"] == "control", "base_molc"]
            if len(trt) < 2 or len(ctrl) < 2:
                continue

            def excess(d):
                t = d.loc[d["treatment"] == "60", "base_molc"].mean()
                c = d.loc[d["treatment"] == "control", "base_molc"].mean()
                return t - c

            boot = plot_block_bootstrap(sub, excess, block_col="plot_half",
                                        n_resamples=2000)
            rows.append({
                "round": rnd, "round_label": ROUND_LABEL[rnd], "depth_cm": depth,
                "excess_molc": boot["stat"], "ci_lo": boot["lo"],
                "ci_hi": boot["hi"], "n_t": int((sub["treatment"] == "60").sum()),
                "n_c": int((sub["treatment"] == "control").sum()),
                "excess_positive": boot["stat"] > 0,
            })
    prof = pd.DataFrame(rows)
    prof.to_csv(RESULT_DIR / "cnew_cec_lag_profile.csv", index=False)

    # Retention index per round: shallow excess that is NOT seen at depth.
    ret_rows = []
    for rnd in (1, 2, 3):
        p = prof[prof["round"] == rnd].set_index("depth_cm")["excess_molc"]
        if 15 in p.index and 100 in p.index:
            shallow, deep = p[15], p[100]
            ret = (shallow - deep)
            ret_rows.append({
                "round": rnd, "round_label": ROUND_LABEL[rnd],
                "excess_15cm": round(float(shallow), 4),
                "excess_100cm": round(float(deep), 4),
                "shallow_minus_deep": round(float(ret), 4),
                "deep_below_control": bool(deep < 0),
            })
    ret = pd.DataFrame(ret_rows)
    ret.to_csv(RESULT_DIR / "cnew_cec_lag_retention.csv", index=False)

    # figure: depth profile per round
    fig, ax = plt.subplots(figsize=(6, 5))
    for rnd in (1, 2, 3):
        p = prof[prof["round"] == rnd]
        if len(p):
            ax.plot(p["excess_molc"], p["depth_cm"], "-o", label=ROUND_LABEL[rnd])
            ax.fill_betweenx(p["depth_cm"], p["ci_lo"], p["ci_hi"], alpha=0.12)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.invert_yaxis()
    ax.set_xlabel("Treated(60) - Control base-cation excess (mol_c, Ca+Mg)")
    ax.set_ylabel("Depth (cm)")
    ax.set_title("Experiment 1: base-cation excess depth profile over season")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cnew_cec_lag_profile.png", dpi=130)
    plt.close(fig)

    lines = ["# Experiment 1: Cation-exchange buffering / CDR lag\n",
             "Base-cation (Ca+Mg) treated-minus-control excess by depth and "
             "round, in charge-equivalents (mol_c). Plot-clustered bootstrap CIs.\n",
             "## Depth x time profile",
             prof.round(4).to_markdown(index=False), "",
             "## Shallow-vs-deep retention",
             ret.to_markdown(index=False), "",
             "Interpretation: a shallow-positive / deep-at-or-below-control "
             "profile is the fingerprint of exchange-complex retention buffering "
             "the downward alkalinity flux - i.e. an empirical CDR *lag*. This is "
             "the field-scale analogue of the cation-exchange delay modelled by "
             "Kanzaki (2025) and discussed by Beerling (2025), and it cautions "
             "against equating shallow cation appearance with realised CDR.",
             "", f"Figure: {FIGURE_DIR/'cnew_cec_lag_profile.png'}"]
    (AUDIT_DIR / "cnew_cec_lag.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
