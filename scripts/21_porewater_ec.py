"""Phase: bulk-EC / pore-water-EC treatment contrast at CORRECTED depths.

Bulk soil EC is the continuous-sensor analogue of the resin cation signal.
This phase aggregates bulk EC over each resin deployment window at the matching
(corrected) depth, tests the treatment contrast (Hedges' g), and correlates
sensor EC with resin Ca. Directly exercises the sensor depth correction.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from src.config import AUDIT_DIR, CACHE_DIR, RESULT_DIR
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin
from src.stats.bootstrap import cohens_d_hedges_g


def main() -> None:
    resin = qa_clean(load_resin())
    sensors = pd.read_parquet(CACHE_DIR / "sensors.parquet")

    rows = []
    for _, r in resin.iterrows():
        ph, depth = r["plot_half"], int(r["depth_cm"])
        win = sensors[(sensors["plot_id"] == ph)
                      & (sensors["timestamp"] >= r["deploy_start"])
                      & (sensors["timestamp"] < r["deploy_end"])]
        ec = win.get(f"ec_{depth}")
        rows.append({
            "plot_half": ph, "plot_id": int(r["plot_id"]), "depth_cm": depth,
            "round": int(r["round"]), "treatment": r["treatment"],
            "ec_mean": float(ec.mean()) if ec is not None and ec.notna().any() else np.nan,
            "vwc_mean": float(win.get(f"vwc_{depth}").mean())
                        if f"vwc_{depth}" in win else np.nan,
            "resin_ca": r["ca_ppm"], "resin_mg": r["mg_ppm"],
        })
    df = pd.DataFrame(rows).dropna(subset=["ec_mean"])
    df.to_csv(RESULT_DIR / "porewater_ec.csv", index=False)

    eff = []
    for depth in (15, 40, 100):
        sub = df[df["depth_cm"] == depth]
        ctrl = sub.loc[sub["treatment"] == "control", "ec_mean"].values
        for arm in ("20", "60"):
            trt = sub.loc[sub["treatment"] == arm, "ec_mean"].values
            es = cohens_d_hedges_g(trt, ctrl)
            eff.append({"depth_cm": depth, "treatment": arm,
                        "hedges_g_ec": round(es["hedges_g"], 3),
                        "mean_diff_ec": round(es["mean_diff"], 4),
                        "n_t": es["n_t"], "n_c": es["n_c"]})
    eff_df = pd.DataFrame(eff)
    eff_df.to_csv(RESULT_DIR / "porewater_ec_effects.csv", index=False)

    corr_rows = []
    for depth in (15, 40, 100):
        sub = df[df["depth_cm"] == depth].dropna(subset=["ec_mean", "resin_ca"])
        if len(sub) >= 5:
            r_ca = stats.pearsonr(sub["ec_mean"], sub["resin_ca"])
            corr_rows.append({"depth_cm": depth, "n": len(sub),
                              "pearson_ec_vs_ca": round(r_ca.statistic, 3),
                              "p": round(r_ca.pvalue, 4)})
    corr = pd.DataFrame(corr_rows)

    lines = ["# Phase: Pore-water / bulk EC (corrected depths)\n",
             f"{len(df)} resin windows with co-located bulk-EC aggregates.\n",
             "## Bulk-EC treatment contrast (Hedges' g)",
             eff_df.to_markdown(index=False), "",
             "## Sensor EC vs resin Ca correlation", corr.to_markdown(index=False), "",
             "Uses the thermal-validated depth mapping, so shallow EC is matched "
             "to shallow resin - a correction the prior repos had inverted."]
    (AUDIT_DIR / "phase_porewater.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
