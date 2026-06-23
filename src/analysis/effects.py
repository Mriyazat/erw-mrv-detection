"""Empirical resin effect sizes and dose-response (canonical conventions).

Resin depths come from unambiguous Sample-ID labels, so these results are
NOT affected by the sensor depth-mapping correction. Effect size = Hedges' g
(locked); dose-response slope reported in canonical ppm per t/ha AND secondary
ppm per kg/m^2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DOSE_THA, TREATMENT_ORDER
from src.stats.bootstrap import cohens_d_hedges_g, plot_block_bootstrap


def qa_clean(resin: pd.DataFrame, drop_unknown: bool = True) -> pd.DataFrame:
    df = resin[resin["qa_flag"] == ""].copy()
    if drop_unknown:
        df = df[df["treatment"] != "unknown"]
    return df


def compute_effects(resin: pd.DataFrame, ions: list[str]) -> pd.DataFrame:
    """Hedges' g (treated vs control) per ion x round x depth x dose arm."""
    df = qa_clean(resin)
    rows = []
    for ion in ions:
        for rnd in sorted(df["round"].unique()):
            for depth in sorted(df["depth_cm"].unique()):
                sub = df[(df["round"] == rnd) & (df["depth_cm"] == depth)]
                ctrl = sub.loc[sub["treatment"] == "control", ion].values
                for arm in ("20", "60"):
                    trt = sub.loc[sub["treatment"] == arm, ion].values
                    es = cohens_d_hedges_g(trt, ctrl)
                    rows.append({
                        "ion": ion, "round": rnd, "depth_cm": depth,
                        "treatment": arm, **es,
                    })
    return pd.DataFrame(rows)


def compute_dose_response(resin: pd.DataFrame, ions: list[str]) -> pd.DataFrame:
    """OLS slope of ppm vs dose per ion x round x depth.

    Reports slope in canonical ppm/(t/ha) and secondary ppm/(kg/m^2).
    """
    df = qa_clean(resin)
    dose_map = DOSE_THA  # t/ha
    rows = []
    for ion in ions:
        for rnd in sorted(df["round"].unique()):
            for depth in sorted(df["depth_cm"].unique()):
                sub = df[(df["round"] == rnd) & (df["depth_cm"] == depth)].copy()
                sub = sub.dropna(subset=[ion])
                sub["dose_tha"] = sub["treatment"].map(dose_map)
                sub = sub.dropna(subset=["dose_tha"])
                if sub["dose_tha"].nunique() < 2 or len(sub) < 3:
                    continue
                lr = stats.linregress(sub["dose_tha"], sub[ion])
                rows.append({
                    "ion": ion, "round": rnd, "depth_cm": depth, "n": len(sub),
                    "slope_ppm_per_tha": lr.slope,
                    "slope_ppm_per_kgm2": lr.slope * 10.0,  # 1 t/ha = 0.1 kg/m^2
                    "intercept_ppm": lr.intercept,
                    "r2": lr.rvalue ** 2, "p_value": lr.pvalue,
                    "stderr_per_tha": lr.stderr,
                })
    return pd.DataFrame(rows)


def pooled_effect_with_ci(resin: pd.DataFrame, ion: str, arm: str,
                          depth_cm: int | None = None,
                          n_boot: int = 2000) -> dict:
    """Plot-clustered block-bootstrap CI on pooled Hedges' g for one arm.

    Pools across rounds; resamples plot_half blocks to respect clustering.
    """
    df = qa_clean(resin)
    if depth_cm is not None:
        df = df[df["depth_cm"] == depth_cm]
    df = df[df["treatment"].isin([arm, "control"])].dropna(subset=[ion])

    def stat(d: pd.DataFrame) -> float:
        t = d.loc[d["treatment"] == arm, ion].values
        c = d.loc[d["treatment"] == "control", ion].values
        return cohens_d_hedges_g(t, c)["hedges_g"]

    res = plot_block_bootstrap(df, stat, block_col="plot_half",
                               n_resamples=n_boot)
    res.update({"ion": ion, "treatment": arm, "depth_cm": depth_cm})
    return res
