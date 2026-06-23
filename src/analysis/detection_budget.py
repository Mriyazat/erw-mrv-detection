"""Reusable aqueous-phase ERW detection-budget calculator.

Turns the per-cell variability into a design tool: given a target true effect
(in control-SD units), how much statistical confidence does a design of
N plot-halves/arm provide, and what replication is needed for a target power?

Analytic two-sample t-test power (statsmodels) keeps it fast and reusable
across plots x depths x seasons x ions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower

_POWER = TTestIndPower()


def power_for(effect_d: float, n_per_arm: int, alpha: float = 0.05) -> float:
    if n_per_arm < 2:
        return np.nan
    return float(_POWER.power(effect_size=effect_d, nobs1=n_per_arm,
                              alpha=alpha, ratio=1.0, alternative="two-sided"))


def n_for_power(effect_d: float, power: float = 0.8, alpha: float = 0.05) -> float:
    if effect_d == 0:
        return np.inf
    try:
        return float(_POWER.solve_power(effect_size=abs(effect_d), power=power,
                                        alpha=alpha, ratio=1.0,
                                        alternative="two-sided"))
    except Exception:
        return np.nan


def mde_for_power(n_per_arm: int, power: float = 0.8, alpha: float = 0.05) -> float:
    """Minimum detectable effect (in SD units) at a given design."""
    if n_per_arm < 2:
        return np.nan
    try:
        return float(_POWER.solve_power(nobs1=n_per_arm, power=power, alpha=alpha,
                                        ratio=1.0, alternative="two-sided"))
    except Exception:
        return np.nan


def detection_budget(cells: pd.DataFrame, n_per_arm: int = 4,
                     target_effects=(0.5, 1.0, 1.5, 2.0)) -> pd.DataFrame:
    """Expand a table of (ion, depth, season, control_sd, control_mean) cells.

    Returns one row per (cell x target_effect) with power, MDE, and required N.
    """
    rows = []
    for _, c in cells.iterrows():
        mde = mde_for_power(n_per_arm)
        for d in target_effects:
            rows.append({
                **{k: c[k] for k in cells.columns},
                "n_per_arm": n_per_arm,
                "target_effect_sd": d,
                "power": round(power_for(d, n_per_arm), 3),
                "mde_sd_at_design": round(mde, 2),
                "n_needed_for_80pct": round(n_for_power(d), 1),
                "target_effect_ppm": (round(d * c["control_sd"], 3)
                                      if "control_sd" in c else np.nan),
            })
    return pd.DataFrame(rows)
