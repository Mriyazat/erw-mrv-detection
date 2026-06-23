"""Block-bootstrap utilities for plot-clustered resin data.

The resampling unit is the PLOT (or plot-half), not the individual capsule.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED


def plot_block_bootstrap(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    block_col: str = "plot_half",
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Block-bootstrap CI for a statistic over plot clusters.

    Returns dict: stat, lo, hi, se, n_blocks.
    """
    rng = np.random.default_rng(seed)
    blocks = df[block_col].unique()
    n_blocks = len(blocks)
    if n_blocks < 2:
        s = statistic(df)
        return {"stat": s, "lo": s, "hi": s, "se": float("nan"),
                "n_blocks": n_blocks}

    blk_index = df.groupby(block_col).indices

    samples = np.empty(n_resamples, dtype=float)
    for r in range(n_resamples):
        draw = rng.choice(blocks, size=n_blocks, replace=True)
        rows = np.concatenate([blk_index[b] for b in draw])
        try:
            samples[r] = statistic(df.iloc[rows])
        except Exception:
            samples[r] = np.nan

    samples = samples[~np.isnan(samples)]
    if len(samples) < 50:
        return {"stat": statistic(df), "lo": np.nan, "hi": np.nan, "se": np.nan,
                "n_blocks": n_blocks}

    lo = np.quantile(samples, alpha / 2)
    hi = np.quantile(samples, 1 - alpha / 2)
    return {
        "stat": statistic(df),
        "lo": float(lo), "hi": float(hi),
        "se": float(samples.std(ddof=1)),
        "n_blocks": int(n_blocks),
        "n_samples_kept": int(len(samples)),
    }


def cohens_d_hedges_g(treated: np.ndarray, control: np.ndarray) -> dict:
    """Hedges' g (small-sample-corrected Cohen's d), pooled SD.

    Sign convention: positive => treated > control.
    """
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)
    treated = treated[~np.isnan(treated)]
    control = control[~np.isnan(control)]
    n_t, n_c = len(treated), len(control)
    if n_t < 2 or n_c < 2:
        return {"d": np.nan, "hedges_g": np.nan, "n_t": n_t, "n_c": n_c,
                "mean_diff": np.nan, "sd_pooled": np.nan}

    var_pooled = (
        (n_t - 1) * treated.var(ddof=1) + (n_c - 1) * control.var(ddof=1)
    ) / max(n_t + n_c - 2, 1)
    sd_pooled = np.sqrt(var_pooled)
    if sd_pooled <= 0:
        return {"d": np.nan, "hedges_g": np.nan, "n_t": n_t, "n_c": n_c,
                "mean_diff": float(treated.mean() - control.mean()),
                "sd_pooled": 0.0}
    d = (treated.mean() - control.mean()) / sd_pooled
    j = 1 - 3 / (4 * (n_t + n_c) - 9)
    return {
        "d": float(d), "hedges_g": float(d * j),
        "n_t": n_t, "n_c": n_c,
        "mean_diff": float(treated.mean() - control.mean()),
        "sd_pooled": float(sd_pooled),
    }
