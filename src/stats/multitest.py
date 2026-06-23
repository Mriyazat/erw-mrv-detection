"""Multiple-comparison control.

Benjamini-Hochberg FDR (q-values) for the many ion x depth x round x arm tests
run across the resin panel. We control FDR *within analyte family* (primary /
negative-control / secondary) because those are distinct hypothesis classes,
and also report a pooled correction across everything for the conservative view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Return BH-adjusted p-values (q-values), NaNs preserved/ignored.

    Monotone step-up procedure; q-values are clipped to <= 1.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return out
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    out[order] = np.clip(q, 0, 1)
    return out


def add_bh(df: pd.DataFrame, pcol: str = "p_value", group: str | None = None,
           qcol: str = "q_value") -> pd.DataFrame:
    """Attach a BH q-value column, optionally computed within `group`."""
    df = df.copy()
    if group is None:
        df[qcol] = benjamini_hochberg(df[pcol].to_numpy())
    else:
        df[qcol] = np.nan
        for _, sub in df.groupby(group):
            df.loc[sub.index, qcol] = benjamini_hochberg(sub[pcol].to_numpy())
    return df
