"""Distribution-free conformal prediction intervals.

Split conformal and a leave-one-plot-out cross-conformal (Jackknife+-style)
variant for the small, plot-clustered resin data. Coverage is guaranteed
(marginally) under exchangeability with no distributional assumptions - the
rigour standard now expected for environmental ML / MRV (e.g. conformal UQ for
soil organic carbon, MDPI RS 2024; CoRel, ICML 2025).

We report empirical coverage and mean interval width, the two numbers a
reviewer checks: a valid method hits ~(1-alpha) coverage, and a *useful* one
does so with the narrowest width.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-alpha) quantile of |residuals|.

    Uses the ceil((n+1)(1-alpha))/n level (Vovk / Lei et al.) so coverage holds
    in finite samples, not just asymptotically.
    """
    r = np.sort(np.asarray(residuals, dtype=float))
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k >= n:                       # not enough calibration points -> widest
        return float(r[-1])
    return float(r[k - 1])


def split_conformal(pred_calib: np.ndarray, y_calib: np.ndarray,
                    pred_test: np.ndarray, alpha: float = 0.1) -> dict:
    """Symmetric split-conformal interval around test predictions."""
    q = conformal_quantile(np.abs(y_calib - pred_calib), alpha)
    lo, hi = pred_test - q, pred_test + q
    return {"q": q, "lo": lo, "hi": hi}


def coverage_width(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not m.any():
        return {"coverage": float("nan"), "mean_width": float("nan"), "n": 0}
    inside = (y[m] >= lo[m]) & (y[m] <= hi[m])
    return {"coverage": float(inside.mean()),
            "mean_width": float((hi[m] - lo[m]).mean()),
            "n": int(m.sum())}


def cross_conformal_from_oof(y: np.ndarray, oof_pred: np.ndarray,
                             alpha: float = 0.1) -> dict:
    """Jackknife+-style interval from leave-one-plot-out OOF residuals.

    The OOF residuals (each point predicted by a model that never saw its plot)
    are the calibration set; the (1-alpha) quantile gives a marginal half-width.
    Returns the half-width q plus realised coverage/width on the same OOF points.
    """
    y, oof_pred = np.asarray(y, dtype=float), np.asarray(oof_pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(oof_pred)
    q = conformal_quantile(np.abs(y[m] - oof_pred[m]), alpha)
    lo, hi = oof_pred - q, oof_pred + q
    cw = coverage_width(y, lo, hi)
    return {"q": q, **cw}
