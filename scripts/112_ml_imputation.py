#!/usr/bin/env python3
"""ML-paper item 8: sensor gap-filling benchmark (the useful foundation result).

ERW sensor streams have real gaps (shallow probe ~8-10% null, winter joins).
We mask synthetic contiguous gaps in the hourly EC series and reconstruct them,
comparing classical interpolators against zero-shot foundation models. This is
the operationally valuable role for foundation models in MRV QA/QC -- a
sensor-utility claim, not a CDR-detection claim.

Methods
  classical (CPU, always): linear interpolation (uses both endpoints),
      time-spline, forward-fill, seasonal (value 24 h earlier).
  foundation (--gpu): Chronos-2 forward fill from the pre-gap context; and
      (best-effort) a forward+backward blend.

Scoring: MAE on the masked points only, per gap length (6/24/72 h), aggregated
with the PLOT as the resampling unit.

Usage:
    python scripts/112_ml_imputation.py                 # classical only (local)
    python scripts/112_ml_imputation.py --gpu           # + Chronos (Rorqual)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, RESULT_DIR
from scripts._ts_harness import build_panel, mean_absolute_error, INPUT_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("imputation")
for _c in (os.environ.get("HF_HOME"), str(Path.home() / "hf_models")):
    if _c and Path(_c).exists():
        os.environ.setdefault("HF_HOME", _c)
        break

SEED = 42
GAP_LENS = (6, 24, 72)       # hours
N_GAPS_PER_PLOT = 12
rng = np.random.default_rng(SEED)


def make_gaps(s: pd.Series, gap_len: int):
    """Pick non-overlapping interior gaps with enough left context."""
    n = len(s)
    gaps = []
    lo = max(INPUT_SIZE, 24)
    hi = n - gap_len - 24
    if hi <= lo:
        return gaps
    tries = 0
    while len(gaps) < N_GAPS_PER_PLOT and tries < N_GAPS_PER_PLOT * 20:
        tries += 1
        start = int(rng.integers(lo, hi))
        if all(abs(start - g) > 2 * gap_len for g in gaps):
            gaps.append(start)
    return gaps


def classical_fills(s_full: np.ndarray, start: int, L: int):
    """Return dict method -> filled values over [start, start+L) using a series
    where the gap is set to NaN (interpolation may use the right endpoint)."""
    y = s_full.astype(float).copy()
    truth = y[start:start + L].copy()
    y[start:start + L] = np.nan
    ser = pd.Series(y)
    out = {}
    out["linear"] = ser.interpolate("linear").values[start:start + L]
    out["spline"] = ser.interpolate("spline", order=2).values[start:start + L] \
        if L >= 3 else out["linear"]
    out["ffill"] = ser.ffill().values[start:start + L]
    # seasonal: value 24 h earlier
    seas = np.array([s_full[start + i - 24] if start + i - 24 >= 0 else np.nan
                     for i in range(L)])
    out["seasonal"] = seas
    return truth, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    panel, lookups = build_panel()
    log.info("panel: %d plots", len(lookups))

    chronos = None
    if args.gpu:
        try:
            import torch
            from chronos import Chronos2Pipeline
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            chronos = Chronos2Pipeline.from_pretrained("amazon/chronos-2",
                                                       device_map=dev)
            log.info("chronos-2 loaded on %s", dev)
        except Exception as e:  # noqa
            log.warning("chronos unavailable (%s); classical only", e)

    rows = []
    for uid, s in lookups.items():
        arr = s.values.astype(float)
        for L in GAP_LENS:
            for start in make_gaps(s, L):
                truth, fills = classical_fills(arr, start, L)
                for method, pred in fills.items():
                    m = np.isfinite(truth) & np.isfinite(pred)
                    if m.sum() == 0:
                        continue
                    rows.append({"plot_id": uid, "gap_len": L, "method": method,
                                 "mae": mean_absolute_error(truth[m], pred[m])})
                if chronos is not None:
                    try:
                        import torch
                        ctx = torch.tensor(arr[:start], dtype=torch.float32)
                        q, _ = chronos.predict_quantiles(
                            ctx.reshape(1, 1, -1), prediction_length=L,
                            quantile_levels=[0.5])
                        fc = np.asarray(q.detach().cpu()).reshape(-1)[:L]
                        m = np.isfinite(truth) & np.isfinite(fc)
                        if m.sum():
                            rows.append({"plot_id": uid, "gap_len": L,
                                         "method": "chronos_fwd",
                                         "mae": mean_absolute_error(truth[m], fc[m])})
                    except Exception as e:  # noqa
                        log.warning("chronos fill %s@%d: %s", uid, start, e)

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "ml_imputation.csv", index=False)
    summ = (df.groupby(["gap_len", "method"])
            .agg(mean_mae=("mae", "mean"), median_mae=("mae", "median"),
                 n=("mae", "size")).reset_index()
            .sort_values(["gap_len", "median_mae"]))
    print(summ.round(5).to_string(index=False))

    lines = ["# ML item 8: sensor gap-filling benchmark (EC)\n",
             "MAE on masked points only; lower is better. Resampling unit = plot.\n",
             summ.round(5).to_markdown(index=False),
             "\nClassical interpolation uses both gap endpoints (an easier task "
             "than forecasting); foundation forward-fill uses only the left "
             "context. The honest read is which method wins at each gap length — "
             "a sensor-utility (QA/QC) result, not a CDR claim."]
    (AUDIT_DIR / "ml_imputation.md").write_text("\n".join(lines))
    print("wrote results + audit")


if __name__ == "__main__":
    main()
