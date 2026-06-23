#!/usr/bin/env python3
"""ML-paper item 6 [H100]: full zero-shot foundation-model bake-off.

The first benchmark of the 2025-26 time-series FOUNDATION models on ERW soil
sensors. Every model is used ZERO-SHOT (no site training) and scored against the
seasonal-naive baseline on identical rows (skill = 1 - MAE/MAE_naive), with the
plot as the resampling unit for CIs (built locally by 65/115).

Panel (best-effort; each guarded so one missing model never aborts the rest):
  * Chronos-2            (amazon/chronos-2)            -- run via harness 63
  * Chronos-Bolt         (amazon/chronos-bolt-base)
  * TimesFM-2            (google/timesfm-2.0-500m)     -- run via harness 63
  * Moirai-1.1-R         (Salesforce/moirai-1.1-R-*)   -- run via harness 63
  * Lag-Llama            (time-series-foundation-models/Lag-Llama)
  * TabPFN-TS            (automl/tabpfn-time-series)

Usage (Rorqual interactive H100):
    source scripts/hpc/activate_env.sh
    python scripts/110_ml_foundation_panel.py --gpu \
        --models chronos,chronos_bolt,timesfm,moirai,lagllama,tabpfn \
        --horizons 6,24,72
Writes outputs/results/ml_foundation_panel.csv (+ audit).
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
from scripts._ts_harness import (  # noqa: E402
    build_panel, skill_rows, rolling_cutoffs, deep_ts, INPUT_SIZE, HORIZON)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("foundation_panel")

# point at offline HF cache
for _c in (os.environ.get("HF_HOME"), str(Path.home() / "hf_models")):
    if _c and Path(_c).exists():
        os.environ.setdefault("HF_HOME", _c)
        break


def _device(gpu):
    import torch
    return "cuda" if gpu and torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# new runners (the models 63 does not already cover)
# --------------------------------------------------------------------------- #
def run_chronos_bolt(lookups, gpu, horizon=HORIZON):
    try:
        import torch
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base", device_map=_device(gpu))
    except Exception as e:  # noqa
        log.warning("chronos-bolt skipped (%s)", e)
        return []
    out = []
    for uid, s in lookups.items():
        rows = {"unique_id": [], "ds": [], "y": [], "chronos_bolt": []}
        for idx in rolling_cutoffs(s, horizon):
            try:
                ctx = torch.tensor(s.values[:idx], dtype=torch.float32)
                # newer chronos renamed the first arg context->inputs; pass positionally
                out_q = pipe.predict_quantiles(ctx,
                                               prediction_length=horizon,
                                               quantile_levels=[0.5])
                q = out_q[0] if isinstance(out_q, tuple) else out_q
                fc = np.asarray(q.detach().cpu() if hasattr(q, "detach")
                                else q).reshape(-1)[:horizon]
            except Exception as e:  # noqa
                log.warning("bolt %s@%d: %s", uid, idx, e); continue
            tgt = s.iloc[idx:idx + horizon]
            n = min(len(tgt), len(fc))
            rows["unique_id"] += [uid] * n; rows["ds"] += list(tgt.index[:n])
            rows["y"] += list(tgt.values[:n]); rows["chronos_bolt"] += list(fc[:n])
        cv = pd.DataFrame(rows)
        if len(cv):
            out += skill_rows(cv, ["chronos_bolt"], lookups, "gpu_foundation",
                              horizon=horizon)
    return out


def run_lagllama(lookups, gpu, horizon=HORIZON):
    """Lag-Llama zero-shot (best-effort; checkpoint must be in HF cache)."""
    try:
        import torch
        from lag_llama.gluon.estimator import LagLlamaEstimator
        ckpt_path = os.path.join(os.environ.get("HF_HOME", ""),
                                 "lag-llama.ckpt")
        ckpt = torch.load(ckpt_path, map_location=_device(gpu))
        est_args = ckpt["hyper_parameters"]["model_kwargs"]
        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path, prediction_length=horizon,
            context_length=INPUT_SIZE,
            input_size=est_args["input_size"], n_layer=est_args["n_layer"],
            n_embd_per_head=est_args["n_embd_per_head"],
            n_head=est_args["n_head"],
            scaling=est_args["scaling"], time_feat=est_args["time_feat"],
            device=_device(gpu))
        predictor = estimator.create_predictor(
            estimator.create_transformation(),
            estimator.create_lightning_module())
    except Exception as e:  # noqa
        log.warning("lag-llama skipped (%s)", e)
        return []
    from gluonts.dataset.pandas import PandasDataset
    out = []
    for uid, s in lookups.items():
        rows = {"unique_id": [], "ds": [], "y": [], "lagllama": []}
        for idx in rolling_cutoffs(s, horizon):
            try:
                hist = s.iloc[:idx]
                ds = PandasDataset(pd.DataFrame({"target": hist.values},
                                                index=hist.index),
                                   target="target", freq="h")
                fc = list(predictor.predict(ds))[0]
                med = np.median(fc.samples, axis=0)[:horizon]
            except Exception as e:  # noqa
                log.warning("lagllama %s@%d: %s", uid, idx, e); continue
            tgt = s.iloc[idx:idx + horizon]
            n = min(len(tgt), len(med))
            rows["unique_id"] += [uid] * n; rows["ds"] += list(tgt.index[:n])
            rows["y"] += list(tgt.values[:n]); rows["lagllama"] += list(med[:n])
        cv = pd.DataFrame(rows)
        if len(cv):
            out += skill_rows(cv, ["lagllama"], lookups, "gpu_foundation",
                              horizon=horizon)
    return out


def run_tabpfn_ts(lookups, gpu, horizon=HORIZON):
    """TabPFN-TS zero-shot tabular foundation forecaster (best-effort)."""
    try:
        from tabpfn_time_series import TabPFNTimeSeriesPredictor
        predictor = TabPFNTimeSeriesPredictor()
    except Exception as e:  # noqa
        log.warning("tabpfn-ts skipped (%s)", e)
        return []
    out = []
    for uid, s in lookups.items():
        rows = {"unique_id": [], "ds": [], "y": [], "tabpfn": []}
        for idx in rolling_cutoffs(s, horizon):
            try:
                hist = s.iloc[max(0, idx - INPUT_SIZE):idx]
                train_df = pd.DataFrame({"target": hist.values}, index=hist.index)
                future_index = s.index[idx:idx + horizon]
                pred = predictor.predict(train_df, future_index)
                fc = np.asarray(pred).reshape(-1)[:horizon]
            except Exception as e:  # noqa
                log.warning("tabpfn %s@%d: %s", uid, idx, e); continue
            tgt = s.iloc[idx:idx + horizon]
            n = min(len(tgt), len(fc))
            rows["unique_id"] += [uid] * n; rows["ds"] += list(tgt.index[:n])
            rows["y"] += list(tgt.values[:n]); rows["tabpfn"] += list(fc[:n])
        cv = pd.DataFrame(rows)
        if len(cv):
            out += skill_rows(cv, ["tabpfn"], lookups, "gpu_foundation",
                              horizon=horizon)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--models",
                    default="chronos,chronos_bolt,timesfm,moirai,lagllama,tabpfn")
    ap.add_argument("--horizons", default="6,24,72")
    args = ap.parse_args()
    want = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    panel, lookups = build_panel()
    log.info("panel: %d plots, %d hourly rows",
             panel["unique_id"].nunique(), len(panel))

    rows = []

    def save():
        if rows:
            pd.DataFrame(rows).to_csv(RESULT_DIR / "ml_foundation_panel.csv",
                                      index=False)

    for h in horizons:
        # models already implemented in harness 63
        if "chronos" in want:
            try:
                rows += deep_ts.run_chronos(lookups, gpu=args.gpu, horizon=h); save()
            except Exception as e:  # noqa
                log.warning("chronos failed h=%d: %s", h, e)
        if "chronos_bolt" in want:
            try:
                rows += run_chronos_bolt(lookups, args.gpu, h); save()
            except Exception as e:  # noqa
                log.warning("bolt failed h=%d: %s", h, e)
        if "lagllama" in want:
            try:
                rows += run_lagllama(lookups, args.gpu, h); save()
            except Exception as e:  # noqa
                log.warning("lagllama failed h=%d: %s", h, e)
        if "tabpfn" in want:
            try:
                rows += run_tabpfn_ts(lookups, args.gpu, h); save()
            except Exception as e:  # noqa
                log.warning("tabpfn failed h=%d: %s", h, e)
        # moirai / timesfm via harness 63 -- now horizon-aware, so they sweep
        # 6/24/72 h on the same footing as Chronos/Chronos-Bolt.
        if want & {"moirai", "timesfm"}:
            try:
                rows += deep_ts.run_optional_foundation(lookups, want,
                                                        gpu=args.gpu, horizon=h)
                save()
            except Exception as e:  # noqa
                log.warning("moirai/timesfm failed h=%d: %s", h, e)

    df = pd.DataFrame(rows)
    if len(df):
        summ = (df.groupby(["model", "horizon"])
                .agg(plots=("plot_id", "nunique"),
                     median_skill=("skill_vs_snaive", "median"),
                     mean_skill=("skill_vs_snaive", "mean"),
                     n_beat=("beats_snaive", "sum")).reset_index()
                .sort_values(["horizon", "median_skill"], ascending=[True, False]))
    else:
        summ = pd.DataFrame()
    lines = ["# ML item 6: zero-shot foundation-model bake-off (soil EC)\n",
             "All models zero-shot; skill vs seasonal-naive on identical rows.\n",
             summ.round(3).to_markdown(index=False) if len(summ) else "_no results_"]
    (AUDIT_DIR / "ml_foundation_panel.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\npull to Mac, then: python scripts/115_ml_foundation_leaderboard.py")


if __name__ == "__main__":
    main()
