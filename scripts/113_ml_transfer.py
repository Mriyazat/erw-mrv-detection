#!/usr/bin/env python3
"""ML-paper item 9 [H100]: cross-arm and cross-depth transfer.

If the ERW treatment genuinely changed the EC dynamics, a forecaster trained on
one arm should transfer POORLY to another. We test the opposite hypothesis (no
detectable dynamical change) directly:

  (a) cross-ARM transfer: train PatchTST on the plots of arm A (e.g. control),
      forecast the plots of arm B (e.g. 60 t/ha), and compare the skill drop vs
      the within-arm baseline. Seamless transfer => treatment does not alter the
      forecastable dynamics (consistent with the detection-floor result).
  (b) cross-DEPTH transfer: train on one depth's EC series, forecast another.

All skill scored vs seasonal-naive on identical rows; plot is the CI unit.

Usage (Rorqual interactive H100):
    source scripts/hpc/activate_env.sh
    python scripts/113_ml_transfer.py --gpu
Writes outputs/results/ml_transfer.csv (+ audit). H100-only (needs torch).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, CACHE_DIR, RESULT_DIR, RANDOM_SEED, TREATMENT_MAP
from scripts._ts_harness import skill_rows, INPUT_SIZE, FREQ, N_WINDOWS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("transfer")

ARMS = {"control": [p for p, t in TREATMENT_MAP.items() if t == "control"],
        "20": [p for p, t in TREATMENT_MAP.items() if t == "20"],
        "60": [p for p, t in TREATMENT_MAP.items() if t == "60"]}


def hourly_panel(target: str):
    sensors = pd.read_parquet(CACHE_DIR / "sensors.parquet")
    frames, lookups = [], {}
    for ph, g in sensors.groupby("plot_id"):
        g = g.dropna(subset=[target]).set_index("timestamp").sort_index()
        h = g[target].resample(FREQ).mean().interpolate(limit=3).dropna()
        if len(h) < 1500:
            continue
        frames.append(pd.DataFrame({"unique_id": ph, "ds": h.index, "y": h.values}))
        lookups[ph] = pd.Series(h.values, index=h.index)
    return pd.concat(frames, ignore_index=True), lookups


def fit_predict(train_panel, test_panel, horizon, seed=RANDOM_SEED):
    import torch
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST
    torch.set_float32_matmul_precision("high")
    m = PatchTST(h=horizon, input_size=INPUT_SIZE, max_steps=400,
                 scaler_type="standard", random_seed=seed, accelerator="gpu",
                 enable_progress_bar=False, logger=False,
                 enable_model_summary=False)
    nf = NeuralForecast(models=[m], freq=FREQ)
    nf.fit(df=train_panel)
    # rolling-origin CV on the TEST panel only (forecasts come from the model
    # trained on the OTHER group)
    cv = nf.cross_validation(df=test_panel, n_windows=N_WINDOWS,
                             step_size=horizon, refit=False)
    return cv.reset_index().rename(columns={"PatchTST": "patchtst"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()

    rows = []

    # (a) cross-arm transfer on shallow EC
    panel, lookups = hourly_panel("ec_15")
    pairs = [("control", "60"), ("60", "control"),
             ("control", "control"), ("60", "60")]   # last two = within-arm refs
    for src, dst in pairs:
        tr = panel[panel["unique_id"].isin(ARMS[src])]
        te = panel[panel["unique_id"].isin(ARMS[dst])]
        if src == dst:  # within-arm: leave-one-plot-out so train!=test plots
            sub = []
            for hold in ARMS[dst]:
                if hold not in lookups:
                    continue
                trn = panel[panel["unique_id"].isin([p for p in ARMS[src]
                                                     if p != hold])]
                tst = panel[panel["unique_id"] == hold]
                try:
                    cv = fit_predict(trn, tst, args.horizon)
                    sub += skill_rows(cv, ["patchtst"], lookups,
                                      f"transfer:{src}->{dst}", horizon=args.horizon)
                except Exception as e:  # noqa
                    log.warning("within-arm %s hold=%s: %s", src, hold, e)
            rows += sub
        else:
            try:
                cv = fit_predict(tr, te, args.horizon)
                rows += skill_rows(cv, ["patchtst"], lookups,
                                   f"transfer:{src}->{dst}", horizon=args.horizon)
            except Exception as e:  # noqa
                log.warning("cross-arm %s->%s: %s", src, dst, e)
        pd.DataFrame(rows).to_csv(RESULT_DIR / "ml_transfer.csv", index=False)

    # (b) cross-depth transfer: train on 40 cm, forecast 15 cm and 100 cm
    for src_d, dst_d in [("ec_40", "ec_15"), ("ec_40", "ec_100")]:
        ptr, ltr = hourly_panel(src_d)
        pte, lte = hourly_panel(dst_d)
        try:
            cv = fit_predict(ptr, pte, args.horizon)
            rows += skill_rows(cv, ["patchtst"], lte,
                               f"depth:{src_d}->{dst_d}", horizon=args.horizon)
        except Exception as e:  # noqa
            log.warning("cross-depth %s->%s: %s", src_d, dst_d, e)
        pd.DataFrame(rows).to_csv(RESULT_DIR / "ml_transfer.csv", index=False)

    df = pd.DataFrame(rows)
    summ = (df.groupby("tier").agg(plots=("plot_id", "nunique"),
            median_skill=("skill_vs_snaive", "median"),
            mean_skill=("skill_vs_snaive", "mean")).reset_index())
    lines = ["# ML item 9: cross-arm and cross-depth transfer\n",
             "Skill vs seasonal-naive when the forecaster is TRAINED on one "
             "group and TESTED on another.\n",
             summ.round(3).to_markdown(index=False) if len(summ) else "_no results_",
             "\nSeamless cross-arm transfer (control->60 ~ within-arm) means the "
             "treatment does not change the forecastable EC dynamics — a "
             "dynamical restatement of the detection-floor result."]
    (AUDIT_DIR / "ml_transfer.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
