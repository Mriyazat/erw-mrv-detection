#!/usr/bin/env python3
"""ML-paper item 7 [H100]: does site-specific training beat zero-shot?

For each plot (held out), we score three forecasters on the SAME evaluation
rows (24 h shallow EC), so the comparison is apples-to-apples:

  * zero-shot Chronos-2        : no training at all (foundation prior only)
  * cross-plot PatchTST        : trained on the OTHER 11 plots (LOPO) -- tests
                                 whether other sites transfer
  * in-series PatchTST         : trained on the held-out plot's OWN past
                                 (70/30 within-series) -- "site-specific
                                 fine-tuning"

The operationally decisive question for MRV deployment: is it worth collecting
and training on site-specific history, or is a zero-shot foundation model
enough? All skill vs seasonal-naive on identical rows; plot is the CI unit.

Usage (Rorqual interactive H100):
    source scripts/hpc/activate_env.sh
    python scripts/111_ml_finetune_vs_zeroshot.py --gpu
Writes outputs/results/ml_finetune_vs_zeroshot.csv (+ audit). H100-only.
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

from src.config import RESULT_DIR, AUDIT_DIR, RANDOM_SEED
from scripts._ts_harness import (build_panel, skill_rows, rolling_cutoffs,
                                 deep_ts, INPUT_SIZE, FREQ, N_WINDOWS, HORIZON)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("finetune")


def patchtst(h, seed=RANDOM_SEED):
    from neuralforecast.models import PatchTST
    return PatchTST(h=h, input_size=INPUT_SIZE, max_steps=400,
                    scaler_type="standard", random_seed=seed, accelerator="gpu",
                    enable_progress_bar=False, logger=False,
                    enable_model_summary=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    args = ap.parse_args()
    h = args.horizon

    panel, lookups = build_panel()
    rows = []

    try:
        import torch  # noqa
        from neuralforecast import NeuralForecast
        torch.set_float32_matmul_precision("high")
    except Exception as e:  # noqa
        log.error("neuralforecast/torch required: %s", e)
        return

    for hold in sorted(lookups):
        te = panel[panel["unique_id"] == hold]
        tr_cross = panel[panel["unique_id"] != hold]

        # (1) cross-plot PatchTST: fit on others, CV on held-out (refit=False)
        try:
            nf = NeuralForecast(models=[patchtst(h)], freq=FREQ)
            nf.fit(df=tr_cross)
            cv = nf.cross_validation(df=te, n_windows=N_WINDOWS, step_size=h,
                                     refit=False).reset_index()
            cv = cv.rename(columns={"PatchTST": "patchtst_crossplot"})
            rows += skill_rows(cv, ["patchtst_crossplot"], lookups,
                               "crossplot", horizon=h)
        except Exception as e:  # noqa
            log.warning("crossplot %s: %s", hold, e)

        # (2) in-series PatchTST: train on held-out plot's own history
        try:
            nf = NeuralForecast(models=[patchtst(h)], freq=FREQ)
            cv = nf.cross_validation(df=te, n_windows=N_WINDOWS, step_size=h
                                     ).reset_index()
            cv = cv.rename(columns={"PatchTST": "patchtst_inseries"})
            rows += skill_rows(cv, ["patchtst_inseries"], lookups,
                               "inseries", horizon=h)
        except Exception as e:  # noqa
            log.warning("inseries %s: %s", hold, e)
        pd.DataFrame(rows).to_csv(RESULT_DIR / "ml_finetune_vs_zeroshot.csv",
                                  index=False)

    # (3) zero-shot Chronos-2 on the same target (reuses harness runner)
    try:
        rows += deep_ts.run_chronos(lookups, gpu=args.gpu, horizon=h)
    except Exception as e:  # noqa
        log.warning("chronos zero-shot failed: %s", e)
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "ml_finetune_vs_zeroshot.csv", index=False)

    summ = (df.groupby("tier").agg(
        plots=("plot_id", "nunique"),
        median_skill=("skill_vs_snaive", "median"),
        mean_skill=("skill_vs_snaive", "mean"),
        n_beat=("beats_snaive", "sum")).reset_index()
        .sort_values("median_skill", ascending=False))
    lines = ["# ML item 7: site-specific training vs zero-shot\n",
             "Identical 24 h eval rows; skill vs seasonal-naive.\n",
             summ.round(3).to_markdown(index=False) if len(summ) else "_no results_",
             "\ntiers: chronos (zero-shot) | crossplot (PatchTST trained on other "
             "11 plots) | inseries (PatchTST trained on the plot's own history)."]
    (AUDIT_DIR / "ml_finetune_vs_zeroshot.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
