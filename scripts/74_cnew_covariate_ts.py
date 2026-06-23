"""Experiment 9: covariate-informed & multivariate forecasting with Chronos-2 (GPU).

Our Experiment 4 benchmark used Chronos UNIVARIATELY. Chronos-2 (Ansari et al. 2025)
natively ingests known-future covariates and forecasts multiple coevolving
series in-context, zero-shot. Since Experiment 5 showed weather is the dominant
predictive stream, the obvious test is: do known-future weather covariates
(rain / air-temp / ET0 / VPD) rescue EC-forecast skill at the longer horizons
(24-72 h) where the univariate models decayed below seasonal-naive?

Three zero-shot modes, scored on identical rows vs seasonal-naive:
    univariate     : target EC history only
    covariate      : EC history + known-future weather covariates
    multivariate   : jointly forecast EC + VWC + soil-temp (coevolving)

GPU only. Run on Rorqual after `chronos-2` is cached (see docs/HPC.md):
    python scripts/74_cnew_covariate_ts.py --horizons 24,72
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, CACHE_DIR, RESULT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("covariate_ts")

TARGET = "ec_15"
COVARS = ["rain_mm", "air_temp_c", "et0_mm", "vpd_kpa"]   # known-future weather
MULTI_TARGETS = ["ec_15", "vwc_15", "temp_15"]            # coevolving channels
INPUT_SIZE = 168
N_WINDOWS = 20
SEASON = 24
FREQ = "h"
MIN_HOURS = 1500


def mae(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.abs(a - b)))


def build_panel() -> pd.DataFrame:
    """Regular-hourly per-plot panel: target EC + weather covars + extra channels.

    Chronos-2's predict_df needs a REGULAR frequency, so each plot is reindexed
    to a gap-free hourly grid and interpolated. An `obs` flag marks hours whose
    target was genuinely observed (pre-interpolation) so scoring never credits a
    model for predicting an interpolated value.
    """
    sensors = pd.read_parquet(CACHE_DIR / "sensors.parquet")
    weather = pd.read_parquet(CACHE_DIR / "weather_15min.parquet")
    wx = (weather.set_index("timestamp").sort_index()
          .resample(FREQ).agg({"rain_mm": "sum", "air_temp_c": "mean",
                               "et0_mm": "sum", "vpd_kpa": "mean"}))
    frames = []
    for ph, g in sensors.groupby("plot_id"):
        g = g.set_index("timestamp").sort_index()
        cols = [c for c in set([TARGET] + MULTI_TARGETS) if c in g]
        hourly = g[cols].resample(FREQ).mean()
        if hourly[TARGET].notna().sum() < MIN_HOURS:
            continue
        # keep the LARGEST contiguous deployment season (split observed target on
        # >14 d gaps, e.g. the 2025->2026 offseason) so Chronos never sees a
        # fabricated interpolated winter as context; within-season multi-day gaps
        # are retained and interpolated.
        obs_idx = hourly[TARGET].dropna().index
        gaps = (obs_idx.to_series().diff().dt.total_seconds() / 3600).to_numpy()
        seg_id = np.cumsum(np.r_[0, (gaps[1:] > 336).astype(int)])
        seg = pd.Series(seg_id, index=obs_idx)
        biggest = seg.value_counts().idxmax()
        block = obs_idx[seg.to_numpy() == biggest]
        hourly = hourly.loc[(hourly.index >= block.min()) & (hourly.index <= block.max())]
        if hourly[TARGET].notna().sum() < MIN_HOURS:
            continue
        # complete hourly grid over this block -> regular frequency
        full = pd.date_range(hourly.index.min(), hourly.index.max(), freq=FREQ)
        obs = hourly[TARGET].reindex(full).notna().to_numpy()
        hourly = hourly.reindex(full).interpolate(limit=24).ffill().bfill()
        hourly = hourly.join(wx.reindex(full).interpolate(limit=24).ffill().bfill())
        hourly = hourly.reset_index().rename(columns={"index": "ds"})
        hourly["unique_id"] = ph
        hourly["obs"] = obs
        frames.append(hourly)
    return pd.concat(frames, ignore_index=True)


def _cutoffs(n: int, horizon: int):
    for k in range(N_WINDOWS, 0, -1):
        idx = n - k * horizon
        if idx - INPUT_SIZE >= 0 and idx + horizon <= n:
            yield idx


def _seasonal_naive_mae(y_series: pd.Series, ds_index, horizon: int):
    prev = y_series.reindex(ds_index - pd.Timedelta(hours=SEASON))
    return prev


def run_mode(panel: pd.DataFrame, pipe, mode: str, horizon: int) -> list[dict]:
    """mode in {univariate, covariate, multivariate}; returns per-plot skill."""
    import torch  # noqa
    rows = []
    targets = MULTI_TARGETS if mode == "multivariate" else [TARGET]
    use_cov = (mode == "covariate")
    for uid, g in panel.groupby("unique_id"):
        g = g.sort_values("ds").reset_index(drop=True)
        s = g.set_index("ds")[TARGET]
        obs_s = g.set_index("ds")["obs"]
        n = len(g)
        y_true, y_pred, ds_keep, obs_keep = [], [], [], []
        for idx in _cutoffs(n, horizon):
            ctx = g.iloc[:idx].copy()
            fut = g.iloc[idx:idx + horizon].copy()
            try:
                context_cols = ["unique_id", "ds"] + targets + (COVARS if use_cov else [])
                context_df = ctx[context_cols]
                future_df = (fut[["unique_id", "ds"] + COVARS] if use_cov else None)
                pred = pipe.predict_df(
                    context_df, future_df=future_df,
                    prediction_length=horizon, quantile_levels=[0.5],
                    id_column="unique_id", timestamp_column="ds",
                    target=targets if mode == "multivariate" else TARGET)
                # pull the median EC prediction column robustly
                col = [c for c in pred.columns
                       if "0.5" in str(c) or c in ("predictions", TARGET, "target")]
                yp = pred[col[0]].to_numpy()[:horizon] if col else \
                    pred.select_dtypes("number").to_numpy()[:horizon, -1]
            except Exception as e:  # noqa
                log.warning("%s %s@%d failed: %s", mode, uid, idx, e)
                continue
            tgt = fut[TARGET].to_numpy()[:len(yp)]
            y_true += list(tgt)
            y_pred += list(yp[:len(tgt)])
            ds_keep += list(fut["ds"].to_numpy()[:len(tgt)])
            obs_keep += list(fut["obs"].to_numpy()[:len(tgt)])
        if not y_true:
            continue
        ds_keep = pd.to_datetime(pd.Index(ds_keep))
        prev = s.reindex(ds_keep - pd.Timedelta(hours=SEASON))
        prev_obs = obs_s.reindex(ds_keep - pd.Timedelta(hours=SEASON)).fillna(False)
        # score only on genuinely observed targets (and observed naive baseline)
        m = (prev.notna().to_numpy() & np.isfinite(y_true)
             & np.asarray(obs_keep, dtype=bool) & prev_obs.to_numpy().astype(bool))
        if m.sum() < 24:
            continue
        yt = np.asarray(y_true)[m]
        mae_sn = mae(yt, prev.to_numpy()[m])
        mae_m = mae(yt, np.asarray(y_pred)[m])
        rows.append({"mode": mode, "model": "chronos2", "horizon": horizon,
                     "plot_id": uid, "n_eval": int(m.sum()),
                     "mae": round(mae_m, 6),
                     "mae_seasonal_naive": round(mae_sn, 6),
                     "skill_vs_snaive": round(1 - mae_m / mae_sn, 3)
                     if mae_sn > 0 else np.nan,
                     "beats_snaive": bool(mae_m < mae_sn)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="24,72")
    ap.add_argument("--modes", default="univariate,covariate,multivariate")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    for cand in (os.environ.get("HF_HOME"), str(Path.home() / "hf_models")):
        if cand and Path(cand).exists():
            os.environ.setdefault("HF_HOME", cand)
            break

    try:
        import torch
        from chronos import Chronos2Pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
        log.info("loaded Chronos-2 on %s", device)
    except Exception as e:  # noqa
        log.error("Chronos-2 unavailable (%s). This phase needs chronos-2 + GPU.", e)
        return

    panel = build_panel()
    log.info("panel: %d plots, %d rows", panel["unique_id"].nunique(), len(panel))

    rows = []
    for h in horizons:
        for mode in modes:
            try:
                rows += run_mode(panel, pipe, mode, h)
                pd.DataFrame(rows).to_csv(
                    RESULT_DIR / "cnew_covariate_ts.csv", index=False)
            except Exception as e:  # noqa
                log.warning("mode %s h=%d failed: %s", mode, h, e)

    df = pd.DataFrame(rows)
    if not len(df):
        log.error("no results produced")
        return
    summ = (df.groupby(["horizon", "mode"])
            .agg(win_rate=("beats_snaive", "mean"),
                 median_skill=("skill_vs_snaive", "median"),
                 n=("plot_id", "nunique")).round(3).reset_index()
            .sort_values(["horizon", "median_skill"], ascending=[True, False]))
    lines = ["# Experiment 9: Covariate-informed & multivariate Chronos-2\n",
             "Zero-shot Chronos-2 on shallow EC: univariate vs known-future "
             "weather covariates vs multivariate (EC+VWC+temp), scored vs "
             "seasonal-naive on identical rows.\n",
             "## Skill by horizon x mode", summ.to_markdown(index=False), "",
             "Question answered: whether known-future weather covariates extend "
             "forecastable skill into the 24-72 h range where univariate models "
             "decayed below persistence (Experiment 4). A positive covariate-minus-"
             "univariate skill gap is the novel result; no gap is also publishable "
             "(weather's value is contemporaneous, not as a forecast driver)."]
    (AUDIT_DIR / "cnew_covariate_ts.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
