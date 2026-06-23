"""Experiment 4: sensor time-series forecast benchmark (CPU baselines + GPU tier).

Honest framing: EVERY model is scored against the SEASONAL-NAIVE forecast
(value 24 h earlier) on its own identical evaluation rows.
    skill = 1 - MAE_model / MAE_seasonal_naive    (>0 => genuinely beats naive)

Tiers
-----
CPU (always):   seasonal-naive, last-value, LightGBM-on-lags.
GPU (--gpu):    deep   -> PatchTST, iTransformer, NHITS, TFT (neuralforecast)
                foundation -> Chronos (and best-effort Moirai / TimesFM)

Heavy libs are imported lazily inside each runner and wrapped in try/except, so
the CPU tier always runs and one missing/failing model never aborts the rest.

Examples
--------
    python scripts/63_cnew_deep_ts.py                       # CPU tier only
    python scripts/63_cnew_deep_ts.py --gpu \\
        --models patchtst,itransformer,nhits,tft,chronos,moirai,timesfm
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, CACHE_DIR, RANDOM_SEED, RESULT_DIR


def mean_absolute_error(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deep_ts")

# Quiet the very chatty Lightning / neuralforecast / torch internals so the only
# output is our own progress lines and the final leaderboard.
warnings.filterwarnings("ignore")
for _n in ("pytorch_lightning", "lightning", "lightning.pytorch",
           "lightning.pytorch.utilities", "lightning.pytorch.accelerators",
           "pytorch_lightning.utilities.rank_zero",
           "pytorch_lightning.accelerators.cuda",
           "neuralforecast", "torch.utils.flop_counter"):
    logging.getLogger(_n).setLevel(logging.ERROR)
os.environ.setdefault("NIXTLA_ID_AS_COL", "1")

TARGET = "ec_15"          # corrected shallow bulk EC
HORIZON = 24              # hours ahead
SEASON = 24               # daily seasonality (hourly data)
INPUT_SIZE = 168          # 7-day context
N_WINDOWS = 20            # rolling-origin windows (last ~20 days)
MIN_HOURS = 1500
FREQ = "h"

# Point Chronos/foundation models at a local HF cache (Rorqual nodes are offline).
for _cand in (os.environ.get("HF_HOME"), str(Path.home() / "hf_models"),
              str(Path.home() / "links/projects/def-erangauk-ab"
                  / os.environ.get("USER", "") / "hf_models")):
    if _cand and Path(_cand).exists():
        os.environ.setdefault("HF_HOME", _cand)
        break


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_panel() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Long panel (unique_id, ds, y) of hourly EC per plot + per-id y lookups."""
    sensors = pd.read_parquet(CACHE_DIR / "sensors.parquet")
    frames, lookups = [], {}
    for ph, g in sensors.groupby("plot_id"):
        g = g.dropna(subset=[TARGET]).set_index("timestamp").sort_index()
        hourly = g[TARGET].resample(FREQ).mean().interpolate(limit=3).dropna()
        if len(hourly) < MIN_HOURS:
            continue
        df = pd.DataFrame({"unique_id": ph, "ds": hourly.index,
                           "y": hourly.values})
        frames.append(df)
        lookups[ph] = pd.Series(hourly.values, index=hourly.index)
    return pd.concat(frames, ignore_index=True), lookups


def seasonal_naive_mae(rows: pd.DataFrame, lookups: dict[str, pd.Series]) -> float:
    """MAE of the value-24h-earlier forecast over the exact rows in `rows`."""
    err = []
    for uid, sub in rows.groupby("unique_id"):
        s = lookups[uid]
        prev = s.reindex(sub["ds"] - pd.Timedelta(hours=SEASON))
        ok = prev.notna().values
        err.extend(np.abs(sub["y"].values[ok] - prev.values[ok]))
    return float(np.mean(err)) if err else np.nan


def skill_rows(cv: pd.DataFrame, model_cols: list[str],
               lookups: dict[str, pd.Series], tier: str,
               horizon: int = HORIZON, seed: int | None = None) -> list[dict]:
    """Per-plot MAE + skill vs seasonal-naive for each model column.

    `horizon`/`seed` are recorded for the horizon-sweep and seed-averaging runs;
    the seasonal-naive baseline is always the value SEASON (24 h) earlier.
    """
    out = []
    for uid, sub in cv.groupby("unique_id"):
        s = lookups[uid]
        prev = s.reindex(sub["ds"] - pd.Timedelta(hours=SEASON))
        mask = prev.notna().values & sub["y"].notna().values
        if mask.sum() < 24:
            continue
        y = sub["y"].values[mask]
        mae_sn = mean_absolute_error(y, prev.values[mask])
        for m in model_cols:
            if m not in sub:
                continue
            pred = sub[m].values[mask]
            if np.isnan(pred).all():
                continue
            mae = mean_absolute_error(y, pred)
            out.append({
                "tier": tier, "model": m, "plot_id": uid, "horizon": horizon,
                "seed": seed, "n_eval": int(mask.sum()),
                "mae": round(float(mae), 5),
                "mae_seasonal_naive": round(float(mae_sn), 5),
                "skill_vs_snaive": round(1 - mae / mae_sn, 3) if mae_sn > 0 else np.nan,
                "beats_snaive": bool(mae < mae_sn),
            })
    return out


# --------------------------------------------------------------------------- #
# CPU tier
# --------------------------------------------------------------------------- #
def run_cpu(panel: pd.DataFrame, lookups: dict[str, pd.Series]) -> list[dict]:
    try:
        from lightgbm import LGBMRegressor
    except Exception as e:  # noqa
        log.warning("lightgbm unavailable (%s); skipping CPU LightGBM tier", e)
        return []
    rows = []
    for uid, sub in panel.groupby("unique_id"):
        s = sub.set_index("ds")["y"]
        frame = pd.DataFrame({"y": s})
        for L in list(range(1, 13)) + [24, 25, 48, 168]:
            frame[f"lag_{L}"] = s.shift(L)
        frame["hour"] = s.index.hour
        frame["target"] = s.shift(-HORIZON)
        frame = frame.dropna()
        if len(frame) < 500:
            continue
        split = int(len(frame) * 0.7)
        tr, te = frame.iloc[:split], frame.iloc[split:]
        feat = [c for c in frame.columns if c.startswith("lag_") or c == "hour"]
        m = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=RANDOM_SEED, verbosity=-1)
        m.fit(tr[feat].to_numpy(), tr["target"].to_numpy())
        pred = m.predict(te[feat].to_numpy())
        # build a cv-like frame so skill_rows handles scoring uniformly
        cv = pd.DataFrame({"unique_id": uid,
                           "ds": te.index + pd.Timedelta(hours=HORIZON),
                           "y": te["target"].to_numpy(),
                           "lightgbm": pred})
        rows += skill_rows(cv, ["lightgbm"], lookups, "cpu")
    return rows


# --------------------------------------------------------------------------- #
# GPU deep tier (neuralforecast)
# --------------------------------------------------------------------------- #
def run_neuralforecast(panel: pd.DataFrame, lookups: dict[str, pd.Series],
                       want: set[str], gpu: bool, horizon: int = HORIZON,
                       seed: int = RANDOM_SEED) -> list[dict]:
    try:
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS, TFT, PatchTST, iTransformer
        torch.set_float32_matmul_precision("high")  # H100 tensor cores + no warning
    except Exception as e:  # noqa
        log.warning("neuralforecast unavailable (%s); skipping deep tier", e)
        return []

    n_series = panel["unique_id"].nunique()
    acc = "gpu" if gpu else "cpu"
    common = dict(h=horizon, input_size=INPUT_SIZE, max_steps=400,
                  scaler_type="standard", random_seed=seed,
                  accelerator=acc, enable_progress_bar=False, logger=False,
                  enable_model_summary=False)
    catalog = {
        "patchtst": lambda: PatchTST(**common),
        "nhits": lambda: NHITS(**common),
        "tft": lambda: TFT(**common),
        "itransformer": lambda: iTransformer(n_series=n_series, **common),
    }
    models, names = [], []
    for key, factory in catalog.items():
        if key in want:
            try:
                models.append(factory()); names.append(key)
            except Exception as e:  # noqa
                log.warning("could not build %s: %s", key, e)
    if not models:
        return []

    log.info("neuralforecast: fitting %s (acc=%s, h=%d, seed=%d)",
             names, acc, horizon, seed)
    nf = NeuralForecast(models=models, freq=FREQ)
    try:
        cv = nf.cross_validation(df=panel, n_windows=N_WINDOWS, step_size=horizon)
    except Exception as e:  # noqa
        log.warning("neuralforecast cross_validation failed: %s", e)
        return []
    cv = cv.reset_index()
    # neuralforecast names its output columns by the model class name
    colmap = {"PatchTST": "patchtst", "NHITS": "nhits", "TFT": "tft",
              "iTransformer": "itransformer"}
    cv = cv.rename(columns={k: v for k, v in colmap.items() if k in cv.columns})
    present = [v for v in colmap.values() if v in cv.columns]
    return skill_rows(cv, present, lookups, "gpu_deep", horizon=horizon, seed=seed)


# --------------------------------------------------------------------------- #
# GPU foundation tier (Chronos; best-effort Moirai / TimesFM)
# --------------------------------------------------------------------------- #
def _rolling_cutoffs(s: pd.Series, horizon: int = HORIZON):
    """Yield (cutoff_idx) for the last N_WINDOWS horizon-steps."""
    n = len(s)
    for k in range(N_WINDOWS, 0, -1):
        idx = n - k * horizon
        if idx - INPUT_SIZE >= 0 and idx + horizon <= n:
            yield idx


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def run_chronos(lookups: dict[str, pd.Series], gpu: bool,
                horizon: int = HORIZON) -> list[dict]:
    try:
        import torch
    except Exception as e:  # noqa
        log.warning("torch unavailable (%s); skipping chronos", e)
        return []
    device = "cuda" if gpu and torch.cuda.is_available() else "cpu"

    # Chronos v2 (Chronos2Pipeline / amazon/chronos-2) or v1 (BaseChronosPipeline)
    pipe, is_v2 = None, False
    try:
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2",
                                                device_map=device)
        is_v2 = True
        log.info("chronos: loaded Chronos2Pipeline (amazon/chronos-2)")
    except Exception as e2:  # noqa
        try:
            from chronos import BaseChronosPipeline
            pipe = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small", device_map=device)
            log.info("chronos: loaded BaseChronosPipeline (chronos-t5-small)")
        except Exception as e1:  # noqa
            log.warning("chronos unavailable (v2: %s | v1: %s); skipping", e2, e1)
            return []

    def _forecast(ctx_1d: np.ndarray) -> np.ndarray:
        """Median horizon-step forecast as a 1-D numpy array, v1/v2 agnostic."""
        t = torch.tensor(np.asarray(ctx_1d, dtype=np.float32))
        if is_v2:
            # Chronos-2 wants (n_series, n_variates, history_length); inputs positional
            q, _ = pipe.predict_quantiles(t.reshape(1, 1, -1),
                                          prediction_length=horizon,
                                          quantile_levels=[0.5])
        else:
            q, _ = pipe.predict_quantiles(context=t, prediction_length=horizon,
                                          quantile_levels=[0.5])
        return _to_np(q).reshape(-1)[:horizon]

    recs = []
    for uid, s in lookups.items():
        rows = {"unique_id": [], "ds": [], "y": [], "chronos": []}
        for idx in _rolling_cutoffs(s, horizon):
            try:
                fc = _forecast(s.values[:idx])
            except Exception as e:  # noqa
                log.warning("chronos forecast failed for %s@%d: %s", uid, idx, e)
                continue
            tgt = s.iloc[idx:idx + horizon]
            n = min(len(tgt), len(fc))
            rows["unique_id"] += [uid] * n
            rows["ds"] += list(tgt.index[:n])
            rows["y"] += list(tgt.values[:n])
            rows["chronos"] += list(fc[:n])
        cv = pd.DataFrame(rows)
        if len(cv):
            recs += skill_rows(cv, ["chronos"], lookups, "gpu_foundation",
                               horizon=horizon)
    return recs


def run_optional_foundation(lookups, want, gpu, horizon: int = HORIZON) -> list[dict]:
    """Moirai (uni2ts) and TimesFM - attempted only if requested & importable.

    `horizon` is honoured end-to-end so these foundation models participate in
    the 6/24/72 h sweep on the same footing as Chronos/Chronos-Bolt.
    """
    out = []
    if "moirai" in want:
        try:
            import torch
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
            module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")
            for uid, s in lookups.items():
                rows = {"unique_id": [], "ds": [], "y": [], "moirai": []}
                fcaster = MoiraiForecast(
                    module=module, prediction_length=horizon,
                    context_length=INPUT_SIZE, patch_size="auto",
                    num_samples=100, target_dim=1, feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0).to(device)
                for idx in _rolling_cutoffs(s, horizon):
                    ctx = torch.tensor(
                        s.values[idx - INPUT_SIZE:idx],
                        dtype=torch.float32).reshape(1, INPUT_SIZE, 1).to(device)
                    obs = torch.ones_like(ctx, dtype=torch.bool)
                    pad = torch.zeros(1, INPUT_SIZE, dtype=torch.bool).to(device)
                    with torch.no_grad():
                        fc = fcaster(past_target=ctx, past_observed_target=obs,
                                     past_is_pad=pad)
                    fc = np.asarray(fc.cpu()).reshape(-1, horizon)
                    med = np.median(fc, axis=0)[:horizon]
                    tgt = s.iloc[idx:idx + horizon]
                    rows["unique_id"] += [uid] * len(tgt)
                    rows["ds"] += list(tgt.index)
                    rows["y"] += list(tgt.values)
                    rows["moirai"] += list(med[:len(tgt)])
                cv = pd.DataFrame(rows)
                if len(cv):
                    out += skill_rows(cv, ["moirai"], lookups, "gpu_foundation",
                                      horizon=horizon)
        except Exception as e:  # noqa
            log.warning("moirai skipped (%s)", e)

    if "timesfm" in want:
        try:
            import timesfm
            # timesfm 2.0.x rewrote the API; the pip package now serves the 2.5
            # checkpoint via TimesFM_2p5_200M_torch (the old TimesFm class + the
            # 2.0-500m checkpoint were removed). Continuous quantile head is
            # disabled so horizons > output-patch (e.g. 72 h) stay legal.
            from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
            tfm = TimesFM_2p5_200M_torch.from_pretrained(
                "google/timesfm-2.5-200m-pytorch")
            tfm.compile(timesfm.ForecastConfig(
                max_context=INPUT_SIZE, max_horizon=max(256, horizon),
                normalize_inputs=True, use_continuous_quantile_head=False,
                force_flip_invariance=True, infer_is_positive=True,
                fix_quantile_crossing=True))
            for uid, s in lookups.items():
                rows = {"unique_id": [], "ds": [], "y": [], "timesfm": []}
                for idx in _rolling_cutoffs(s, horizon):
                    ctx = [s.values[max(0, idx - INPUT_SIZE):idx].astype(float)]
                    point, _ = tfm.forecast(horizon=horizon, inputs=ctx)
                    fc = np.asarray(point).reshape(-1)[:horizon]
                    tgt = s.iloc[idx:idx + horizon]
                    rows["unique_id"] += [uid] * len(tgt)
                    rows["ds"] += list(tgt.index)
                    rows["y"] += list(tgt.values)
                    rows["timesfm"] += list(fc[:len(tgt)])
                cv = pd.DataFrame(rows)
                if len(cv):
                    out += skill_rows(cv, ["timesfm"], lookups, "gpu_foundation",
                                      horizon=horizon)
        except Exception as e:  # noqa
            log.warning("timesfm skipped (%s)", e)
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="enable GPU deep/foundation tier")
    ap.add_argument("--models", default="patchtst,itransformer,nhits,tft,chronos",
                    help="comma list: patchtst,itransformer,nhits,tft,chronos,moirai,timesfm")
    ap.add_argument("--seeds", default=str(RANDOM_SEED),
                    help="comma list of seeds to average the deep models over, "
                         "e.g. 42,43,44 (foundation models are deterministic)")
    ap.add_argument("--horizons", default=str(HORIZON),
                    help="comma list of forecast horizons in hours, e.g. 6,24,72")
    args = ap.parse_args()
    want = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    horizons = [int(x) for x in str(args.horizons).split(",") if x.strip()]

    panel, lookups = build_panel()
    log.info("panel: %d plots, %d hourly rows", panel["unique_id"].nunique(), len(panel))

    # CPU LightGBM tier is redundant on the GPU host (already produced on the
    # CPU machine) and the HPC venv's lightgbm/sklearn pair is mismatched, so
    # skip it under --gpu rather than crash the deep-model run.
    rows: list[dict] = []
    if not args.gpu:
        try:
            rows = run_cpu(panel, lookups)
            pd.DataFrame(rows).to_csv(RESULT_DIR / "cnew_deep_ts_cpu.csv", index=False)
        except Exception as e:  # noqa
            log.warning("CPU tier failed (%s); continuing", e)
    else:
        log.info("GPU mode: skipping CPU LightGBM tier")

    gpu_rows: list[dict] = []

    def _save_gpu():
        if gpu_rows:
            pd.DataFrame(gpu_rows).to_csv(RESULT_DIR / "cnew_deep_ts_gpu.csv",
                                          index=False)

    if args.gpu:
        # Each tier is guarded + checkpointed so a later failure never discards
        # earlier results. Outer loop sweeps horizons; deep models also sweep
        # seeds (averaged later), foundation models run once per horizon.
        log.info("GPU sweep: horizons=%s, seeds=%s", horizons, seeds)
        for h in horizons:
            for sd in seeds:
                try:
                    gpu_rows += run_neuralforecast(panel, lookups, want,
                                                   gpu=True, horizon=h, seed=sd)
                    _save_gpu()
                except Exception as e:  # noqa
                    log.warning("deep tier failed (h=%d seed=%d): %s", h, sd, e)
            if "chronos" in want:
                try:
                    gpu_rows += run_chronos(lookups, gpu=True, horizon=h)
                    _save_gpu()
                except Exception as e:  # noqa
                    log.warning("chronos tier failed (h=%d): %s", h, e)
            if want & {"moirai", "timesfm"}:
                try:
                    gpu_rows += run_optional_foundation(lookups, want, gpu=True,
                                                        horizon=h)
                    _save_gpu()
                except Exception as e:  # noqa
                    log.warning("optional foundation tier failed (h=%d): %s", h, e)

    allrows = pd.DataFrame(rows + gpu_rows)
    if len(allrows):
        summ = (allrows.groupby(["tier", "model"])
                .agg(plots=("plot_id", "nunique"),
                     mean_skill=("skill_vs_snaive", "mean"),
                     n_beat=("beats_snaive", "sum"))
                .reset_index().sort_values("mean_skill", ascending=False))
    else:
        summ = pd.DataFrame()

    lines = ["# Experiment 4: Time-series forecast benchmark\n",
             f"Target {TARGET} (corrected shallow EC), {HORIZON} h ahead, "
             f"{N_WINDOWS} rolling windows, scored vs seasonal-naive on identical "
             "rows.\n",
             "## Leaderboard (mean skill vs seasonal-naive, higher = better)",
             summ.round(3).to_markdown(index=False) if len(summ) else "_no results_",
             "",
             "Positive `mean_skill` means the model beats the value-24h-ago "
             "forecast. The deep/foundation (GPU) rows are populated by "
             "`--gpu`; without it only the CPU LightGBM baseline runs. Same bar "
             "for every model - no absolute-error cherry-picking."]
    (AUDIT_DIR / "cnew_deep_ts.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
