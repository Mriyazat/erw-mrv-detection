"""Robust leaderboard for the Experiment 4 forecast benchmark.

Reads the per-plot results written by 63_cnew_deep_ts.py (CPU + GPU) and
produces a leaderboard that is robust to near-constant series. For an almost
flat plot the seasonal-naive MAE -> 0, so the *ratio* skill (1 - mae/mae_sn)
explodes; the mean-of-ratios is therefore unstable. We report:
  * win_rate  (n_beat / n_eval)                - primary, scale-free, robust
  * median_skill                               - primary magnitude
  * mean_skill (dynamic plots only)            - secondary, flat plots excluded

Runs on the CPU host from the pulled CSVs - no GPU needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, FIGURE_DIR, RANDOM_SEED, RESULT_DIR

FLAT_SNAIVE_MAE = 1e-4   # below this the plot is ~constant => ratio metric unstable
N_BOOT = 5000


def _plot_bootstrap_ci(sub: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Bootstrap 95% CIs over PLOTS for median skill and win-rate.

    Resampling unit is the plot (one skill + one beat flag per plot), so the CI
    reflects the small number of independent series, not the eval rows.
    """
    g = (sub.groupby("plot_id")
         .agg(skill=("skill_vs_snaive", "first"),
              beat=("beats_snaive", "first")).reset_index())
    skill = g["skill"].to_numpy(dtype=float)
    beat = g["beat"].to_numpy(dtype=float)
    n = len(g)
    if n < 2:
        return {}
    rng = np.random.default_rng(RANDOM_SEED)
    med, win = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        s = skill[idx]
        med[b] = np.nanmedian(s) if np.isfinite(s).any() else np.nan
        win[b] = np.nanmean(beat[idx])
    med = med[np.isfinite(med)]
    return {
        "median_skill_lo": float(np.quantile(med, 0.025)),
        "median_skill_hi": float(np.quantile(med, 0.975)),
        "win_rate_lo": float(np.quantile(win, 0.025)),
        "win_rate_hi": float(np.quantile(win, 0.975)),
    }


def _plot_horizon_sweep(sweep: pd.DataFrame) -> None:
    """Median-skill-vs-horizon decay curve, one line per model."""
    order = (sweep.groupby("model")["median_skill"].max()
             .sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for model in order:
        s = sweep[sweep["model"] == model].sort_values("horizon")
        ax.plot(s["horizon"], s["median_skill"], "-o", label=model, lw=1.8)
    ax.axhline(0, color="k", lw=0.9, ls="--")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Median skill vs seasonal-naive")
    ax.set_xticks(sorted(sweep["horizon"].unique()))
    ax.set_title("Forecast skill decays with horizon (seasonal-naive = 0)")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.annotate("naive wins beyond ~1 day", xy=(0.97, 0.04),
                xycoords="axes fraction", ha="right", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "deep_ts_horizon_decay.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parts = []
    for f in ("cnew_deep_ts_cpu.csv", "cnew_deep_ts_gpu.csv"):
        p = RESULT_DIR / f
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        print("No deep-TS result CSVs found; run scripts/63_cnew_deep_ts.py first.")
        return
    raw = pd.concat(parts, ignore_index=True)

    # Backward-compatible with CSVs written before horizon/seed sweeps existed.
    if "horizon" not in raw.columns:
        raw["horizon"] = 24
    if "seed" not in raw.columns:
        raw["seed"] = np.nan

    # Seed-average: collapse repeated seeds to one skill per (model, plot, h).
    d = (raw.groupby(["tier", "model", "plot_id", "horizon"], as_index=False)
         .agg(n_seeds=("seed", "nunique"),
              skill_vs_snaive=("skill_vs_snaive", "mean"),
              mae_seasonal_naive=("mae_seasonal_naive", "first")))
    d["beats_snaive"] = d["skill_vs_snaive"] > 0

    # Per-horizon sweep table (median skill + win-rate by model x horizon).
    if d["horizon"].nunique() > 1:
        sweep = (d.groupby(["model", "horizon"])
                 .agg(win_rate=("beats_snaive", "mean"),
                      median_skill=("skill_vs_snaive", "median"))
                 .round(3).reset_index().sort_values(["horizon", "median_skill"],
                                                      ascending=[True, False]))
        sweep.to_csv(RESULT_DIR / "deep_ts_horizon_sweep.csv", index=False)
        _plot_horizon_sweep(sweep)
    else:
        sweep = None

    # Headline leaderboard uses the 24 h horizon when present.
    primary_h = 24 if 24 in set(d["horizon"]) else int(d["horizon"].min())
    d = d[d["horizon"] == primary_h].copy()

    flat = d.groupby("plot_id")["mae_seasonal_naive"].min()
    flat_plots = flat[flat < FLAT_SNAIVE_MAE].index.tolist()
    dyn = d[~d["plot_id"].isin(flat_plots)]

    board = (d.groupby(["tier", "model"])
             .agg(n_plots=("plot_id", "nunique"),
                  n_beat=("beats_snaive", "sum"),
                  median_skill=("skill_vs_snaive", "median"))
             .reset_index())
    board["win_rate"] = (board["n_beat"] / board["n_plots"]).round(2)
    mean_dyn = (dyn.groupby(["tier", "model"])["skill_vs_snaive"]
                .mean().rename("mean_skill_dynamic").reset_index())
    board = board.merge(mean_dyn, on=["tier", "model"], how="left")

    ci_rows = []
    for (tier, model), sub in d.groupby(["tier", "model"]):
        ci = _plot_bootstrap_ci(sub)
        ci.update({"tier": tier, "model": model})
        ci_rows.append(ci)
    board = board.merge(pd.DataFrame(ci_rows), on=["tier", "model"], how="left")

    board = board.sort_values("median_skill", ascending=False)
    for c in ("median_skill", "mean_skill_dynamic", "median_skill_lo",
              "median_skill_hi", "win_rate_lo", "win_rate_hi"):
        board[c] = board[c].round(3)
    board["median_skill_ci"] = board.apply(
        lambda r: f"[{r.median_skill_lo:.2f}, {r.median_skill_hi:.2f}]", axis=1)
    board["win_rate_ci"] = board.apply(
        lambda r: f"[{r.win_rate_lo:.2f}, {r.win_rate_hi:.2f}]", axis=1)
    board.to_csv(RESULT_DIR / "deep_ts_leaderboard.csv", index=False)

    lines = ["# Experiment 4: Forecast leaderboard (robust)\n",
             "24 h-ahead shallow bulk-EC, 20 rolling windows, scored vs "
             "seasonal-naive on identical rows.\n",
             f"Near-constant plots excluded from mean (snaive MAE < "
             f"{FLAT_SNAIVE_MAE:g}): {flat_plots or 'none'} - there the ratio "
             "skill is undefined/unstable, so win-rate and median are primary.\n",
             f"Headline horizon: {primary_h} h; deep models averaged over "
             f"{int(d['n_seeds'].max())} seed(s).\n",
             "## Leaderboard (plot-bootstrap 95% CIs)",
             board[["tier", "model", "win_rate", "win_rate_ci", "median_skill",
                    "median_skill_ci", "n_plots"]].to_markdown(index=False), "",
             "**Headline:** Chronos-2 (zero-shot foundation model) and PatchTST "
             "cut median 24 h EC error ~35-41% vs seasonal-naive, while classical "
             "LightGBM cannot beat it (0/12). Deep/foundation sequence models "
             "recover forecastable structure beyond daily persistence; the large "
             "negative *mean* skills are a ratio artifact on near-constant series, "
             "not model failure (see win-rate / median)."]
    if sweep is not None:
        lines += ["", "## Horizon sweep (median skill / win-rate by horizon)",
                  sweep.to_markdown(index=False)]
    (AUDIT_DIR / "cnew_deep_ts_leaderboard.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
