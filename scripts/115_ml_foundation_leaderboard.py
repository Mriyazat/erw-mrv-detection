#!/usr/bin/env python3
"""ML-paper item 6 (post-processing, local): foundation-panel leaderboard + CIs.

Reads the per-plot rows written on the H100 by 110_ml_foundation_panel.py (and,
if present, the deep/foundation rows from 63's cnew_deep_ts_gpu.csv) and builds
a robust leaderboard: median skill + win-rate with PLOT-bootstrap 95% CIs (the
resampling unit is the plot, not the eval row), per horizon. Mirrors the robust
metric choice in scripts/65_deep_ts_leaderboard.py.

Local (Mac, ../.venv). Run after `scripts/hpc/sync.sh pull`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import RESULT_DIR, AUDIT_DIR, FIGURE_DIR, RANDOM_SEED

FLAT = 1e-4
N_BOOT = 5000
rng = np.random.default_rng(RANDOM_SEED)


def load():
    frames = []
    for f in ("ml_foundation_panel.csv", "cnew_deep_ts_gpu.csv",
              "cnew_deep_ts_cpu.csv"):
        p = RESULT_DIR / f
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("no foundation/deep result CSVs found — run 110 on H100 "
                         "and `scripts/hpc/sync.sh pull` first.")
    df = pd.concat(frames, ignore_index=True)
    if "horizon" not in df:
        df["horizon"] = 24
    return df


def boot_ci(sub):
    g = (sub.groupby("plot_id")
            .agg(skill=("skill_vs_snaive", "median"),
                 beat=("beats_snaive", "mean")).reset_index())
    dyn = g[sub.groupby("plot_id")["mae_seasonal_naive"].median().values > FLAT] \
        if "mae_seasonal_naive" in sub else g
    msk, wr = [], []
    for _ in range(N_BOOT):
        bs = g.sample(len(g), replace=True)
        msk.append(bs["skill"].median())
        wr.append(bs["beat"].mean())
    return {"median_skill": float(g["skill"].median()),
            "skill_lo": float(np.percentile(msk, 2.5)),
            "skill_hi": float(np.percentile(msk, 97.5)),
            "win_rate": float(g["beat"].mean()),
            "wr_lo": float(np.percentile(wr, 2.5)),
            "wr_hi": float(np.percentile(wr, 97.5)),
            "n_plots": int(len(g))}


def main():
    df = load()
    rows = []
    for (model, h), sub in df.groupby(["model", "horizon"]):
        rows.append({"model": model, "horizon": int(h), **boot_ci(sub)})
    lb = pd.DataFrame(rows).sort_values(["horizon", "median_skill"],
                                        ascending=[True, False])
    lb.to_csv(RESULT_DIR / "ml_foundation_leaderboard.csv", index=False)
    print(lb.round(3).to_string(index=False))

    # figure: 24 h leaderboard with CIs
    h0 = 24 if 24 in lb["horizon"].values else lb["horizon"].min()
    d = lb[lb["horizon"] == h0].sort_values("median_skill")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.5))
    y = np.arange(len(d))
    ax.errorbar(d["median_skill"], y,
                xerr=[d["median_skill"] - d["skill_lo"],
                      d["skill_hi"] - d["median_skill"]],
                fmt="o", color="#2c3e50", capsize=3)
    ax.axvline(0, ls="--", color="#c0392b", lw=1, label="seasonal-naive")
    ax.set_yticks(y); ax.set_yticklabels(d["model"])
    ax.set_xlabel("median skill vs seasonal-naive (plot-bootstrap 95% CI)")
    ax.set_title(f"Zero-shot foundation-model leaderboard, {h0} h EC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_ml_foundation_leaderboard.png", dpi=300)
    plt.close(fig)

    (AUDIT_DIR / "ml_foundation_leaderboard.md").write_text(
        "# ML item 6: foundation leaderboard (plot-bootstrap CIs)\n\n"
        + lb.round(3).to_markdown(index=False)
        + "\n\nResampling unit is the plot. CI excluding 0 = robustly beats "
          "seasonal-naive at that horizon.\n")
    print("wrote leaderboard + figure")


if __name__ == "__main__":
    main()
