#!/usr/bin/env python3
"""ML-paper: skill-vs-horizon decay figure for the full forecaster panel.

Reads the per-horizon leaderboard produced by 115_ml_foundation_leaderboard.py
and draws median skill (vs seasonal-naive) as a function of forecast horizon for
every model, separating the three zero-shot transformer FOUNDATION models from
the four purpose-trained DEEP models. The single visual makes the paper's
thesis legible: short-horizon structure is real and foundation models lead it,
but by 72 h every model has collapsed to or below persistence.

Local (Mac, ../.venv). Run after 115.
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

from src.config import RESULT_DIR, FIGURE_DIR

# zero-shot transformer foundation models vs purpose-trained deep models
FOUNDATION = {
    "timesfm":      ("TimesFM-2.5",  "#b2182b"),
    "chronos":      ("Chronos-2",    "#d6604d"),
    "chronos_bolt": ("Chronos-Bolt", "#f4a582"),
}
DEEP = {
    "patchtst":     ("PatchTST",     "#2166ac"),
    "nhits":        ("N-HiTS",       "#4393c3"),
    "tft":          ("TFT",          "#92c5de"),
    "itransformer": ("iTransformer", "#5aae61"),
}


def main():
    p = RESULT_DIR / "ml_foundation_leaderboard.csv"
    if not p.exists():
        raise SystemExit("run scripts/115_ml_foundation_leaderboard.py first")
    lb = pd.read_csv(p)
    horizons = sorted(lb["horizon"].unique())

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(horizons))  # categorical spacing: 6/24/72 are not linear

    def _plot(group, ls, lw, marker, z):
        for key, (label, color) in group.items():
            d = lb[lb["model"] == key].set_index("horizon").reindex(horizons)
            if d["median_skill"].isna().all():
                continue
            ax.plot(x, d["median_skill"].values, ls=ls, lw=lw, marker=marker,
                    ms=7, color=color, label=label, zorder=z)

    _plot(DEEP, "--", 1.6, "s", 2)
    _plot(FOUNDATION, "-", 2.6, "o", 3)

    ax.axhline(0, ls=":", color="#444", lw=1.4, zorder=1)
    ax.text(x[-1], 0.012, "seasonal-naive (persistence)", ha="right", va="bottom",
            fontsize=8.5, color="#444")
    # shade the sub-persistence region
    ax.axhspan(ax.get_ylim()[0], 0, color="#f0f0f0", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} h" for h in horizons])
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("median skill vs seasonal-naive")
    ax.set_title("Forecast skill decays to persistence by 72 h\n"
                 "zero-shot foundation models (solid) lead trained deep models "
                 "(dashed) at every horizon")
    ax.set_xlim(-0.2, len(horizons) - 0.8)

    # two-column legend separating the two model families
    leg = ax.legend(loc="upper right", ncol=2, fontsize=8.5, frameon=True,
                    title="foundation (solid)   |   deep (dashed)",
                    title_fontsize=8.5)
    leg.get_frame().set_alpha(0.9)

    fig.tight_layout()
    out = FIGURE_DIR / "fig_ml_horizon_decay.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
