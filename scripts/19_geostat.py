"""Phase: spatial structure of the resin signal (Moran's I + variogram).

Uses per-plot GPS centroids (plot metadata) and round-pooled resin Ca to test
whether residual ERW signal has spatial autocorrelation (a confound for a
randomised-plot design and a target for spatial-block CV).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, CACHE_DIR, RESULT_DIR
from src.analysis.effects import qa_clean
from src.io.load_plot_metadata import per_plot_centroid
from src.io.load_resin import load_resin


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def morans_i(values, coords):
    n = len(values)
    z = values - values.mean()
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = haversine_m(coords[i, 0], coords[i, 1],
                                coords[j, 0], coords[j, 1])
                W[i, j] = 1.0 / d if d > 0 else 0.0
    S0 = W.sum()
    num = (W * np.outer(z, z)).sum()
    den = (z ** 2).sum()
    return (n / S0) * (num / den) if S0 > 0 and den > 0 else np.nan


def empirical_variogram(values, coords, n_bins=6):
    n = len(values)
    dists, semis = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_m(coords[i, 0], coords[i, 1],
                            coords[j, 0], coords[j, 1])
            dists.append(d)
            semis.append(0.5 * (values[i] - values[j]) ** 2)
    dists, semis = np.array(dists), np.array(semis)
    edges = np.linspace(0, dists.max() + 1e-6, n_bins + 1)
    rows = []
    for b in range(n_bins):
        m = (dists >= edges[b]) & (dists < edges[b + 1])
        if m.sum():
            rows.append({"bin_center_m": round((edges[b] + edges[b + 1]) / 2, 1),
                         "n_pairs": int(m.sum()),
                         "semivariance": round(float(semis[m].mean()), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    meta = pd.read_parquet(CACHE_DIR / "plot_metadata.parquet")
    cent = per_plot_centroid(meta)  # one centroid per plot-half (plot_id like '3E')
    resin = qa_clean(load_resin())
    # group resin Ca by PLOT-HALF to match per-half GPS centroids (avoids the
    # artificial autocorrelation from assigning one value to both W/E halves).
    ca = (resin.groupby("plot_half")["ca_ppm"].mean().reset_index()
          .rename(columns={"ca_ppm": "ca_mean"}))
    m = cent.merge(ca, left_on="plot_id", right_on="plot_half", how="inner"
                   ).dropna(subset=["latitude", "longitude", "ca_mean"])
    coords = m[["latitude", "longitude"]].to_numpy()
    vals = m["ca_mean"].to_numpy()

    mi = morans_i(vals, coords)
    vg = empirical_variogram(vals, coords)
    pd.DataFrame([{"morans_I": round(mi, 4), "n_plots": len(m)}]).to_csv(
        RESULT_DIR / "geostat_morans_i.csv", index=False)
    vg.to_csv(RESULT_DIR / "geostat_variogram.csv", index=False)

    lines = ["# Phase: Geostatistics\n",
             f"Per-plot GPS centroids x round-pooled resin Ca ({len(m)} plots).\n",
             f"## Moran's I = **{mi:.4f}**",
             "(near 0 => little spatial autocorrelation; positive => clustering)\n",
             "## Empirical variogram", vg.to_markdown(index=False), "",
             ("Moran's I indicates appreciable spatial clustering of the resin "
              "signal - motivating spatial-block / variogram-buffered holdout."
              if abs(mi) >= 0.2 else
              "Weak spatial autocorrelation supports plot-level CV without a "
              "strong distance-decay confound.")]
    (AUDIT_DIR / "phase_geostat.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
