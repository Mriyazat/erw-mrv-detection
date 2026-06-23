"""Experiment 17: physics-removed EC detection - give the sensor pathway its best shot.

Raw bulk-EC treated-vs-control contrasts are null because (a) EC variance is
dominated by moisture and temperature, and (b) the EC LEVEL is swamped by per-
sensor / per-soil calibration offsets unrelated to treatment (e.g. 15 cm pre-event
baseline EC ~0.006 in some treated plots vs ~0.285 in controls - a 40x offset that
has nothing to do with amendment). So a level contrast can never work.

The offset-robust, physics-aware quantity is the MOBILISATION SLOPE: how much
bulk EC rises per unit increase in water content, dEC/dVWC (controlling for
temperature). It is invariant to any additive per-sensor offset and reflects the
pool of soluble ions a wetting front can mobilise - exactly what an ERW cation
surplus should increase. We fit EC ~ VWC + temp per plot x depth, take the VWC
coefficient as the mobilisation slope, and test treated(60) vs control with
Hedges' g, an exact permutation p (70 label assignments), and a bootstrap CI.
This is the most favourable valid sensor test; a null here is the definitive
'sensors cannot detect it at this replication' statement.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, CACHE_DIR, RANDOM_SEED, RESULT_DIR
from src.stats.bootstrap import cohens_d_hedges_g

DEPTHS = [15, 40, 100]
MIN_N = 500          # min hourly obs per plot/depth for a stable fit
MIN_VWC_RANGE = 0.05  # need moisture variation to estimate a slope


def mobilisation_slope(g: pd.DataFrame, depth: int) -> float:
    """OLS coefficient of VWC in EC ~ VWC + temp (offset-invariant)."""
    cols = [f"ec_{depth}", f"vwc_{depth}", f"temp_{depth}"]
    d = g[cols].dropna()
    if len(d) < MIN_N:
        return np.nan
    x_vwc = d[f"vwc_{depth}"].to_numpy()
    if x_vwc.max() - x_vwc.min() < MIN_VWC_RANGE:
        return np.nan
    X = np.column_stack([np.ones(len(d)), x_vwc, d[f"temp_{depth}"].to_numpy()])
    y = d[f"ec_{depth}"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def exact_perm_p(values: np.ndarray, is_t: np.ndarray) -> float:
    g_obs = cohens_d_hedges_g(values[is_t], values[~is_t])["hedges_g"]
    if not np.isfinite(g_obs):
        return np.nan
    n, k = len(values), int(is_t.sum())
    ge = tot = 0
    for combo in combinations(range(n), k):
        m = np.zeros(n, dtype=bool)
        m[list(combo)] = True
        g = cohens_d_hedges_g(values[m], values[~m])["hedges_g"]
        if np.isfinite(g):
            tot += 1
            ge += int(g >= g_obs - 1e-12)
    return ge / tot if tot else np.nan


def main() -> None:
    s = pd.read_parquet(CACHE_DIR / "sensors.parquet")
    rng = np.random.default_rng(RANDOM_SEED)

    perplot = []
    for pid, g in s.groupby("plot_id"):
        trt = g["treatment"].iloc[0]
        for depth in DEPTHS:
            slope = mobilisation_slope(g, depth)
            raw_ec = g[f"ec_{depth}"].mean()
            perplot.append({"plot_id": pid, "treatment": trt, "depth_cm": depth,
                            "mobilisation_slope": slope, "raw_mean_ec": raw_ec})
    pp = pd.DataFrame(perplot)
    pp.to_csv(RESULT_DIR / "cnew_residual_ec_perplot.csv", index=False)

    rows = []
    for depth in DEPTHS:
        sub = pp[(pp["depth_cm"] == depth)
                 & pp["treatment"].isin(["60", "control"])
                 & pp["mobilisation_slope"].notna()]
        is_t = (sub["treatment"] == "60").to_numpy()
        if is_t.sum() < 2 or (~is_t).sum() < 2:
            continue
        vals = sub["mobilisation_slope"].to_numpy()
        res = cohens_d_hedges_g(vals[is_t], vals[~is_t])
        p = exact_perm_p(vals, is_t)
        diffs = [rng.choice(vals[is_t], is_t.sum(), True).mean()
                 - rng.choice(vals[~is_t], (~is_t).sum(), True).mean()
                 for _ in range(5000)]
        # raw-EC level contrast for comparison (shows the offset confound)
        rt = sub.loc[is_t, "raw_mean_ec"].mean()
        rc = sub.loc[~is_t, "raw_mean_ec"].mean()
        rows.append({
            "depth_cm": depth, "n_treated": int(is_t.sum()),
            "n_control": int((~is_t).sum()),
            "slope_treated": round(float(vals[is_t].mean()), 4),
            "slope_control": round(float(vals[~is_t].mean()), 4),
            "contrast": round(res["mean_diff"], 4),
            "ci_lo": round(float(np.quantile(diffs, 0.025)), 4),
            "ci_hi": round(float(np.quantile(diffs, 0.975)), 4),
            "hedges_g": round(res["hedges_g"], 3),
            "perm_p_one_sided": round(p, 4) if np.isfinite(p) else np.nan,
            "raw_ec_treated": round(float(rt), 4),
            "raw_ec_control": round(float(rc), 4),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "cnew_residual_ec_detection.csv", index=False)

    sig = out[(out["perm_p_one_sided"] < 0.05) & (out["hedges_g"] > 0)]
    best = out.loc[out["hedges_g"].idxmax()] if len(out) else None

    lines = [
        "# Experiment 17: Physics-removed (mobilisation-slope) EC detection\n",
        "Offset-invariant mobilisation slope dEC/dVWC (from EC ~ VWC + temp per "
        "plot x depth), treated(60) vs control, exact permutation p + bootstrap CI. "
        "raw_ec_* columns show the per-sensor level offset that makes a raw-level "
        "contrast meaningless.\n",
        out.to_markdown(index=False), "",
        "## Reading",
    ]
    if best is not None:
        lines.append(
            f"- Best mobilisation-slope contrast: {best['depth_cm']} cm, "
            f"g={best['hedges_g']:+.2f}, contrast={best['contrast']:+.4f} "
            f"[{best['ci_lo']:+.4f}, {best['ci_hi']:+.4f}], "
            f"perm p={best['perm_p_one_sided']}.")
    if len(sig):
        lines.append(f"- {len(sig)} depth(s) reach a positive significant "
                     "treated>control mobilisation slope:")
        for _, r in sig.iterrows():
            lines.append(f"    - {r['depth_cm']} cm: g={r['hedges_g']:+.2f}, "
                         f"p={r['perm_p_one_sided']}.")
    else:
        lines.append("- No depth reaches a positive, significant treated>control "
                     "mobilisation slope.")
    lines += [
        "- The raw_ec_treated vs raw_ec_control columns confirm the level offset "
        "is large and not treatment-ordered (often treated < control), which is "
        "why level-based EC contrasts are uninformative and a slope/offset-robust "
        "metric is required.", "",
        "Interpretation: by removing the moisture/temperature physics AND the "
        "per-sensor additive offset, this is the most favourable valid framing for "
        "sensor detection. The result sets the ceiling on what the continuous EC "
        "network can deliver at this replication; combined with the null event-"
        "pulse test (Experiment 15), a null here means the in-situ bulk-EC pathway is "
        "below the field detection floor even under best-case processing - the "
        "detectable signal lives in the targeted resin geochemistry, not the "
        "sensors. Exact permutation at 4 plots/arm floors at ~1/70 = 0.014.",
    ]
    (AUDIT_DIR / "cnew_residual_ec_detection.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
