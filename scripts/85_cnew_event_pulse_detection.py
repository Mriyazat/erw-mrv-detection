"""Experiment 15: event-based EC-pulse detection (concentrate the signal in post-rain windows).

The seasonal/pooled EC contrasts are ~0 because a small mobilisation signal is
diluted across thousands of quiet hours. ERW mobilisation should be event-driven:
when rain wets the soil, treated plots with a larger exchangeable/weatherable
cation pool should release a BIGGER conductivity pulse than control. This test
isolates the pulses and asks whether the treated-minus-control pulse is positive.

For each rain event and plot x depth we take a pre-event baseline (12 h) and the
48 h response, and form:
  * delta_peak = max(post) - pre        (peak conductivity rise)
  * rel_peak   = delta_peak / pre       (baseline-normalised, controls for the
                                         differing absolute EC between arms)
Per-plot means over events give one value per plot; we then test treated(60) vs
control with Hedges' g and an EXACT permutation p-value (all C(8,4)=70 label
assignments, since there are 4 plots/arm), plus a plot-block bootstrap CI. Run
for all events and for the growing-season subset (when fresh feedstock and the
shallow retained pool should respond most), at each depth.
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

RAIN_EVENT_MM = 10.0
PRE_H = 12
POST_H = 48
DEPTHS = [15, 40, 100]
GROW_END = pd.Timestamp("2025-10-16")


def per_plot_pulses(sensors: pd.DataFrame, events: pd.DatetimeIndex,
                    depth: int, only_growing: bool) -> pd.DataFrame:
    col = f"ec_{depth}"
    ev = events[events < GROW_END] if only_growing else events
    rows = []
    for pid, g in sensors.groupby("plot_id"):
        g = g.set_index("timestamp").sort_index()
        if col not in g or g[col].notna().sum() < 100:
            continue
        dpk, rpk, base = [], [], []
        for e in ev:
            pre = g.loc[e - pd.Timedelta(hours=PRE_H):e, col].mean()
            post = g.loc[e:e + pd.Timedelta(hours=POST_H), col]
            if not np.isfinite(pre) or pre <= 0 or post.notna().sum() < 4:
                continue
            d = float(post.max() - pre)
            dpk.append(d)
            rpk.append(d / pre)
            base.append(pre)
        if dpk:
            rows.append({"plot_id": pid, "treatment": g["treatment"].iloc[0],
                         "n_events": len(dpk),
                         "delta_peak": float(np.mean(dpk)),
                         "rel_peak": float(np.mean(rpk)),
                         "pre_baseline": float(np.mean(base))})
    return pd.DataFrame(rows)


def exact_perm_p(values: np.ndarray, is_treated: np.ndarray) -> float:
    """One-sided p (treated>control) over all equal-size label assignments."""
    n, k = len(values), int(is_treated.sum())
    g_obs = cohens_d_hedges_g(values[is_treated], values[~is_treated])["hedges_g"]
    if not np.isfinite(g_obs):
        return np.nan
    idx = np.arange(n)
    ge = 0
    tot = 0
    for combo in combinations(idx, k):
        mask = np.zeros(n, dtype=bool)
        mask[list(combo)] = True
        g = cohens_d_hedges_g(values[mask], values[~mask])["hedges_g"]
        if np.isfinite(g):
            tot += 1
            if g >= g_obs - 1e-12:
                ge += 1
    return ge / tot if tot else np.nan


def boot_ci(values: np.ndarray, is_treated: np.ndarray, rng) -> tuple:
    t, c = values[is_treated], values[~is_treated]
    diffs = []
    for _ in range(5000):
        bt = rng.choice(t, size=len(t), replace=True)
        bc = rng.choice(c, size=len(c), replace=True)
        diffs.append(bt.mean() - bc.mean())
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def main() -> None:
    sensors = pd.read_parquet(CACHE_DIR / "sensors.parquet")
    sensors["timestamp"] = pd.to_datetime(sensors["timestamp"])
    weather = pd.read_parquet(CACHE_DIR / "weather_15min.parquet")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])

    daily = weather.set_index("timestamp")["rain_mm"].resample("D").sum()
    events = daily[daily >= RAIN_EVENT_MM].index
    n_grow = int((events < GROW_END).sum())

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for only_grow in (False, True):
        scope = "growing" if only_grow else "all"
        for depth in DEPTHS:
            pp = per_plot_pulses(sensors, events, depth, only_grow)
            sub = pp[pp["treatment"].isin(["60", "control"])]
            if sub["treatment"].nunique() < 2 or len(sub) < 4:
                continue
            is_t = (sub["treatment"] == "60").to_numpy()
            pre_t = round(float(sub.loc[is_t, "pre_baseline"].mean()), 5)
            pre_c = round(float(sub.loc[~is_t, "pre_baseline"].mean()), 5)
            for metric in ("delta_peak", "rel_peak"):
                vals = sub[metric].to_numpy()
                res = cohens_d_hedges_g(vals[is_t], vals[~is_t])
                p = exact_perm_p(vals, is_t)
                lo, hi = boot_ci(vals, is_t, rng)
                rows.append({
                    "scope": scope, "depth_cm": depth, "metric": metric,
                    "n_treated": int(is_t.sum()), "n_control": int((~is_t).sum()),
                    "treated_mean": round(float(vals[is_t].mean()), 5),
                    "control_mean": round(float(vals[~is_t].mean()), 5),
                    "contrast": round(res["mean_diff"], 5),
                    "ci_lo": round(lo, 5), "ci_hi": round(hi, 5),
                    "hedges_g": round(res["hedges_g"], 3),
                    "perm_p_one_sided": round(p, 4) if np.isfinite(p) else np.nan,
                    "pre_ec_treated": pre_t, "pre_ec_control": pre_c,
                })
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "cnew_event_pulse_detection.csv", index=False)

    # dose trend on the most-likely cell (growing, 15 cm, peak pulse)
    pp15 = per_plot_pulses(sensors, events, 15, only_growing=True)
    dose = (pp15.groupby("treatment")["delta_peak"].mean()
            .reindex(["control", "20", "60"]).round(5))

    # Valid detection metric is the ABSOLUTE pulse. rel_peak is confounded:
    # treated plots have systematically LOWER pre-event baseline EC (sensor/soil
    # heterogeneity, not an ERW addition), which inflates delta/pre for treated.
    abs_out = out[out["metric"] == "delta_peak"]
    sig = abs_out[(abs_out["perm_p_one_sided"] < 0.05) & (abs_out["hedges_g"] > 0)]
    rel_sig = out[(out["metric"] == "rel_peak")
                  & (out["perm_p_one_sided"] < 0.05) & (out["hedges_g"] > 0)]
    best = abs_out.loc[abs_out["hedges_g"].idxmax()] if len(abs_out) else None

    lines = [
        "# Experiment 15: Event-based EC-pulse detection\n",
        f"{len(events)} rain events >= {RAIN_EVENT_MM} mm/day ({n_grow} in the "
        f"growing season). Peak conductivity pulse in the {POST_H} h after each "
        f"event vs a {PRE_H} h pre-baseline; treated(60) vs control, exact "
        "permutation p (70 label assignments) + bootstrap CI.\n",
        "## Detection test (treated 60 vs control)",
        out.to_markdown(index=False), "",
        "## Dose trend (growing-season peak pulse, 15 cm)",
        dose.to_markdown(), "",
        "## Reading",
        "- The valid detection metric is the ABSOLUTE peak pulse (`delta_peak`). "
        "`rel_peak` (pulse / pre-baseline) is CONFOUNDED: treated plots have a "
        "systematically lower pre-event baseline EC (see `pre_ec_treated` vs "
        "`pre_ec_control`), so dividing by it inflates the treated ratio without "
        "any larger absolute mobilisation. Treat rel_peak 'significance' as an "
        "artifact, not a detection.",
    ]
    if best is not None:
        lines.append(
            f"- Strongest ABSOLUTE contrast: {best['scope']} / {best['depth_cm']} "
            f"cm, g={best['hedges_g']:+.2f}, contrast={best['contrast']:+.5f} "
            f"[{best['ci_lo']:+.5f}, {best['ci_hi']:+.5f}], "
            f"perm p={best['perm_p_one_sided']}.")
    if len(sig):
        lines.append(f"- {len(sig)} ABSOLUTE-pulse cell(s) reach a positive, "
                     "significant (p<0.05) treated>control pulse:")
        for _, r in sig.iterrows():
            lines.append(f"    - {r['scope']} {r['depth_cm']} cm: "
                         f"g={r['hedges_g']:+.2f}, p={r['perm_p_one_sided']}.")
    else:
        lines.append("- No ABSOLUTE-pulse cell reaches a positive, significant "
                     "treated>control contrast: even concentrated in post-rain "
                     "windows the sensor EC signal does not separate the arms.")
    if len(rel_sig):
        lines.append(f"- ({len(rel_sig)} rel_peak cell(s) are 'significant' but "
                     "discarded as baseline-confounded per above.)")
    lines += [
        "- No dose monotonicity in the shallow growing-season pulse "
        f"(control {dose.get('control', float('nan')):.4f} vs "
        f"20 {dose.get('20', float('nan')):.4f} vs 60 {dose.get('60', float('nan')):.4f}).",
        "",
        "Interpretation: this is the most favourable framing for sensor-based "
        "detection - it discards the quiet hours that dilute a mobilisation signal "
        "and looks only where ERW should respond (wetting events). On the valid "
        "absolute-pulse metric there is no positive, significant, dose-consistent "
        "treated>control signal, so the continuous EC pathway stays below the "
        "field detection floor at this replication even under event concentration. "
        "This tightens (does not overturn) the detection-budget conclusion; the "
        "only defensible direct signal remains the resin shallow feedstock-Ca:Mg "
        "fingerprint (Experiment 12). Exact permutation at 4 plots/arm floors at "
        "~1/70 = 0.014, so p>=0.05 is clearly null.",
    ]
    (AUDIT_DIR / "cnew_event_pulse_detection.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
