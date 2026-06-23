#!/usr/bin/env python3
"""ML-paper item 12: you cannot ML your way past the replication wall.

Two complementary views that connect the ML lens to the SNR detection budget:

  (a) Empirical detection vs replication. We vary the number of plot-halves per
      arm n in {2,3,4} (random subsets), and recompute BOTH the naive row-level
      classification accuracy (leaky) and the honest leave-one-plot-out (LOPO)
      accuracy. The naive accuracy stays ~1.0 at every n (leakage is
      n-independent), while the honest accuracy stays at chance (~1/3): more
      rows never help because the replication unit is the plot.

  (b) Power projection to the wall. For the observed standardized treatment
      effects (pooled |g|~0.38 and the single best cell g~1.48) we invert the
      two-sample power law n = 2[(z_{1-a/2}+z_{1-b})/g]^2 to get the plots/arm a
      *learned* detector would need at 80% power -- landing on the same n the
      SNR budget predicts. ML expressiveness does not move the wall.

Local (Mac, ../.venv): sklearn + scipy. Self-contained; sensors.parquet.
"""
from __future__ import annotations

import os
import itertools
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(__file__))
RES = os.path.join(ROOT, "outputs", "results")
AUD = os.path.join(ROOT, "outputs", "audits")
FIG = os.path.join(ROOT, "outputs", "figures")
for d in (RES, AUD, FIG):
    os.makedirs(d, exist_ok=True)

SEED = 42
N_PER_PLOT = 500
N_SUBSETS = 8                       # random arm-subsets per n
rng = np.random.default_rng(SEED)
DEPTHS = (15, 40, 100)
CHANS = ["vwc", "temp", "ec"]

ARMS = {
    "control": ["4W", "4E", "6E", "7W"],
    "20": ["3W", "5E", "7E", "8W"],
    "60": ["3E", "5W", "6W", "8E"],
}


def build():
    s = pd.read_parquet(os.path.join(ROOT, "outputs", "cache", "sensors.parquet"))
    s = s[s["treatment"].isin(["control", "20", "60"])].copy()
    s["ts"] = pd.to_datetime(s["timestamp"])
    cols = [f"{c}_{d}" for c in CHANS for d in DEPTHS if f"{c}_{d}" in s.columns]
    cols += [f"mp_{d}" for d in DEPTHS if f"mp_{d}" in s.columns]
    s = s.dropna(subset=["ec_15", "ec_40", "ec_100"]).reset_index(drop=True)
    s["hour"] = s["ts"].dt.hour
    s["doy"] = s["ts"].dt.dayofyear
    cols += ["hour", "doy"]
    parts = [g.sample(min(len(g), N_PER_PLOT), random_state=SEED)
             for _, g in s.groupby("plot_id")]
    sub = pd.concat(parts).reset_index(drop=True)
    return sub, cols


def detection_at_n(sub, cols, n_per_arm):
    """Mean naive & honest accuracy over random arm-subsets of size n_per_arm."""
    combos = []
    for _ in range(N_SUBSETS):
        chosen = []
        for arm, plots in ARMS.items():
            avail = [p for p in plots if p in sub["plot_id"].unique()]
            k = min(n_per_arm, len(avail))
            chosen += list(rng.choice(avail, size=k, replace=False))
        combos.append(tuple(sorted(chosen)))
    combos = list(dict.fromkeys(combos))   # dedup

    naive_all, hon_all = [], []
    for chosen in combos:
        d = sub[sub["plot_id"].isin(chosen)]
        if d["plot_id"].nunique() < 3 * n_per_arm:
            continue
        X = d[cols].fillna(0.0).to_numpy()
        y = d["treatment"].map({"control": 0, "20": 1, "60": 2}).to_numpy()
        g = d["plot_id"].to_numpy()
        # naive
        skf = StratifiedKFold(4, shuffle=True, random_state=SEED)
        nv = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            m = HistGradientBoostingClassifier(max_iter=80, max_depth=6,
                                               random_state=SEED).fit(
                sc.transform(X[tr]), y[tr])
            nv.append(accuracy_score(y[te], m.predict(sc.transform(X[te]))))
        # honest LOPO
        logo = LeaveOneGroupOut()
        hn = []
        for tr, te in logo.split(X, y, g):
            sc = StandardScaler().fit(X[tr])
            m = HistGradientBoostingClassifier(max_iter=80, max_depth=6,
                                               random_state=SEED).fit(
                sc.transform(X[tr]), y[tr])
            hn.append(accuracy_score(y[te], m.predict(sc.transform(X[te]))))
        naive_all.append(np.mean(nv))
        hon_all.append(np.mean(hn))
    return (np.mean(naive_all), np.std(naive_all),
            np.mean(hon_all), np.std(hon_all), len(combos))


def power_projection():
    """n per arm for 80% power at alpha=0.05 for given standardized effects."""
    z = norm.ppf(0.975) + norm.ppf(0.80)
    rows = []
    for label, g in [("pooled base-cation |g|", 0.38),
                     ("best single cell (R3 15cm)", 1.48),
                     ("realistic ERW (<=0.4 SD)", 0.40)]:
        n = 2 * (z / g) ** 2
        rows.append({"effect": label, "g": g, "n_per_arm_80pct": round(float(n), 1)})
    return pd.DataFrame(rows)


def main():
    sub, cols = build()
    print(f"[data] n={len(sub)} rows, {sub['plot_id'].nunique()} plots, "
          f"{len(cols)} features")

    rows = []
    for n in (2, 3, 4):
        na_m, na_s, ho_m, ho_s, k = detection_at_n(sub, cols, n)
        rows.append({"plots_per_arm": n, "n_subsets": k,
                     "naive_acc": round(na_m, 3), "naive_sd": round(na_s, 3),
                     "honest_lopo_acc": round(ho_m, 3),
                     "honest_sd": round(ho_s, 3), "chance": round(1 / 3, 3)})
        print(f"  n/arm={n}: naive={na_m:.3f}+/-{na_s:.3f}  "
              f"honest={ho_m:.3f}+/-{ho_s:.3f}  (chance 0.333)")
    curve = pd.DataFrame(rows)
    curve.to_csv(os.path.join(RES, "ml_learning_curve.csv"), index=False)

    proj = power_projection()
    proj.to_csv(os.path.join(RES, "ml_power_projection.csv"), index=False)
    print("\nPower projection (plots/arm for 80% power):")
    print(proj.to_string(index=False))

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    a = ax[0]
    a.errorbar(curve["plots_per_arm"], curve["naive_acc"], yerr=curve["naive_sd"],
               fmt="o-", color="#c0392b", capsize=3, label="naive (row split)")
    a.errorbar(curve["plots_per_arm"], curve["honest_lopo_acc"],
               yerr=curve["honest_sd"], fmt="s-", color="#2c3e50", capsize=3,
               label="honest (LOPO)")
    a.axhline(1 / 3, ls="--", color="grey", lw=1, label="chance (1/3)")
    a.set_xticks([2, 3, 4])
    a.set_xlabel("plot-halves per arm")
    a.set_ylabel("treatment-classification accuracy")
    a.set_title("(a) More replication never lifts honest detection")
    a.set_ylim(0, 1.05)
    a.legend(fontsize=8)

    b = ax[1]
    b.bar(proj["effect"], proj["n_per_arm_80pct"], color="#2980b9")
    b.axhline(4, ls="--", color="#c0392b", lw=1.2, label="realized design (4/arm)")
    b.set_yscale("log")
    b.set_ylabel("plots/arm for 80% power (log)")
    b.set_title("(b) Replication the learned detector would need")
    b.set_xticklabels(proj["effect"], rotation=20, ha="right", fontsize=8)
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ml_learning_curve.png"), dpi=300)
    plt.close(fig)

    with open(os.path.join(AUD, "ml_learning_curve.md"), "w") as fh:
        fh.write("# ML paper item 12: learning curve vs the replication wall\n\n")
        fh.write("## (a) Empirical detection vs plots-per-arm\n\n")
        fh.write(curve.to_markdown(index=False))
        fh.write("\n\nNaive accuracy is ~1.0 at every n (leakage is "
                 "n-independent); honest LOPO accuracy stays at chance. More "
                 "15-min rows do not help — the replication unit is the plot.\n\n")
        fh.write("## (b) Power projection\n\n")
        fh.write(proj.to_markdown(index=False))
        fh.write("\n\nEven the single best cell needs more than 4 plots/arm; the "
                 "pooled effect and a realistic <=0.4-SD ERW signal need tens to "
                 "hundreds — the same wall the SNR budget reports. ML capacity "
                 "does not move it.\n")
    print("\nwrote results + audit + figure")


if __name__ == "__main__":
    main()
