"""Experiment 20: evidence synthesis - one calibrated posterior for "is the ERW signal real?"

The project produces many separate detection-relevant tests. This capstone fuses
the (approximately) independent lines into a single, honest strength-of-evidence
statement for a true ERW signal in the shallow detection cell, instead of leaving
the reader to eyeball a dozen p-values.

Method (deliberately conservative):
  * Each line yields a one-sided p-value in the ERW-PREDICTED direction.
  * p -> Bayes factor via the Sellke-Bayarri-Berger calibration: the MAXIMUM
    evidence a p-value can carry against the null is BF10_max = 1 / (-e p ln p)
    for p < 1/e, else 1. Using the upper bound means every number below is the
    most generous reading still statistically defensible - if even this is modest,
    the honest verdict is "suggestive, not proven".
  * Posterior(ERW) = f(prior) via posterior odds = prior odds x combined BF10.
  * Two combinations: ALL-LINES (treats lines as independent - optimistic upper
    bound) and INDEPENDENT-MEASUREMENT (collapses the correlated resin-geochemistry
    lines to their single strongest, then multiplies the independent sensor and gas
    lines) - the conservative headline.
  * Plus a sign-concordance binomial test and leave-one-line-out robustness.

Lines (mechanism -> measurement, all from this repo's results):
  L1 feedstock cations   resin R3 15 cm Ca+Mg excess > 0          (computed here)
  L2 anti-acidification  resin R3 15 cm Al excess < 0             (computed here)
  L3 CDR-lag depth shape resin retention index > 0   (Experiment 14 / scripts/84 CSV)
  L4 sensor mobilisation 100 cm dEC/dVWC slope > 0    (Experiment 17 / scripts/87 CSV)
  L5 gas co-benefit      chamber N2O suppression      (Experiment 19 / scripts/89 CSV)
"""

from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from src.config import AUDIT_DIR, ION_CHARGE, ION_MOLAR_MASS, RESULT_DIR
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin
from src.stats.bootstrap import cohens_d_hedges_g

PRIORS = [0.10, 0.25, 0.50]
RESIN_LINES = {"L1", "L2", "L3"}  # share the plot-level randomisation


def bf10_max(p: float) -> float:
    """Sellke-Bayarri-Berger upper bound on BF10 from a one-sided p-value."""
    if not np.isfinite(p) or p <= 0 or p >= 1 / math.e:
        return 1.0
    return 1.0 / (-math.e * p * math.log(p))


def posterior(prior: float, bf: float) -> float:
    odds = (prior / (1 - prior)) * bf
    return odds / (1 + odds)


def exact_cell_p(resin: pd.DataFrame, value_col: str, direction: int) -> float:
    """Exact one-sided permutation p for treated(60) vs control in the R3 15cm cell."""
    cell = resin[(resin["round"] == 3) & (resin["depth_cm"] == 15)]
    cell = cell[cell["treatment"].isin(["60", "control"])]
    vals = cell[value_col].to_numpy()
    is_t = (cell["treatment"] == "60").to_numpy()
    g_obs = cohens_d_hedges_g(vals[is_t], vals[~is_t])["hedges_g"]
    if not np.isfinite(g_obs):
        return np.nan
    n, k = len(vals), int(is_t.sum())
    gs = []
    for combo in combinations(range(n), k):
        m = np.zeros(n, dtype=bool)
        m[list(combo)] = True
        g = cohens_d_hedges_g(vals[m], vals[~m])["hedges_g"]
        if np.isfinite(g):
            gs.append(g)
    gs = np.array(gs)
    return float((gs >= g_obs - 1e-12).mean()) if direction > 0 \
        else float((gs <= g_obs + 1e-12).mean())


def read_csv_p(name: str, mask_col: str, mask_val, p_col: str,
               require_col: str | None = None, require_sign: int = 0) -> float:
    path = RESULT_DIR / name
    if not path.exists():
        return np.nan
    df = pd.read_csv(path)
    if mask_col == "__contains__":
        row = df[df.iloc[:, 0].astype(str).str.contains(mask_val, case=False)]
    else:
        row = df[df[mask_col] == mask_val]
    if row.empty:
        return np.nan
    row = row.iloc[0]
    if require_col is not None and require_sign != 0:
        if np.sign(row[require_col]) != np.sign(require_sign):
            return 1.0  # wrong direction -> no evidence
    return float(row[p_col])


def main() -> None:
    resin = qa_clean(load_resin())
    resin = resin[resin["treatment"].isin(["60", "control"])].copy()
    bc = np.zeros(len(resin))
    for ion in ("ca_ppm", "mg_ppm"):
        bc += (resin[ion].fillna(0) / ION_MOLAR_MASS[ion]) * ION_CHARGE[ion]
    resin["base_molc"] = bc

    p_L1 = exact_cell_p(resin, "base_molc", +1)
    p_L2 = exact_cell_p(resin, "al_ppm", -1)
    p_L3 = read_csv_p("cnew_cdr_lag_significance.csv", "__contains__",
                      "retention", "perm_p_one_sided")
    p_L4 = read_csv_p("cnew_residual_ec_detection.csv", "depth_cm", 100,
                      "perm_p_one_sided", require_col="hedges_g", require_sign=+1)
    p_L5 = read_csv_p("cnew_gas_cobenefit.csv", "gas", "N2O", "exact_perm_p",
                      require_col="hedges_g", require_sign=-1)

    lines = [
        {"id": "L1", "name": "feedstock cations (Ca+Mg excess>0, R3 15cm)", "p": p_L1},
        {"id": "L2", "name": "anti-acidification (Al excess<0, R3 15cm)", "p": p_L2},
        {"id": "L3", "name": "CDR-lag depth shape (retention index>0)", "p": p_L3},
        {"id": "L4", "name": "sensor mobilisation slope (100cm dEC/dVWC>0)", "p": p_L4},
        {"id": "L5", "name": "gas co-benefit (N2O suppression)", "p": p_L5},
    ]
    for L in lines:
        L["bf10_max"] = round(bf10_max(L["p"]), 3)
        L["points_to_erw"] = bool(np.isfinite(L["p"]) and L["p"] < 0.5)
    tab = pd.DataFrame(lines)
    tab.to_csv(RESULT_DIR / "cnew_evidence_synthesis_lines.csv", index=False)

    # ALL-LINES combined (independent assumption: optimistic upper bound)
    bf_all = float(np.prod([L["bf10_max"] for L in lines if np.isfinite(L["p"])]))

    # INDEPENDENT-MEASUREMENT: collapse resin lines to their single strongest
    resin_bf = max((L["bf10_max"] for L in lines if L["id"] in RESIN_LINES), default=1.0)
    other_bf = float(np.prod([L["bf10_max"] for L in lines
                              if L["id"] not in RESIN_LINES and np.isfinite(L["p"])]))
    bf_cons = resin_bf * other_bf

    post_rows = []
    for prior in PRIORS:
        post_rows.append({
            "prior": prior,
            "posterior_all_lines": round(posterior(prior, bf_all), 3),
            "posterior_conservative": round(posterior(prior, bf_cons), 3),
        })
    post = pd.DataFrame(post_rows)
    post.to_csv(RESULT_DIR / "cnew_evidence_synthesis_posterior.csv", index=False)

    # leave-one-out on the all-lines BF (sensitivity)
    loo = []
    for L in lines:
        if not np.isfinite(L["p"]):
            continue
        rest = [x["bf10_max"] for x in lines
                if x["id"] != L["id"] and np.isfinite(x["p"])]
        bf = float(np.prod(rest))
        loo.append({"dropped": L["id"], "bf10": round(bf, 2),
                    "posterior@0.25": round(posterior(0.25, bf), 3)})
    loo = pd.DataFrame(loo)

    # sign concordance: NAIVE (treats 5 lines as independent - optimistic) vs
    # BLOCK-LEVEL (collapses the correlated resin lines L1-L3 into one block, so
    # the independent blocks are {resin geochemistry, sensor, gas}). The block
    # version is the honest headline; the naive 5-line p overstates independence.
    n_lines = int(sum(np.isfinite(L["p"]) for L in lines))
    n_point = int(sum(L["points_to_erw"] for L in lines))
    binom_p = float(stats.binomtest(n_point, n_lines, 0.5,
                                    alternative="greater").pvalue)

    blocks = {
        "resin geochemistry (L1-L3, shared plots/cell)": RESIN_LINES,
        "sensor mobilisation (L4)": {"L4"},
        "gas N2O (L5)": {"L5"},
    }
    block_point = []
    for _, ids in blocks.items():
        bl = [L for L in lines if L["id"] in ids and np.isfinite(L["p"])]
        if not bl:
            continue
        # a block "points to ERW" if a majority of its (finite) lines do
        block_point.append(sum(L["points_to_erw"] for L in bl) > len(bl) / 2)
    n_blocks = len(block_point)
    n_block_point = int(sum(block_point))
    binom_p_blocks = float(stats.binomtest(n_block_point, n_blocks, 0.5,
                                           alternative="greater").pvalue)

    out_lines = [
        "# Experiment 20: Evidence synthesis - one calibrated detection posterior\n",
        "Independent lines fused via Sellke-Bayarri-Berger p->BF upper bounds. "
        "Every BF/posterior is the MOST GENEROUS reading still defensible.\n",
        "## Evidence lines",
        tab[["id", "name", "p", "bf10_max", "points_to_erw"]].to_markdown(index=False),
        "",
        f"- ALL-LINES combined BF10 (independent assumption, upper bound): "
        f"**{bf_all:.2f}**",
        f"- CONSERVATIVE combined BF10 (resin lines collapsed to strongest x "
        f"sensor x gas): **{bf_cons:.2f}**", "",
        "## Posterior P(real ERW signal) vs prior",
        post.to_markdown(index=False), "",
        "## Leave-one-line-out (all-lines BF, posterior at prior 0.25)",
        loo.to_markdown(index=False), "",
        "## Sign concordance",
        f"- BLOCK-LEVEL (headline): {n_block_point}/{n_blocks} independent blocks "
        f"{{resin geochemistry, sensor, gas}} point toward ERW; binomial "
        f"p = **{binom_p_blocks:.3f}**. This respects the strong correlation among "
        "the three resin lines (same R3 shallow cell, same plots), which are "
        "collapsed into one block.",
        f"- NAIVE (5 lines as independent, OPTIMISTIC upper bound): {n_point}/{n_lines} "
        f"point toward ERW, binomial p = {binom_p:.3f} - reported for transparency "
        "only; it overstates independence and should not be quoted as the headline.",
        "",
        "## Reading",
        f"- Headline (conservative, prior 0.25): "
        f"P(real ERW signal) = **{posterior(0.25, bf_cons):.2f}**.",
        "- The conservative combination avoids double-counting the correlated "
        "resin-geochemistry lines (they share the same plot randomisation); it is "
        "the number to quote. The all-lines figure is an upper bound only.",
        "- Even using the most generous calibration, the evidence is best described "
        "as moving the prior MODESTLY upward, not as decisive proof - consistent "
        "with a real-but-detection-limited signal. The synthesis is driven by the "
        "shallow geochemistry (feedstock cations + Al-pH); the sensor and gas lines "
        "add little independent weight because each is individually weak/underpowered.",
        "- Honest limits: BF upper bounds overstate evidence; lines are not fully "
        "independent; p-values come from small-n permutation tests. Treat this as a "
        "transparent aggregation of weak signals, not a confirmatory test.",
    ]
    (AUDIT_DIR / "cnew_evidence_synthesis.md").write_text("\n".join(out_lines))
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
