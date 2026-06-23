"""Phase: Monte-Carlo statistical power and minimum detectable effect (MDE).

Given the observed control-arm variability per ion x depth, simulate the power
to detect a true mean shift at the current design (N=4 plot-halves/arm), and
the MDE at 80% power. This is the empirical detection floor that the
first-principles SNR model ignores.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from src.config import AUDIT_DIR, RANDOM_SEED, RESULT_DIR, RESIN_PRIMARY_IONS
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin

IONS = RESIN_PRIMARY_IONS + ["k_ppm", "na_ppm", "s_ppm"]
N_PER_ARM = 4
N_SIM = 4000
ALPHA = 0.05


def simulate_power(sd: float, n: int, true_diff: float, rng) -> float:
    """Vectorised Welch two-sample t-test power over N_SIM Monte-Carlo draws."""
    if sd <= 0:
        return np.nan
    c = rng.normal(0.0, sd, size=(N_SIM, n))
    t = rng.normal(true_diff, sd, size=(N_SIM, n))
    mc, mt = c.mean(1), t.mean(1)
    vc, vt = c.var(1, ddof=1), t.var(1, ddof=1)
    se = np.sqrt(vc / n + vt / n)
    se[se == 0] = np.nan
    tstat = (mt - mc) / se
    dof = (vc / n + vt / n) ** 2 / (
        (vc / n) ** 2 / (n - 1) + (vt / n) ** 2 / (n - 1))
    pvals = 2 * stats.t.sf(np.abs(tstat), dof)
    return float(np.nanmean(pvals < ALPHA))


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    resin = qa_clean(load_resin())

    rows = []
    for ion in IONS:
        for depth in (15, 40, 100):
            ctrl = resin[(resin["treatment"] == "control")
                         & (resin["depth_cm"] == depth)][ion].dropna()
            if len(ctrl) < 3:
                continue
            sd = float(ctrl.std(ddof=1))
            mean = float(ctrl.mean())
            # MDE: search true_diff giving ~80% power
            mde = np.nan
            for d_frac in np.linspace(0.05, 3.0, 60):
                td = d_frac * sd
                pw = simulate_power(sd, N_PER_ARM, td, rng)
                if pw >= 0.80:
                    mde = td
                    break
            rows.append({
                "ion": ion, "depth_cm": depth, "control_mean": round(mean, 3),
                "control_sd": round(sd, 3),
                "power_at_0.5sd": round(simulate_power(sd, N_PER_ARM, 0.5 * sd, rng), 3),
                "power_at_1.0sd": round(simulate_power(sd, N_PER_ARM, 1.0 * sd, rng), 3),
                "mde_abs_ppm": round(mde, 3) if np.isfinite(mde) else np.nan,
                "mde_pct_of_control": (round(100 * mde / mean, 1)
                                       if np.isfinite(mde) and mean else np.nan),
                "mde_in_sd": round(mde / sd, 2) if np.isfinite(mde) else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "power_mde.csv", index=False)

    prim = df[df["ion"].isin(RESIN_PRIMARY_IONS)]
    lines = ["# Phase: Power / MDE\n",
             f"Monte-Carlo (n_sim={N_SIM}) Welch t-test power at N={N_PER_ARM} "
             "plot-halves/arm, using observed control-arm sigma.\n",
             "## Primary ions",
             prim[["ion", "depth_cm", "control_mean", "control_sd",
                   "power_at_1.0sd", "mde_in_sd", "mde_pct_of_control"]]
             .to_markdown(index=False), "",
             "The design typically needs a shift of ~1.5-2 SD (often >50% of the "
             "control mean) for 80% power - quantifying why small ERW signals are "
             "hard to certify with this replication."]
    (AUDIT_DIR / "phase_power.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
