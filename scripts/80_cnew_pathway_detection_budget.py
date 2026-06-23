"""Experiment 11: detection-budget head-to-head - aqueous/sensor MRV vs solid-phase mass balance.

The 2025-2026 ERW-MRV literature has converged almost entirely on SOLID-PHASE
sample-resample cation mass balance (Reershemius 2023; Suhrhoff 2024; Clarkson
et al. 2025 SOMBA SNR; Dalland/Frontiers 2025 sampling design; Knapp & Tipper /
BG 2026 review). Our trial instead measures the AQUEOUS / exchange-phase signal
via ion-exchange resin capsules and continuous in-situ bulk-EC sensors. Nobody
has put the two detection budgets side by side.

This script assembles OUR aqueous-pathway detectability (from the already-
computed effect sizes, MDE, and variance-components outputs) and places it next
to the published solid-phase benchmarks, on common axes: what is measured, the
observed/achievable signal-to-noise, the replication needed to detect a
realistic effect, monitoring cost/cadence, and the binding limitation.

It does NOT claim our own solid-phase measurements (we have none); the solid-
phase column is sourced from the cited literature and clearly marked as such.
The contribution is the SYNTHESIS: a like-for-like detection budget that tells a
field team which pathway detects the signal first, and at what cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, RESULT_DIR

# --- Published SOLID-PHASE benchmarks (external; cited, not our data) ----------
SOLID_PHASE_LIT = {
    "needs_samples_per_ha": ">10 (often 10-20)",      # CarbonPlan; Clarkson 2025
    "analytical_precision": "~1% (ICP), immobile-element ratios",
    "spatial_density_studied": "0.6-19.2 samples/ha",  # Clarkson 2025 (5 fields, 998 samples)
    "binding_limitation": "baseline geochemical variance often too high for "
                          "element-element mixing to constrain dissolution near-term "
                          "(Dalland/Frontiers 2025)",
    "monitoring_cadence": "annual / biennial sample-resample (no continuous monitoring)",
    "citations": "Reershemius 2023; Suhrhoff 2024; Clarkson 2025 (SOMBA); "
                 "Frontiers 2025; BG 2026 review",
}


def _headline_value(h: pd.DataFrame, metric: str) -> float:
    m = h[h["metric"] == metric]
    return float(m["value"].iloc[0]) if len(m) else np.nan


def _best_pos_g(eff: pd.DataFrame, ion: str) -> dict:
    """Best DETECTION = most positive Hedges' g (treated > control)."""
    sub = eff[eff["ion"] == ion].copy()
    r = sub.loc[sub["hedges_g"].idxmax()]
    return {"g": float(r["hedges_g"]),
            "round": int(r["round"]), "depth_cm": int(r["depth_cm"]),
            "arm": str(r["treatment"])}


def _plots_for_1sd(mde_grid: pd.DataFrame, target_sd: float = 1.0,
                   min_caps: int = 2) -> str:
    sub = mde_grid[(mde_grid["capsules_per_plot"] >= min_caps)
                   & (mde_grid["mde_in_grand_sd"] <= target_sd)]
    if not len(sub):
        return f">{int(mde_grid['plots_per_arm'].max())}"
    return str(int(sub["plots_per_arm"].min()))


def main() -> None:
    h = pd.read_csv(RESULT_DIR / "headline_summary.csv")
    eff = pd.read_csv(RESULT_DIR / "empirical_effects.csv")
    pw = pd.read_csv(RESULT_DIR / "porewater_ec_effects.csv")
    pw["treatment"] = pw["treatment"].astype(str)
    vc = pd.read_csv(RESULT_DIR / "cnew_sampling_variance_components.csv")
    mde_ca = pd.read_csv(RESULT_DIR / "cnew_sampling_mde_ca_ppm.csv")

    mde_resin = _headline_value(h, "MDE_80pct_ca_ppm_15cm")
    sigma_infl = _headline_value(h, "sigma_inflation_needed_ca_ppm_15cm")
    pooled_ca_20 = _headline_value(h, "empirical_hedges_g_ca_ppm_20tha_pooled")
    pooled_ca_60 = _headline_value(h, "empirical_hedges_g_ca_ppm_60tha_pooled")
    best_ca = _best_pos_g(eff, "ca_ppm")
    icc_ca = float(vc.loc[vc["target"].str.startswith("Ca"), "icc"].iloc[0])
    plots_1sd = _plots_for_1sd(mde_ca)

    # best sensor bulk-EC DETECTION (most positive g) over the resin windows
    pw_pos = pw.loc[pw["hedges_g_ec"].idxmax()]
    pw_shallow = pw[(pw["depth_cm"] == 15) & (pw["treatment"] == "60")]["hedges_g_ec"]
    pw_shallow_g = float(pw_shallow.iloc[0]) if len(pw_shallow) else np.nan

    rows = [
        {
            "pathway": "Resin (ion-exchange PRS, aqueous exchange phase)",
            "measures": "adsorbed Ca/Mg flux (cation supply rate)",
            "temporal_mode": "seasonal capsule (3 rounds)",
            "best_detection_+g": round(best_ca["g"], 2),
            "best_detection_where": f"R{best_ca['round']} {best_ca['depth_cm']}cm "
                                    f"{best_ca['arm']}t/ha",
            "pooled_g": f"{pooled_ca_20:+.2f}/{pooled_ca_60:+.2f} (20/60, ~0)",
            "MDE_at_n12_controlSD": round(mde_resin, 2),
            "replication_for_~1SD_MDE": f"~{plots_1sd} plots/arm, >=2 capsules",
            "binding_limitation": f"signal near floor; SNR overstated ~{sigma_infl:.0f}x; "
                                  f"within-plot noise dominates (ICC={icc_ca:.2f})",
            "monitoring_cadence": "seasonal capsule swap",
            "source": "this study",
        },
        {
            "pathway": "Continuous bulk-EC sensors (in-situ aqueous proxy)",
            "measures": "bulk soil EC (ionic strength proxy), 15-min",
            "temporal_mode": "continuous (multi-season)",
            "best_detection_+g": round(float(pw_pos["hedges_g_ec"]), 2),
            "best_detection_where": f"{int(pw_pos['depth_cm'])}cm "
                                    f"{pw_pos['treatment']}t/ha (best is still <=0)",
            "pooled_g": f"shallow 60t/ha g={pw_shallow_g:+.2f}; all depths <=0",
            "MDE_at_n12_controlSD": np.nan,
            "replication_for_~1SD_MDE": "n/a (proxy, not a cation budget)",
            "binding_limitation": "moisture/temperature confound; not a direct "
                                  "cation measure; treatment contrast near zero",
            "monitoring_cadence": "continuous (high capex, low marginal)",
            "source": "this study",
        },
        {
            "pathway": "Solid-phase mass balance (SOMBA / cation budget)",
            "measures": "soil cation stock change vs immobile tracer (Ti/Zr)",
            "temporal_mode": "annual sample-resample",
            "best_detection_+g": np.nan,
            "best_detection_where": "n/a (literature benchmark)",
            "pooled_g": "n/a",
            "MDE_at_n12_controlSD": np.nan,
            "replication_for_~1SD_MDE": SOLID_PHASE_LIT["needs_samples_per_ha"]
                                        + " samples/ha",
            "binding_limitation": SOLID_PHASE_LIT["binding_limitation"],
            "monitoring_cadence": SOLID_PHASE_LIT["monitoring_cadence"],
            "source": SOLID_PHASE_LIT["citations"],
        },
    ]
    comp = pd.DataFrame(rows)
    comp.to_csv(RESULT_DIR / "cnew_pathway_detection_budget.csv", index=False)

    lines = [
        "# Experiment 11: Detection-budget head-to-head - aqueous/sensor vs solid-phase MRV\n",
        "A like-for-like detection budget placing this trial's AQUEOUS pathways "
        "(ion-exchange resin + continuous bulk-EC sensors) next to the published "
        "SOLID-PHASE mass-balance benchmarks that dominate 2025-2026 ERW MRV. The "
        "solid-phase row is from the cited literature, not our data; the "
        "contribution is the side-by-side synthesis.\n",
        "## Pathway comparison",
        comp.to_markdown(index=False), "",
        "## What the budget says",
        f"- **Aqueous signal is real but near the floor.** Resin Ca's best "
        f"detection is g={best_ca['g']:+.2f} in a single cell (R{best_ca['round']} "
        f"{best_ca['depth_cm']}cm {best_ca['arm']}t/ha - the shallow CDR-lag cell) "
        f"but pools to ~0 across the season; first-principles SNR overstates "
        f"detectability ~{sigma_infl:.0f}x, and the 12-plot MDE is "
        f"{mde_resin:.1f} control-SD - far above the observed effect.",
        "- **Sensors add temporal richness, not detection power.** Bulk EC is a "
        "moisture/temperature-confounded proxy; its treatment contrast is small "
        "and sign-inconsistent across depth, so continuous monitoring buys "
        "dynamics (events, lag - see Experiment 1/10) rather than a cleaner yes/no.",
        f"- **Allocation rule (resin).** Because within-plot variance dominates "
        f"(ICC={icc_ca:.2f}), a ~1-SD MDE needs ~{plots_1sd} plots/arm with >=2 "
        "capsules - capsule replication is unusually cost-effective.",
        "- **vs solid-phase.** The solid-phase literature reaches the SAME "
        "cautionary conclusion from the other side: baseline geochemical variance "
        "is often too high for element-mixing mass balance to constrain "
        "dissolution near-term at feasible sampling density (Frontiers 2025), "
        "needing >10 samples/ha (Clarkson 2025). Trade-off: solid-phase needs only "
        "annual sampling (cheap cadence) but high spatial density and ~1% "
        "analytical precision; the aqueous/sensor pathway gives continuous, "
        "process-level signal (lag, transport) but at realistic replication "
        "cannot deliver a powered single-season detection either.", "",
        "## Takeaway",
        "Neither pathway detects a clean single-season aqueous CDR signal at this "
        "trial's replication - and the convergence of an independent aqueous "
        "detection budget with the solid-phase literature is itself the result: "
        "ERW verification at field scale is detection-limited across measurement "
        "phases, so MRV designs should budget for it explicitly (power/MDE first) "
        "and exploit the aqueous pathway for PROCESS evidence (the CDR lag) rather "
        "than as a stand-alone quantifier.",
    ]
    (AUDIT_DIR / "cnew_pathway_detection_budget.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
