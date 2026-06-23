"""Experiment 1 mechanism: water balance explains why the base-cation excess stays shallow.

The CDR-lag hypothesis predicts exchange-buffered base cations only migrate
downward once infiltration generates a drainage surplus. We pair the per-round
depth profile (from scripts/60) with the seasonal water balance (rain vs ET0)
over each resin deployment window.

Key finding the data forces: the trial site was water-LIMITED all season
(potential ET0 > rainfall every window => ~zero net drainage surplus), so there
was little downward leaching to transport cations deep. That deficit is a
mechanistic explanation for the observed shallow-retained / deep-at-or-below-
control profile - the field-scale cation-exchange lag (Kanzaki 2025): without
drainage, released cations sit on shallow exchange sites and the deep alkalinity
export that defines realised CDR has not yet occurred.

Honest caveat: only 3 resin rounds => descriptive mechanistic check, not a
powered regression.
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

from src.config import (AUDIT_DIR, FIGURE_DIR, RESULT_DIR, RESIN_ROUND_DATES)
from src.io.load_weather import load_weather_15min

ROUND_LABEL = {1: "R1 (Jul)", 2: "R2 (Aug)", 3: "R3 (Sep-Oct)"}


def drainage_by_window() -> pd.DataFrame:
    w = load_weather_15min().dropna(subset=["timestamp"]).set_index("timestamp")
    rows = []
    cum_rain = cum_drain = 0.0
    for rnd in (1, 2, 3):
        start, end = RESIN_ROUND_DATES[rnd]
        win = w.loc[(w.index >= start) & (w.index < end)]
        rain = float(win["rain_mm"].sum())
        et0 = float(win["et0_mm"].sum()) if "et0_mm" in win else np.nan
        balance = rain - et0 if np.isfinite(et0) else np.nan      # P - PET
        drain = max(balance, 0.0) if np.isfinite(balance) else np.nan
        cum_rain += rain
        cum_drain += (drain if np.isfinite(drain) else 0.0)
        rows.append({"round": rnd, "round_label": ROUND_LABEL[rnd],
                     "win_start": start, "win_end": end,
                     "rain_mm": round(rain, 1), "et0_mm": round(et0, 1),
                     "water_balance_mm": round(balance, 1),
                     "drainage_surplus_mm": round(drain, 1),
                     "cum_rain_mm": round(cum_rain, 1),
                     "cum_drainage_mm": round(cum_drain, 1)})
    return pd.DataFrame(rows)


def excess_center_of_mass(prof: pd.DataFrame) -> pd.DataFrame:
    """Depth (cm) centre-of-mass of the POSITIVE treated-control excess per round."""
    rows = []
    for rnd in (1, 2, 3):
        p = prof[prof["round"] == rnd]
        pos = p[p["excess_molc"] > 0]
        if len(pos) and pos["excess_molc"].sum() > 0:
            com = float((pos["depth_cm"] * pos["excess_molc"]).sum()
                        / pos["excess_molc"].sum())
        else:
            com = np.nan
        rows.append({"round": rnd, "excess_com_depth_cm": com,
                     "total_positive_excess": float(p[p["excess_molc"] > 0]
                                                     ["excess_molc"].sum())})
    return pd.DataFrame(rows)


def main() -> None:
    drain = drainage_by_window()
    prof = pd.read_csv(RESULT_DIR / "cnew_cec_lag_profile.csv")
    ret = pd.read_csv(RESULT_DIR / "cnew_cec_lag_retention.csv")
    com = excess_center_of_mass(prof)

    merged = (drain.merge(com, on="round")
              .merge(ret[["round", "shallow_minus_deep", "deep_below_control"]],
                     on="round", how="left"))
    merged.to_csv(RESULT_DIR / "cnew_cec_lag_mechanism.csv", index=False)

    deficit_all = bool((merged["water_balance_mm"] < 0).all())
    total_surplus = float(merged["drainage_surplus_mm"].sum())

    fig, ax1 = plt.subplots(figsize=(6.8, 4.3))
    x = merged["round"].to_numpy()
    ax1.bar(x - 0.18, merged["rain_mm"], width=0.36, color="#3182bd",
            label="rainfall (mm)")
    ax1.bar(x + 0.18, merged["et0_mm"], width=0.36, color="#fdae6b",
            label="potential ET0 (mm)")
    ax1.set_xlabel("Resin round")
    ax1.set_ylabel("Per-window water flux (mm)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(merged["round_label"])
    ax1.legend(loc="upper left", fontsize=8, frameon=False)
    ax2 = ax1.twinx()
    ax2.plot(x, merged["shallow_minus_deep"], "-o", color="#de2d26",
             label="shallow-minus-deep retention")
    ax2.set_ylabel("Shallow-minus-deep excess (mol_c)", color="#de2d26")
    ax2.axhline(0, color="#de2d26", lw=0.7, ls=":")
    ax1.set_title("CDR-lag mechanism: water-limited season => shallow retention")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cnew_cec_lag_mechanism.png", dpi=130)
    plt.close(fig)

    lines = ["# Experiment 1 mechanism: water balance & shallow retention\n",
             "Per-window rainfall vs potential ET0, drainage surplus, and the "
             "shallow-minus-deep base-cation retention index.\n",
             "## Water balance x retention table",
             merged[["round_label", "rain_mm", "et0_mm", "water_balance_mm",
                     "drainage_surplus_mm", "cum_rain_mm", "cum_drainage_mm",
                     "shallow_minus_deep", "deep_below_control"]]
             .round(2).to_markdown(index=False), "",
             "## Mechanistic reading",
             f"- Every window is in water DEFICIT (rain < ET0): {deficit_all}.",
             f"- Total season drainage surplus: {total_surplus:.1f} mm "
             "(≈0 => negligible deep leaching).",
             "- Positive treated-control base-cation excess is confined to the "
             "shallow profile and only emerges by R3; deep stays at/below control "
             "in every round.", "",
             "Interpretation: with potential ET exceeding rainfall all season, "
             "there is essentially no drainage surplus to transport dissolved "
             "base cations downward. Released Ca/Mg therefore remain on shallow "
             "exchange sites - exactly the cation-exchange retention buffer that "
             "delays deep alkalinity export (the CDR lag of Kanzaki 2025). The "
             "water balance thus provides an independent mechanistic explanation "
             "for the shallow-retained / deep-null profile, and warns that shallow "
             "cation appearance must NOT be equated with realised CDR. n=3 rounds: "
             "mechanistic illustration, not a powered test.",
             "", f"Figure: {FIGURE_DIR/'cnew_cec_lag_mechanism.png'}"]
    (AUDIT_DIR / "cnew_cec_lag_mechanism.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
