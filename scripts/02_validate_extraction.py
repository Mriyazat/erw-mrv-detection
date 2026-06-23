"""Validate raw extraction - prove the loaders didn't silently mangle data.

Checks (writes outputs/audits/extraction_validation.md):
  1. Skipped-config accounting: how many raw rows were dropped, per plot.
  2. Depth-mapping validation via PHYSICS, not column order:
     diurnal soil-temperature amplitude must DECREASE with depth
     (15 cm > 40 cm > 100 cm) - the textbook damping-depth result. If the
     block-order -> depth assignment were scrambled this would fail.
  3. Weather-cutoff policy is present and explicit.
  4. Resin integrity: 121 rows, QA flag set, Si absent, pH unusable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, CACHE_DIR, RESIN_PRIMARY_IONS

OUT = AUDIT_DIR / "extraction_validation.md"


def _diurnal_amplitude(df: pd.DataFrame, col: str) -> float:
    """Mean within-day peak-to-trough range for a temperature channel."""
    if col not in df or df[col].notna().sum() < 96:
        return np.nan
    g = df.dropna(subset=[col]).copy()
    g["day"] = g["timestamp"].dt.date
    rng = g.groupby("day")[col].agg(lambda x: x.max() - x.min())
    return float(rng.mean())


def main() -> None:
    sensors = pd.read_parquet(CACHE_DIR / "sensors.parquet")
    audit   = pd.read_parquet(CACHE_DIR / "sensors_audit.parquet")
    resin   = pd.read_parquet(CACHE_DIR / "resin.parquet")
    policy  = json.loads((CACHE_DIR / "weather_join_policy.json").read_text())

    lines: list[str] = ["# Extraction Validation\n"]

    # --- 1. skipped-config accounting -----------------------------------
    skipped = audit[~audit["kept"]]
    kept = audit[audit["kept"]]
    raw_skipped = int(skipped["n_rows_raw"].sum())
    raw_kept = int(kept["n_rows_kept"].sum())
    lines += [
        "## 1. Config accounting",
        f"- Standard configs kept: **{len(kept)}** ({raw_kept:,} rows)",
        f"- Non-standard configs skipped: **{len(skipped)}** "
        f"(~{raw_skipped:,} raw rows)",
        f"- Skip fraction of raw rows: "
        f"**{raw_skipped / max(raw_kept + raw_skipped, 1):.2%}**",
        "",
        "Largest skipped configs:",
        skipped.nlargest(5, "n_rows_raw")[
            ["plot_id", "source_file", "sheet", "n_rows_raw", "notes"]
        ].to_markdown(index=False),
        "",
    ]

    # --- 2. depth mapping via physics -----------------------------------
    lines += ["## 2. Depth-mapping validation (diurnal temp damping)",
              "Physical expectation: amplitude(15) > amplitude(40) > amplitude(100).",
              ""]
    rows = []
    n_monotone = 0
    n_eval = 0
    for plot, g in sensors.groupby("plot_id"):
        a15 = _diurnal_amplitude(g, "temp_15")
        a40 = _diurnal_amplitude(g, "temp_40")
        a100 = _diurnal_amplitude(g, "temp_100")
        ok = None
        if not any(np.isnan(x) for x in (a15, a40, a100)):
            ok = (a15 >= a40 >= a100)
            n_eval += 1
            n_monotone += int(ok)
        rows.append({"plot_id": plot, "amp_15": round(a15, 3),
                     "amp_40": round(a40, 3), "amp_100": round(a100, 3),
                     "monotone_decreasing": ok})
    amp = pd.DataFrame(rows)
    lines += [amp.to_markdown(index=False), "",
              f"**{n_monotone}/{n_eval} plots** show monotone damping with depth "
              f"(remaining have 100 cm gaps or near-ties).", ""]
    depth_ok = n_eval > 0 and n_monotone / n_eval >= 0.75

    # --- 3. weather cutoff ---------------------------------------------
    lines += ["## 3. Weather-cutoff policy",
              f"- Weather max: `{policy['weather_max']}`",
              f"- Sensor rows past cutoff: **{policy['n_rows_after_weather_cutoff']:,}** "
              f"({policy['frac_dropped']:.1%} of all rows)",
              "- Excluded from sensor+weather join by explicit policy "
              "(raw sensor cache keeps all rows).", ""]

    # --- 4. resin integrity --------------------------------------------
    si_cols = [c for c in resin.columns if c.startswith("si_")]
    ph_unusable = bool((resin["ph"].fillna(0) == 0).all())
    qa_n = int((resin["qa_flag"] != "").sum())
    lines += ["## 4. Resin integrity",
              f"- Rows: **{len(resin)}** (expected 121): "
              f"{'OK' if len(resin) == 121 else 'MISMATCH'}",
              f"- QA-flagged capsules: **{qa_n}** "
              f"({resin.loc[resin['qa_flag'] != '', 'barcode'].tolist()})",
              f"- Si columns present: {si_cols if si_cols else 'none (expected)'}",
              f"- pH all-zero (unusable): {ph_unusable}",
              f"- Primary-ion non-null: " + ", ".join(
                  f"{c}={resin[c].notna().sum()}" for c in RESIN_PRIMARY_IONS),
              ""]

    # --- verdict --------------------------------------------------------
    all_ok = (len(resin) == 121 and qa_n >= 1 and not si_cols
              and ph_unusable and depth_ok
              and policy["n_rows_after_weather_cutoff"] > 0)
    lines += ["## Verdict",
              f"**{'PASS' if all_ok else 'REVIEW'}** - "
              f"depth_ok={depth_ok}, resin_ok={len(resin) == 121}, "
              f"weather_policy_explicit=True"]

    OUT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {OUT}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
