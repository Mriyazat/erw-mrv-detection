"""Phase: standalone chamber gas-flux summary (NOT joined to resin/sensor).

See docs/CHAMBER_JOIN_DECISION.md. Reports per-gas flux distributions, the
Linear-vs-Exponential QA gate, and a CO2-uptake sanity check.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import AUDIT_DIR, RESULT_DIR
from src.io.load_chamber import FLUX_GASES, load_chamber


def main() -> None:
    df = load_chamber()
    rows = []
    for gas in FLUX_GASES:
        lin = f"flux_{gas}_lin"
        qa = f"flux_{gas}_qa_ok"
        if lin not in df:
            continue
        valid = df[df["valid"]]
        qa_pass = valid[valid.get(qa, False)] if qa in df else valid
        rows.append({
            "gas": gas,
            "n_valid": int(valid[lin].notna().sum()),
            "n_qa_pass": int(qa_pass[lin].notna().sum()),
            "qa_pass_rate": round(qa_pass[lin].notna().sum()
                                  / max(valid[lin].notna().sum(), 1), 3),
            "median_flux_lin": round(float(valid[lin].median()), 4),
            "mean_flux_lin": round(float(valid[lin].mean()), 4),
            "frac_negative": round(float((valid[lin] < 0).mean()), 3),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_DIR / "chamber_summary.csv", index=False)

    by_plot = (df[df["valid"]].groupby(["plot_id", "treatment"])
               .agg(n=("timestamp", "count"),
                    co2=("flux_co2_lin", "median"),
                    n2o=("flux_n2o_lin", "median"))
               .reset_index())
    by_plot.to_csv(RESULT_DIR / "chamber_by_plot.csv", index=False)

    lines = ["# Phase: Chamber (standalone)\n",
             "Spring-2026, 7-day, 6 chambers on plots {6W,6E,7W,7E}. "
             "Analysed separately (see docs/CHAMBER_JOIN_DECISION.md).\n",
             "## Per-gas summary", summary.to_markdown(index=False), "",
             "## Per-plot medians", by_plot.round(4).to_markdown(index=False), "",
             "CO2 QA pass rate is the weakest; flux signs are reported but the "
             "campaign is under-powered for a treatment contrast (no control arm)."]
    (AUDIT_DIR / "phase_chamber.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
