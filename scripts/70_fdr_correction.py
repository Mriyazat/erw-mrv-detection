"""Multiple-comparison correction across the resin effect tests.

We run many hypothesis tests (every ion x round x depth x dose arm), so raw
p-values overstate significance. This phase:
  1. Computes a Welch t-test p-value for each treated-vs-control cell.
  2. Applies Benjamini-Hochberg FDR within analyte family (primary / negative-
     control / secondary) and pooled across all cells.
  3. Re-checks the dose-response OLS p-values the same way.

Output makes explicit how many "significant" cells survive FDR - the honest
reviewer-proof version of the effect tables.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", message=".*catastrophic cancellation.*")

from src.config import (AUDIT_DIR, RESIN_NEGATIVE_CONTROLS, RESIN_PRIMARY_IONS,
                        RESIN_SECONDARY_IONS, RESULT_DIR)
from src.analysis.effects import qa_clean
from src.io.load_resin import load_resin
from src.stats.multitest import add_bh

FAMILY = ({i: "primary" for i in RESIN_PRIMARY_IONS}
          | {i: "negative_control" for i in RESIN_NEGATIVE_CONTROLS}
          | {i: "secondary" for i in RESIN_SECONDARY_IONS})
IONS = list(FAMILY)
ALPHA = 0.05


def welch_cells(resin: pd.DataFrame) -> pd.DataFrame:
    df = qa_clean(resin)
    rows = []
    for ion in IONS:
        if ion not in df.columns:
            continue
        for rnd in sorted(df["round"].unique()):
            for depth in sorted(df["depth_cm"].unique()):
                sub = df[(df["round"] == rnd) & (df["depth_cm"] == depth)]
                ctrl = sub.loc[sub["treatment"] == "control", ion].dropna().values
                for arm in ("20", "60"):
                    trt = sub.loc[sub["treatment"] == arm, ion].dropna().values
                    if len(trt) < 2 or len(ctrl) < 2:
                        continue
                    t, p = stats.ttest_ind(trt, ctrl, equal_var=False)
                    rows.append({
                        "ion": ion, "family": FAMILY[ion], "round": rnd,
                        "depth_cm": depth, "treatment": arm,
                        "mean_diff": float(trt.mean() - ctrl.mean()),
                        "t_stat": float(t), "p_value": float(p),
                        "n_t": len(trt), "n_c": len(ctrl),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    resin = load_resin()

    cells = welch_cells(resin)
    cells = add_bh(cells, "p_value", group="family", qcol="q_within_family")
    cells = add_bh(cells, "p_value", group=None, qcol="q_pooled")
    cells.to_csv(RESULT_DIR / "fdr_effect_cells.csv", index=False)

    dose = pd.read_csv(RESULT_DIR / "empirical_dose_response.csv")
    dose["family"] = dose["ion"].map(FAMILY)
    dose = add_bh(dose, "p_value", group="family", qcol="q_within_family")
    dose = add_bh(dose, "p_value", group=None, qcol="q_pooled")
    dose.to_csv(RESULT_DIR / "fdr_dose_response.csv", index=False)

    def tally(df: pd.DataFrame, label: str) -> pd.DataFrame:
        g = (df.groupby("family")
             .agg(n_tests=("p_value", "size"),
                  raw_sig=("p_value", lambda s: int((s < ALPHA).sum())),
                  fdr_sig_family=("q_within_family",
                                  lambda s: int((s < ALPHA).sum())),
                  fdr_sig_pooled=("q_pooled",
                                  lambda s: int((s < ALPHA).sum())))
             .reset_index())
        g.insert(0, "test", label)
        return g

    summary = pd.concat([tally(cells, "effect_welch"),
                         tally(dose, "dose_response")], ignore_index=True)
    summary.to_csv(RESULT_DIR / "fdr_summary.csv", index=False)

    survivors = cells[cells["q_within_family"] < ALPHA].sort_values("q_within_family")

    lines = ["# Phase: Multiple-comparison (FDR) correction\n",
             f"Benjamini-Hochberg at q<{ALPHA}. Family-wise correction treats "
             "primary / negative-control / secondary ions as separate hypothesis "
             "classes; pooled corrects across every cell.\n",
             "## How many 'significant' cells survive FDR",
             summary.to_markdown(index=False), "",
             "## Effect cells surviving family-wise FDR (q<0.05)",
             (survivors[["ion", "family", "round", "depth_cm", "treatment",
                         "mean_diff", "p_value", "q_within_family"]]
              .round(4).to_markdown(index=False) if len(survivors)
              else "_none survive FDR_"), "",
             "Interpretation: raw p<0.05 counts are inflated by the number of "
             "cells tested; the FDR-surviving set is the defensible signal. A "
             "small survivor set against many tests is itself the headline - it "
             "quantifies how weak/localised the aqueous resin signal is and "
             "directly motivates the detection-budget contribution."]
    (AUDIT_DIR / "phase_fdr.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
