"""Publication figures for the manuscript (paper/erw_aqueous_mrv.tex).

Builds three polished figures from existing result CSVs for the strongest
results that previously had no figure:
  * fig_feedstock_fingerprint.png  - excess Ca:Mg vs feedstock window & background (Experiment 12)
  * fig_multielement_fingerprint.png - per-element detection-cell Hedges g (Experiment 16)
  * fig_evidence_synthesis.png      - per-line Bayes factors + posterior-vs-prior (Experiment 20)

Run after the analysis scripts that produce the source CSVs (82, 86, 90).
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

from src.config import FIGURE_DIR, RESULT_DIR, SNR_MODEL

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 200, "savefig.dpi": 200, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

C_POS = "#1b7837"   # supports ERW (correct direction)
C_NEG = "#762a83"   # pH/anti-acidification (expected negative)
C_FLAG = "#d95f02"  # red flag / non-specific
C_NULL = "#999999"  # inert / null


def feedstock_ratios() -> tuple[float, float]:
    mw_w, mw_d = SNR_MODEL["mw_wollastonite_g_mol"], SNR_MODEL["mw_diopside_g_mol"]
    fw, fd = SNR_MODEL["f_rate_wollastonite"], SNR_MODEL["f_rate_diopside"]
    mol_w, mol_d = 50.0 / mw_w, 50.0 / mw_d
    stock = (mol_w + mol_d) / mol_d
    rate = (mol_w * fw + mol_d * fd) / (mol_d * fd)
    return stock, rate


def fig_feedstock() -> None:
    df = pd.read_csv(RESULT_DIR / "cnew_feedstock_fingerprint.csv")
    stock, rate = feedstock_ratios()
    cell = df[(df["round"] == 3) & (df["depth_cm"] == 15)].iloc[0]
    bg = float(cell["control_bg_ca_mg"])
    ratio = float(cell["excess_ca_mg_molar"])
    lo, hi = float(cell["ratio_lo"]), float(cell["ratio_hi"])

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.axvspan(stock, rate, color="#a6dba0", alpha=0.45,
               label=f"feedstock window [{stock:.1f}, {rate:.0f}]")
    ax.axvline(bg, color=C_FLAG, ls="--", lw=1.6,
               label=f"control background ({bg:.2f})")
    ax.errorbar([ratio], [0], xerr=[[ratio - lo], [hi - ratio]], fmt="o",
                color=C_POS, ms=9, capsize=5, lw=2,
                label=f"R3 15 cm excess ({ratio:.1f} [{lo:.1f}, {hi:.1f}])")
    ax.set_xscale("log")
    ax.set_xlim(2, 60)
    ax.set_yticks([])
    ax.set_xlabel(r"Excess Ca:Mg molar ratio (treated $-$ control), log scale")
    ax.set_title("Feedstock dissolution fingerprint of the shallow CDR-lag signal")
    ax.set_xticks([2, 3, 5, 8, 10, 20, 40])
    ax.set_xticklabels(["2", "3", "5", "8", "10", "20", "40"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=1, fontsize=8.5)
    fig.subplots_adjust(bottom=0.42, top=0.86)
    fig.savefig(FIGURE_DIR / "fig_feedstock_fingerprint.png", bbox_inches="tight")
    plt.close(fig)


def fig_multielement() -> None:
    df = pd.read_csv(RESULT_DIR / "cnew_multielement_fingerprint.csv",
                     keep_default_na=False)
    for c in ("expected_sign", "g_15cm_pooled", "ci_lo", "ci_hi", "g_R3_15cm_cell"):
        df[c] = pd.to_numeric(df[c])
    order = ["CA", "MG", "AL", "MN", "FE", "K", "NA", "NO3_N", "NH4_N"]
    df = df.set_index("element").loc[order].reset_index()
    labels = df["element"].str.replace("_N", "-N", regex=False)

    def color(r):
        if r["element"] in ("CA", "MG"):
            return C_POS
        if r["element"] == "AL":
            return C_NEG
        if r["element"] in ("MN", "FE"):
            return "#c2a5cf"
        if r["element"] == "K":
            return C_FLAG
        return C_NULL
    colors = [color(r) for _, r in df.iterrows()]

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    y = np.arange(len(df))[::-1]
    ax.barh(y, df["g_R3_15cm_cell"], color=colors, height=0.66)
    # expected-direction markers
    for yi, (_, r) in zip(y, df.iterrows()):
        if r["expected_sign"] != 0:
            ax.plot(0.18 * np.sign(r["expected_sign"]), yi, marker="|",
                    ms=10, color="k", mew=1.5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Treated(60)$-$control Hedges' $g$ (R3, 15 cm detection cell)")
    ax.set_title("Multi-element geochemical fingerprint")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (C_POS, C_NEG, "#c2a5cf", C_FLAG, C_NULL)]
    ax.legend(handles, ["feedstock cation (exp +)", "Al: anti-acidification (exp -)",
                        "Mn/Fe: redox-confounded", "K: non-specific flag",
                        "inert control (exp 0)"],
              fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, columnspacing=1.2)
    ax.text(0.015, 0.02, "ticks = ERW-predicted direction", transform=ax.transAxes,
            fontsize=7.5, style="italic", color="#444")
    fig.subplots_adjust(bottom=0.30, top=0.92)
    fig.savefig(FIGURE_DIR / "fig_multielement_fingerprint.png", bbox_inches="tight")
    plt.close(fig)


def fig_evidence() -> None:
    lines = pd.read_csv(RESULT_DIR / "cnew_evidence_synthesis_lines.csv")
    post = pd.read_csv(RESULT_DIR / "cnew_evidence_synthesis_posterior.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    # left: per-line Bayes factor (max), log scale
    short = {"L1": "L1 feedstock cations", "L2": "L2 anti-acidification (Al)",
             "L3": "L3 CDR-lag depth shape", "L4": "L4 sensor mobilisation",
             "L5": "L5 gas N2O suppression"}
    y = np.arange(len(lines))[::-1]
    ax1.barh(y, lines["bf10_max"], color="#2166ac", height=0.6)
    ax1.axvline(1, color="k", ls="--", lw=1, label="BF = 1 (no evidence)")
    ax1.set_yticks(y)
    ax1.set_yticklabels([short[i] for i in lines["id"]])
    ax1.set_xlabel(r"per-line BF$_{10}$ (SBB upper bound)")
    ax1.set_title("Independent evidence lines")
    ax1.legend(fontsize=8, loc="lower right")
    for yi, v in zip(y, lines["bf10_max"]):
        ax1.text(v + 0.05, yi, f"{v:.2f}", va="center", fontsize=8)
    ax1.set_xlim(0, max(lines["bf10_max"]) * 1.25)

    # right: posterior vs prior
    pri = np.linspace(0.02, 0.8, 200)
    # recover combined BFs implied by the saved posterior@0.25
    def bf_from(col):
        p0, q0 = 0.25, post.loc[post["prior"] == 0.25, col].iloc[0]
        return (q0 / (1 - q0)) / (p0 / (1 - p0))
    for col, c, lab in [("posterior_conservative", "#1b7837", "conservative"),
                        ("posterior_all_lines", "#999999", "all-lines (upper bound)")]:
        bf = bf_from(col)
        curve = (pri / (1 - pri) * bf) / (1 + pri / (1 - pri) * bf)
        ax2.plot(pri, curve, color=c, lw=2.2, label=f"{lab} (BF$\\approx${bf:.1f})")
    ax2.plot([0, 0.8], [0, 0.8], color="k", ls=":", lw=0.9, label="no update")
    qc = post.loc[post["prior"] == 0.25, "posterior_conservative"].iloc[0]
    ax2.plot([0.25], [qc], "o", color="#1b7837", ms=8)
    ax2.annotate(f"0.25 $\\rightarrow$ {qc:.2f}", (0.25, qc),
                 textcoords="offset points", xytext=(8, -12), fontsize=8.5)
    ax2.set_xlabel("prior P(real ERW signal)")
    ax2.set_ylabel("posterior P(real ERW signal)")
    ax2.set_title("Calibrated detection posterior")
    ax2.set_xlim(0, 0.8)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8, loc="lower right")
    fig.suptitle("Five-line evidence synthesis (all sign-concordant; "
                 "block-level binomial $p=0.13$)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig_evidence_synthesis.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_feedstock()
    fig_multielement()
    fig_evidence()
    print("Wrote: fig_feedstock_fingerprint.png, fig_multielement_fingerprint.png, "
          "fig_evidence_synthesis.png")


if __name__ == "__main__":
    main()
