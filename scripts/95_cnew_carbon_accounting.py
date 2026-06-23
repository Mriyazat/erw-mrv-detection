#!/usr/bin/env python3
"""Experiment 24: Committed-vs-realized CDR accounting (tCO2 framing of the lag).

Translates the reactive-transport forward model (scripts/92) into a
carbon-accounting waterfall: gross *committed* alkalinity potential at the
surface, what is retained shallow, what would be exported at 1 m under
steady-state drainage, and what is *realized* (exported) under the MEASURED
water-limited drainage. The committed-realized gap, in tCO2/ha/yr, is the
CDR lag made climate-relevant.

Honest scope: the surface flux is a FIRST-PRINCIPLES potential (config.SNR_MODEL),
NOT a measured CDR rate (the empirical aqueous effect is null). This figure
quantifies the POTENTIAL-vs-REALIZED gap and the within-season export deficit;
it is not a certified removal. Carbonic-acid stoichiometry: 1 mol divalent
base cation released <-> 2 mol charge <-> 2 mol HCO3- <-> 2 mol CO2 (gross,
bicarbonate-stable; downstream re-degassing/precipitation not included).
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(ROOT, "outputs", "figures")
RESDIR = os.path.join(ROOT, "outputs", "results")
AUDDIR = os.path.join(ROOT, "outputs", "audits")
for d in (FIGDIR, RESDIR, AUDDIR):
    os.makedirs(d, exist_ok=True)

MW_CO2 = 44.01  # g/mol
# 1 mol/m2/yr of CO2 == 44.01 g/m2/yr == 0.4401 t/ha/yr  (1 g/m2 = 10 kg/ha)
def mol_co2_to_t_ha(mol_m2_yr):
    return mol_m2_yr * MW_CO2 * 10.0 / 1000.0

# --- forward-model base-cation fluxes at 60 t/ha (mol/m2/yr) ---
# surface production (z=0) from config.SNR_MODEL, reproduced by scripts/92
F_surface = {"Ca": 9.07, "Mg": 0.462}
# retention-attenuated divalent flux at each depth (scripts/92 depth table)
F_depth = {
    15:  {"Ca": 8.045, "Mg": 0.4284},
    40:  {"Ca": 6.587, "Mg": 0.3781},
    100: {"Ca": 4.076, "Mg": 0.2801},
}

def potential_co2(fdict):
    """2 mol CO2 per mol divalent base cation -> tCO2/ha/yr."""
    mol_co2 = 2.0 * (fdict["Ca"] + fdict["Mg"])
    return mol_co2_to_t_ha(mol_co2)

committed_surface = potential_co2(F_surface)
retained_15 = potential_co2(F_depth[15])
steady_100 = potential_co2(F_depth[100])   # would-be export at 1 m, steady drainage

# --- measured water-limited export fraction ---
# growing-season drainage surplus ~ 0 mm (ET0 >> rain every resin window) and
# observed deep (100 cm) treated-control excess is null/negative -> realized
# export within the observation window is statistically indistinguishable from 0.
realized_export_waterlimited = 0.0

rows = [
    ("committed_surface_potential", committed_surface,
     "first-principles gross alkalinity potential at z=0, 60 t/ha"),
    ("retained_to_15cm",            retained_15,
     "potential reaching 15 cm after exchange retention (model)"),
    ("steadystate_export_1m",       steady_100,
     "would-be export at 1 m under year-round (design) drainage"),
    ("realized_export_waterlimited", realized_export_waterlimited,
     "observed export at 1 m under measured ~0 mm season drainage surplus"),
]
gap = committed_surface - realized_export_waterlimited

with open(os.path.join(RESDIR, "cnew_carbon_accounting.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["stage", "tCO2_ha_yr", "note"])
    for k, v, note in rows:
        w.writerow([k, round(v, 3), note])
    w.writerow(["committed_minus_realized_gap", round(gap, 3),
                "the CDR lag, in tCO2/ha/yr"])

# --- waterfall figure ---
labels = ["Committed\n(surface\npotential)", "Retained\nto 15 cm",
          "Steady-state\nexport at 1 m", "Realized export\n(water-limited,\nobserved)"]
vals = [committed_surface, retained_15, steady_100, realized_export_waterlimited]
colors = ["#3b6ea5", "#5a9bd4", "#bcbddc", "#d98c5f"]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.annotate("", xy=(3, realized_export_waterlimited + 0.15), xytext=(0, committed_surface),
            arrowprops=dict(arrowstyle="-|>", color="#b03030", lw=1.6,
                            connectionstyle="arc3,rad=-0.25"))
ax.text(1.5, committed_surface * 0.62,
        f"CDR lag\n= {gap:.1f} tCO$_2$ ha$^{{-1}}$ yr$^{{-1}}$\nnot yet realized",
        ha="center", va="center", fontsize=10, color="#b03030", fontweight="bold")
ax.set_ylabel("Potential CDR  (t CO$_2$ ha$^{-1}$ yr$^{-1}$, 60 t ha$^{-1}$)")
ax.set_title("Committed vs. realized CDR: the cation-storage lag in carbon terms")
ax.set_ylim(0, committed_surface * 1.18)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = os.path.join(FIGDIR, "fig_carbon_accounting.png")
fig.savefig(out, dpi=150)
print("wrote", out)

with open(os.path.join(AUDDIR, "cnew_carbon_accounting.md"), "w") as fh:
    fh.write("# Experiment 24: Committed-vs-realized CDR accounting\n\n")
    fh.write("First-principles potential (config.SNR_MODEL), carbonic-acid "
             "stoichiometry (2 CO2 per divalent base cation). NOT a measured "
             "CDR rate; the empirical aqueous effect is null.\n\n")
    fh.write("| stage | tCO2/ha/yr |\n|---|---|\n")
    for k, v, _ in rows:
        fh.write(f"| {k} | {v:.2f} |\n")
    fh.write(f"| **committed - realized gap (the lag)** | **{gap:.2f}** |\n\n")
    fh.write("At 60 t/ha the model commits ~{:.1f} tCO2/ha/yr of gross "
             "alkalinity potential; exchange retention holds ~{:.1f} at 15 cm; "
             "steady drainage would export ~{:.1f} at 1 m, but the measured "
             "~0 mm growing-season surplus realizes ~0 within the window. The "
             "committed-realized gap (~{:.1f} tCO2/ha/yr) is the cation-storage "
             "CDR lag in climate-relevant units.\n".format(
                 committed_surface, retained_15, steady_100, gap))
print("done")
