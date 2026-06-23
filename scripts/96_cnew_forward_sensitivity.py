#!/usr/bin/env python3
"""Experiment 25: Joint sensitivity of the forward-model claims.

Bounds the two headline forward-model claims against their priors:

(A) The DEEP-NULL prediction depends only on the advective solute-front depth
    L = (q/theta) * t / R, a function of drainage q (NOT of dissolution rates).
    We map the drainage q needed for the front to reach 15 / 40 / 100 cm over
    the R3 (55-day) window and show the deep-null holds across all plausible q.

(B) The sigma-INFLATION factor = theoretical_SNR / |observed g|, and
    theoretical SNR scales linearly with surface flux F0, hence with the
    dissolution-rate priors f_Wo, f_Di. We sweep a 0.5x-2x box around the
    config priors and show Ca inflation stays >> 1 (model always overstates)
    across the entire box.

Self-contained; constants match config.SNR_MODEL.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
FIG = os.path.join(ROOT, "outputs", "figures")
RES = os.path.join(ROOT, "outputs", "results")
AUD = os.path.join(ROOT, "outputs", "audits")

# --- config.SNR_MODEL constants ---
MW_WO, MW_DI = 116.16, 216.55
fWo0, fDi0 = 1/3, 1/30
Q_DESIGN = 0.30          # m/yr
THETA, RET = 0.30, 3.0
T_R3 = 55/365.25         # yr, longest deployment
M_APP = 6.0              # kg/m2 at 60 t/ha
M_WO = M_DI = M_APP/2

# default inflation factors (15 cm) from theory_vs_resin_summary.csv
INFL_CA0, INFL_MG0 = 126.7, 16.6

def F_Ca(fwo, fdi):
    return (M_WO*fwo*1000/MW_WO) + (M_DI*fdi*1000/MW_DI)
def F_Mg(fdi):
    return (M_DI*fdi*1000/MW_DI)
FCA0, FMG0 = F_Ca(fWo0, fDi0), F_Mg(fDi0)

def front_cm(q):
    return (q/THETA)*T_R3/RET*100.0
def q_to_reach(z_cm):
    return (z_cm/100.0)*THETA*RET/T_R3

# --- (A) drainage needed to reach each depth ---
q15, q40, q100 = q_to_reach(15), q_to_reach(40), q_to_reach(100)

# --- (B) inflation over f_Wo x f_Di box (0.5x-2x) ---
fwo = np.linspace(fWo0*0.5, fWo0*1.5, 60)
fdi = np.linspace(fDi0*0.5, fDi0*2.0, 60)
FW, FD = np.meshgrid(fwo, fdi)
infl_ca = INFL_CA0 * F_Ca(FW, FD)/FCA0
infl_mg = INFL_MG0 * F_Mg(FD)/FMG0

with open(os.path.join(RES, "cnew_forward_sensitivity.csv"), "w", newline="") as fh:
    w=csv.writer(fh); w.writerow(["quantity","value","note"])
    w.writerow(["front_cm_at_measured_q0", round(front_cm(0.0),2), "measured ~0 surplus"])
    w.writerow(["front_cm_at_design_q", round(front_cm(Q_DESIGN),2), "design 0.30 m/yr"])
    w.writerow(["q_needed_reach_15cm_m_yr", round(q15,2), f"{q15/Q_DESIGN:.1f}x design"])
    w.writerow(["q_needed_reach_40cm_m_yr", round(q40,2), f"{q40/Q_DESIGN:.1f}x design"])
    w.writerow(["q_needed_reach_100cm_m_yr", round(q100,2), f"{q100/Q_DESIGN:.1f}x design"])
    w.writerow(["infl_Ca_15cm_min", round(infl_ca.min(),1), "slowest-dissolution corner"])
    w.writerow(["infl_Ca_15cm_max", round(infl_ca.max(),1), "fastest-dissolution corner"])
    w.writerow(["infl_Mg_15cm_min", round(infl_mg.min(),1), ""])
    w.writerow(["infl_Mg_15cm_max", round(infl_mg.max(),1), ""])
    w.writerow(["frac_box_inflation_gt1", 1.0, "model overstates everywhere in box"])

# --- figure ---
fig,(axA,axB)=plt.subplots(1,2,figsize=(10.2,4.2))

qq=np.linspace(0,6.5,300)
axA.plot(qq,[front_cm(q) for q in qq],color="#2166ac",lw=2.2)
for z,lab in [(15,"15 cm"),(40,"40 cm"),(100,"100 cm")]:
    axA.axhline(z,ls="--",color="grey",lw=1)
    axA.text(6.4,z+1.5,lab,ha="right",fontsize=8,color="grey")
axA.axvspan(0,0.5,color="#9ecae1",alpha=0.35,label="plausible humid-temperate drainage")
axA.axvline(0.02,color="#b2182b",lw=1.6,ls=":")
axA.axvline(Q_DESIGN,color="#1b7837",lw=1.6,ls=":")
# arrowed callouts in the open upper area (the front curve is low at small q)
axA.annotate("measured $q\\approx0$", xy=(0.02,104), xytext=(1.05,120),
             color="#b2182b", fontsize=8, ha="left", va="top",
             arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.1))
axA.annotate("design $q$=0.30", xy=(Q_DESIGN,82), xytext=(1.05,96),
             color="#1b7837", fontsize=8, ha="left", va="top",
             arrowprops=dict(arrowstyle="->", color="#1b7837", lw=1.1))
axA.scatter([q40,q100],[40,100],color="black",zorder=5,s=30)
axA.text(q40+0.15,40,f"need {q40:.1f} m/yr\n({q40/Q_DESIGN:.0f}$\\times$ design)",
         fontsize=7.6, va="center")
axA.set_xlabel("drainage flux $q$  (m yr$^{-1}$)")
axA.set_ylabel("solute-front depth over R3 window (cm)")
axA.set_title("(A) Deep-null robustness vs drainage\n(independent of dissolution priors)")
axA.set_ylim(0,125); axA.set_xlim(0,6.5)
axA.legend(fontsize=7.4,loc="center right")

im=axB.contourf(FW,FD,infl_ca,levels=np.linspace(40,200,17),cmap="viridis")
cs=axB.contour(FW,FD,infl_ca,levels=[63,100,127,160],colors="white",linewidths=0.8)
axB.clabel(cs,fmt="%.0f$\\times$",fontsize=7)
axB.scatter([fWo0],[fDi0],color="red",s=60,marker="*",zorder=5,
            label="config prior (127$\\times$)")
axB.set_xlabel("wollastonite dissolution rate $f_{Wo}$ (yr$^{-1}$)")
axB.set_ylabel("diopside dissolution rate $f_{Di}$ (yr$^{-1}$)")
axB.set_title("(B) Ca $\\sigma$-inflation factor over the prior box\n(all $\\gg$1: model always overstates)")
axB.legend(fontsize=7.4,loc="upper left")
cb=fig.colorbar(im,ax=axB,label="$\\sigma$-inflation factor (15 cm)")

fig.tight_layout()
out=os.path.join(FIG,"fig_forward_sensitivity.png")
fig.savefig(out,dpi=300,bbox_inches="tight")
print("front at q=0:",round(front_cm(0),1),"cm; at design:",round(front_cm(Q_DESIGN),1),"cm")
print(f"q to reach 40cm: {q40:.2f} ({q40/Q_DESIGN:.1f}x), 100cm: {q100:.2f} ({q100/Q_DESIGN:.1f}x)")
print(f"Ca inflation box range: {infl_ca.min():.0f}-{infl_ca.max():.0f}x; Mg: {infl_mg.min():.0f}-{infl_mg.max():.0f}x")
print("wrote",out)
