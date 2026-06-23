#!/usr/bin/env python3
"""Experiment 27: The leakage demonstration -- naive vs honest cross-validation.

Reproduces the prior 'ML detects ERW treatment with ~100% accuracy at every
depth' result and shows it is spatial MEMORIZATION, not detection:

  * NAIVE  (row-level stratified k-fold): rows from all 12 plots appear in both
    train and test, so the model memorizes each plot's sensor fingerprint and,
    because plot->treatment is a fixed lookup at n=12, classifies treatment
    near-perfectly -- including at 100 cm where the physical SNR is LOWEST
    (the tell-tale sign of leakage).
  * HONEST (leave-one-plot-out / GroupKFold by plot_id): test plots are unseen,
    so the model must generalize across the natural between-plot heterogeneity
    and collapses toward chance (1/3).

Also: a plot-identity probe (predict plot_id from the same features) confirms the
features encode plot identity, the mechanism that makes treatment trivially
'predictable' under a leaky split.

Self-contained; sensors.parquet only.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(__file__))
RES = os.path.join(ROOT, "outputs", "results")
AUD = os.path.join(ROOT, "outputs", "audits")
FIG = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(42)

s = pd.read_parquet(os.path.join(ROOT, "outputs", "cache", "sensors.parquet"))
s = s[s["treatment"].isin(["control", "20", "60"])].copy()
chans = [f"{v}_{d}" for v in ("vwc","temp","ec") for d in (15,40,100)] + \
        [f"mp_{d}" for d in (15,40,100)]
chans = [c for c in chans if c in s.columns]
s = s.dropna(subset=["ec_15","ec_40","ec_100","vwc_15","vwc_40","vwc_100"])
# add weak temporal features (as the prior pipeline did)
s["ts"] = pd.to_datetime(s["timestamp"])
s["hour"] = s["ts"].dt.hour
s["doy"]  = s["ts"].dt.dayofyear
feat_all = chans + ["hour","doy"]

# subsample for speed, balanced across plots (explicit loop preserves plot_id
# under pandas 2.x, where group_keys=False apply drops the grouping column)
s = pd.concat([g.sample(min(len(g), 1000), random_state=42)
               for _, g in s.groupby("plot_id")]).reset_index(drop=True)
y = s["treatment"].map({"control":0,"20":1,"60":2}).values
groups = s["plot_id"].values
print(f"n={len(s)} rows, {s['plot_id'].nunique()} plots, classes={np.bincount(y)}")

def run(X, y, groups, label):
    clf = lambda: HistGradientBoostingClassifier(max_iter=80, max_depth=6, random_state=42)
    # NAIVE: stratified k-fold over rows (leaky)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    naive=[]
    for tr,te in skf.split(X,y):
        m=clf().fit(X[tr],y[tr]); naive.append(accuracy_score(y[te],m.predict(X[te])))
    # HONEST: leave-one-plot-out
    logo=LeaveOneGroupOut(); hon=[]; hpred=[]; htrue=[]
    for tr,te in logo.split(X,y,groups):
        m=clf().fit(X[tr],y[tr]); p=m.predict(X[te])
        hon.append(accuracy_score(y[te],p)); hpred+=list(p); htrue+=list(y[te])
    return (np.mean(naive), np.mean(hon),
            f1_score(htrue,hpred,average="macro"))

rows=[]
# (a) full feature set, all depths
Xall=s[feat_all].fillna(0).values
n,h,f=run(Xall,y,groups,"all")
rows.append(("all_depths_full_features",n,h,f))
# (b) per-depth: 15 cm (highest SNR) and 100 cm (lowest SNR) -- the key contrast
for d in (15,100):
    fd=[f"{v}_{d}" for v in ("vwc","temp","ec")]+([f"mp_{d}"] if f"mp_{d}" in s else [])
    Xd=s[fd].fillna(0).values
    n,h,f=run(Xd,y,groups,f"d{d}")
    rows.append((f"depth_{d}cm_only",n,h,f))

# (c) plot-identity probe: predict plot_id from features (leaky split)
gp=pd.factorize(s["plot_id"])[0]
skf=StratifiedKFold(3,shuffle=True,random_state=42); pid=[]
for tr,te in skf.split(Xall,gp):
    m=HistGradientBoostingClassifier(max_iter=80,max_depth=6,random_state=42).fit(Xall[tr],gp[tr])
    pid.append(accuracy_score(gp[te],m.predict(Xall[te])))
plot_id_acc=float(np.mean(pid))

df=pd.DataFrame(rows,columns=["setting","naive_acc","honest_lopo_acc","honest_macroF1"])
df["chance"]=1/3
df.to_csv(os.path.join(RES,"cnew_leakage_demo.csv"),index=False)
print(df.round(3).to_string(index=False))
print(f"\nplot-identity probe accuracy (leaky split): {plot_id_acc:.3f} (chance=1/12={1/12:.3f})")

with open(os.path.join(AUD,"cnew_leakage_demo.md"),"w") as fh:
    fh.write("# Experiment 27: Leakage demonstration (naive vs honest CV)\n\n")
    fh.write(df.round(3).to_markdown(index=False))
    fh.write(f"\n\n- Plot-identity probe accuracy (leaky split) = {plot_id_acc:.3f} "
             f"(chance 1/12 = {1/12:.3f}): the features encode plot identity.\n")
    fh.write("- NAIVE row-level CV near-perfectly 'classifies treatment' at every "
             "depth (incl. 100 cm where SNR is lowest) = spatial memorization. "
             "Under leave-one-plot-out the same model collapses toward chance "
             "(1/3): the treatment signal is not generalizable at n=12, "
             "consistent with the SNR detection floor.\n")
print("\nwrote csv + audit")

# --- figure: naive vs honest accuracy, with a non-overlapping callout -------- #
labels = ["All depths\n(full features)", "15 cm only\n(SNR \u2248 67)",
          "100 cm only\n(SNR \u2248 34)"]
naive_v = df["naive_acc"].to_numpy()
honest_v = df["honest_lopo_acc"].to_numpy()
xpos = np.arange(len(labels))
bw = 0.38
fig, ax = plt.subplots(figsize=(8.4, 5.0))
b1 = ax.bar(xpos - bw / 2, naive_v, bw, color="#c0392b", edgecolor="k",
            label="Naive (row-level CV) \u2014 leaky")
b2 = ax.bar(xpos + bw / 2, honest_v, bw, color="#2166ac", edgecolor="k",
            label="Honest (leave-one-plot-out)")
ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9, fontweight="bold")
ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9, fontweight="bold")
ax.axhline(1 / 3, ls="--", color="grey", lw=1)
ax.text(len(labels) - 0.52, 1 / 3 + 0.012, "chance (1/3)", fontsize=8,
        color="grey", ha="right", va="bottom")
ax.set_ylim(0, 1.34)
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_ylabel("treatment-classification accuracy")
ax.set_title("'ML detects ERW treatment' is spatial memorization, not detection")
# explanatory callout placed in the clear headroom above the bars (no overlap)
ax.text((len(labels) - 1) / 2, 1.30, "Naive accuracy is HIGHER at 100 cm "
        "(SNR\u224834) than at 15 cm (SNR\u224867):\ndetection improving as the "
        "physical signal weakens is the signature of leakage.",
        ha="center", va="top", fontsize=8.4, color="#9b2226")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_leakage_demo.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig_leakage_demo.png")
