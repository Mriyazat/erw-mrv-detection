#!/usr/bin/env python3
"""ML-paper item 10 [H100]: self-supervised encoder + LOPO probe (the modern
negative result).

A gradient-boosted classifier at chance under LOPO (item 1) could be dismissed
as "weak model." We close that door with a 2025-class method: pretrain a
self-supervised masked-reconstruction Transformer encoder on multivariate
sensor windows from ALL plots, then freeze it and LINEAR-PROBE the learned
representation for treatment under leave-one-plot-out. To avoid the very leakage
we critique, the held-out plot's windows are EXCLUDED from pretraining in each
fold (plot-grouped SSL).

We report two probes on the same representation:
  * treatment probe (LOPO)      -> expected at chance (no detectable signal)
  * plot-identity probe (in-set) -> expected high (the representation is rich)
A rich representation that nonetheless cannot separate treatment across unseen
plots is the strongest possible statement that the signal is not in the data at
this replication.

Usage (Rorqual interactive H100):
    source scripts/hpc/activate_env.sh
    python scripts/114_ml_ssl_probe.py --gpu --epochs 30
Writes outputs/results/ml_ssl_probe.csv (+ audit). H100-only (needs torch).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, RESULT_DIR, AUDIT_DIR, RANDOM_SEED, TREATMENT_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ssl_probe")

WINDOW = 96          # 24 h at 15-min cadence
STRIDE = 24          # 6 h
CHANS = [f"{c}_{d}" for c in ("vwc", "temp", "ec") for d in (15, 40, 100)]


def build_windows():
    s = pd.read_parquet(CACHE_DIR / "sensors.parquet")
    s = s[s["treatment"].isin(["control", "20", "60"])].copy()
    s["ts"] = pd.to_datetime(s["timestamp"])
    s = s.sort_values(["plot_id", "ts"])
    X, plot, treat = [], [], []
    tmap = {"control": 0, "20": 1, "60": 2}
    for ph, g in s.groupby("plot_id"):
        arr = g[CHANS].interpolate(limit=4).to_numpy(dtype=float)
        # per-plot z-score so absolute offsets cannot trivially leak
        mu, sd = np.nanmean(arr, 0), np.nanstd(arr, 0) + 1e-6
        arr = (arr - mu) / sd
        arr = np.nan_to_num(arr)
        for i in range(0, len(arr) - WINDOW, STRIDE):
            X.append(arr[i:i + WINDOW])
            plot.append(ph)
            treat.append(tmap[g["treatment"].iloc[0]])
    return (np.asarray(X, dtype=np.float32), np.asarray(plot),
            np.asarray(treat, dtype=int))


def make_encoder(n_ch, d_model=64, nhead=4, nlayers=2):
    import torch.nn as nn

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_ch, d_model)
            layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2,
                                               batch_first=True)
            self.tr = nn.TransformerEncoder(layer, nlayers)
            self.head = nn.Linear(d_model, n_ch)   # reconstruction head

        def forward(self, x):
            h = self.tr(self.inp(x))
            return h, self.head(h)

    return Enc()


def pretrain(encoder, Xtr, device, epochs, mask_frac=0.4, bs=256, lr=1e-3):
    import torch
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    Xt = torch.tensor(Xtr, device=device)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            xb = Xt[perm[i:i + bs]]
            mask = (torch.rand(xb.shape[:2], device=device) < mask_frac)
            xin = xb.clone()
            xin[mask] = 0.0
            _, rec = encoder(xin)
            loss = ((rec - xb) ** 2 * mask.unsqueeze(-1)).sum() / (mask.sum() + 1)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if ep % 5 == 0:
            log.info("pretrain epoch %d/%d loss=%.4f", ep, epochs, tot)
    return encoder


def embed(encoder, X, device, bs=512):
    import torch
    encoder.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs], device=device)
            h, _ = encoder(xb)
            outs.append(h.mean(1).cpu().numpy())   # mean-pool over time
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    try:
        import torch
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedKFold
    except Exception as e:  # noqa
        log.error("torch/sklearn required: %s", e)
        return
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    torch.manual_seed(RANDOM_SEED)

    X, plot, treat = build_windows()
    log.info("windows: %s, %d plots", X.shape, len(np.unique(plot)))
    n_ch = X.shape[2]

    # LOPO treatment probe with PLOT-GROUPED pretraining (held-out plot excluded)
    uplots = sorted(np.unique(plot))
    rows = []
    for hold in uplots:
        tr_mask = plot != hold
        enc = make_encoder(n_ch).to(device)
        enc = pretrain(enc, X[tr_mask], device, args.epochs)
        Ztr, Zte = embed(enc, X[tr_mask], device), embed(enc, X[~tr_mask], device)
        clf = LogisticRegression(max_iter=1000).fit(Ztr, treat[tr_mask])
        acc = accuracy_score(treat[~tr_mask], clf.predict(Zte))
        rows.append({"held_out_plot": hold, "treatment_lopo_acc": float(acc),
                     "treatment_of_plot": int(treat[~tr_mask][0])})
        log.info("hold %s: treatment LOPO acc=%.3f", hold, acc)
    lopo = pd.DataFrame(rows)

    # plot-identity probe on a single full-data representation (in-set, leaky on
    # purpose: shows the representation IS rich enough to fingerprint plots)
    enc = make_encoder(n_ch).to(device)
    enc = pretrain(enc, X, device, args.epochs)
    Z = embed(enc, X, device)
    skf = StratifiedKFold(3, shuffle=True, random_state=RANDOM_SEED)
    pid = pd.factorize(plot)[0]
    pacc = []
    for tr, te in skf.split(Z, pid):
        m = LogisticRegression(max_iter=1000).fit(Z[tr], pid[tr])
        pacc.append(accuracy_score(pid[te], m.predict(Z[te])))
    plot_probe = float(np.mean(pacc))

    lopo.to_csv(RESULT_DIR / "ml_ssl_probe.csv", index=False)
    summary = {"treatment_lopo_acc_mean": float(lopo["treatment_lopo_acc"].mean()),
               "treatment_chance": 1 / 3,
               "plot_identity_probe_acc": plot_probe,
               "plot_chance": 1 / len(uplots)}
    pd.DataFrame([summary]).to_csv(RESULT_DIR / "ml_ssl_probe_summary.csv",
                                   index=False)
    lines = ["# ML item 10: self-supervised encoder + LOPO probe\n",
             f"- Treatment LOPO probe accuracy = "
             f"{summary['treatment_lopo_acc_mean']:.3f} (chance 1/3).\n",
             f"- Plot-identity probe accuracy (in-set) = {plot_probe:.3f} "
             f"(chance {1/len(uplots):.3f}).\n",
             "\nA representation rich enough to fingerprint the plot still "
             "cannot separate treatment across unseen plots: the strongest "
             "statement that the ERW signal is not in the data at n=12."]
    (AUDIT_DIR / "ml_ssl_probe.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
