"""Leakage-safe cross-validation for the small, spatially-clustered resin data.

Design rules (from the verification pass):
  * Holdout unit is the PHYSICAL PLOT (plot_id), never plot_half, so a held-out
    plot's sibling W/E half cannot leak into training.
  * ALL preprocessing (scaling, imputation) is fit on TRAIN folds only.
  * Honest baseline = train-fold mean predictor; skill is reported RELATIVE to
    it (R^2 vs mean, MAE ratio), never as an absolute R^2 in isolation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED


def leave_one_plot_out(groups: pd.Series):
    """Yield (train_idx, test_idx) holding out one plot_id at a time."""
    groups = groups.reset_index(drop=True)
    for plot in sorted(groups.unique()):
        test = groups.index[groups == plot].to_numpy()
        train = groups.index[groups != plot].to_numpy()
        if len(test) and len(train):
            yield plot, train, test


@dataclass
class CVResult:
    target: str
    model_name: str
    n: int
    oof_pred: np.ndarray
    y_true: np.ndarray
    r2_oof: float
    mae_oof: float
    baseline_mae: float
    r2_vs_mean: float
    mae_skill: float            # 1 - mae/baseline_mae (>0 means beats mean)
    per_fold: list = field(default_factory=list)


def run_grouped_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                   model_factory, model_name: str, target: str) -> CVResult:
    """Leave-one-plot-out CV with in-fold scaling + imputation.

    `model_factory` returns a fresh estimator each call.
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True).astype(float)
    groups = groups.reset_index(drop=True)

    oof = np.full(len(y), np.nan)
    per_fold = []
    for plot, tr, te in leave_one_plot_out(groups):
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        Xtr = sc.fit_transform(imp.fit_transform(X.iloc[tr]))
        Xte = sc.transform(imp.transform(X.iloc[te]))
        ytr = y.iloc[tr].to_numpy()

        model = model_factory()
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        oof[te] = pred
        yte = y.iloc[te].to_numpy()
        per_fold.append({
            "held_out_plot": plot, "n_test": len(te),
            "mae": float(mean_absolute_error(yte, pred)) if len(te) else np.nan,
            "train_mean": float(ytr.mean()),
        })

    mask = ~np.isnan(oof)
    yt = y.to_numpy()[mask]
    yp = oof[mask]
    base_pred = np.full_like(yt, yt.mean())  # conservative pooled-mean baseline
    base_mae = mean_absolute_error(yt, base_pred)
    mae = mean_absolute_error(yt, yp)
    return CVResult(
        target=target, model_name=model_name, n=int(mask.sum()),
        oof_pred=yp, y_true=yt,
        r2_oof=float(r2_score(yt, yp)),
        mae_oof=float(mae), baseline_mae=float(base_mae),
        r2_vs_mean=float(r2_score(yt, yp)),
        mae_skill=float(1 - mae / base_mae) if base_mae > 0 else np.nan,
        per_fold=per_fold,
    )


def mean_predictor_baseline(y: pd.Series, groups: pd.Series) -> dict:
    """LOPO mean-predictor: predict each plot with the mean of the others."""
    y = y.reset_index(drop=True).astype(float)
    groups = groups.reset_index(drop=True)
    oof = np.full(len(y), np.nan)
    for _, tr, te in leave_one_plot_out(groups):
        oof[te] = y.iloc[tr].mean()
    mask = ~np.isnan(oof)
    return {
        "r2_oof": float(r2_score(y[mask], oof[mask])),
        "mae_oof": float(mean_absolute_error(y[mask], oof[mask])),
        "n": int(mask.sum()),
    }
