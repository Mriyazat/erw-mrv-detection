"""ML utilities: leakage-safe CV, in-fold preprocessing, honest baselines."""

from src.ml.cv import (
    leave_one_plot_out,
    run_grouped_cv,
    mean_predictor_baseline,
)

__all__ = ["leave_one_plot_out", "run_grouped_cv", "mean_predictor_baseline"]
