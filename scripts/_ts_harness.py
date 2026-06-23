"""Shared loader for the ML-paper forecasting/imputation scripts (110-114).

Reuses the mature, honest harness in `63_cnew_deep_ts.py` (panel builder,
seasonal-naive baseline, skill-vs-naive scoring, rolling cutoffs) without
duplicating it. Numbered module files cannot be imported normally, so we load
63 via importlib and re-export the pieces the new scripts need.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "deep_ts_63", _HERE / "63_cnew_deep_ts.py")
deep_ts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(deep_ts)  # type: ignore

# re-exports
build_panel = deep_ts.build_panel
seasonal_naive_mae = deep_ts.seasonal_naive_mae
skill_rows = deep_ts.skill_rows
rolling_cutoffs = deep_ts._rolling_cutoffs
mean_absolute_error = deep_ts.mean_absolute_error
HORIZON = deep_ts.HORIZON
SEASON = deep_ts.SEASON
INPUT_SIZE = deep_ts.INPUT_SIZE
N_WINDOWS = deep_ts.N_WINDOWS
FREQ = deep_ts.FREQ
TARGET = deep_ts.TARGET
