"""Golden-master tests pinning verified data-extraction invariants."""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    CACHE_DIR,
    DOSE_KGM2,
    DOSE_THA,
    SENSOR_BLOCK_DEPTH_ORDER,
    TREATMENT_MAP,
)

CACHE = CACHE_DIR


def _need(name):
    p = CACHE / name
    if not p.exists():
        pytest.skip(f"{name} not built; run `make data` first")
    return pd.read_parquet(p)


def test_config_locked_conventions():
    assert SENSOR_BLOCK_DEPTH_ORDER == [100, 40, 15]
    assert DOSE_THA == {"control": 0.0, "20": 20.0, "60": 60.0}
    assert DOSE_KGM2["60"] == 6.0
    assert len(TREATMENT_MAP) == 12


def test_cache_row_counts():
    # All four incremental ZENTRA downloads per plot are ingested -> continuous
    # May 2025 -> Apr 2026 coverage (393,844 rows). See docs/DATA_AUDIT.md.
    assert len(_need("sensors.parquet")) == 393844
    assert len(_need("resin.parquet")) == 121
    assert len(_need("weather_15min.parquet")) == 39685
    assert len(_need("chamber.parquet")) == 972
    assert len(_need("plot_metadata.parquet")) == 24


def test_no_winter_gap():
    """The 2025-10-21 -> 2026-03-25 winter must be present for every plot."""
    s = _need("sensors.parquet")
    winter = s[(s["timestamp"] > "2025-10-22") & (s["timestamp"] < "2026-03-24")]
    assert len(winter) > 100_000
    assert set(winter["plot_id"].unique()) == set(TREATMENT_MAP)


def test_resin_qa_and_si_absence():
    r = _need("resin.parquet")
    assert (r["qa_flag"] != "").sum() == 1
    assert "55303" in r.loc[r["qa_flag"] != "", "barcode"].values
    assert not [c for c in r.columns if c.startswith("si_")]
    assert (r["ph"].fillna(0) == 0).all()  # pH unusable


def test_depth_correction_thermal_damping():
    """Corrected mapping => diurnal temp amplitude decreases with depth."""
    s = _need("sensors.parquet")
    su = s[(s["timestamp"] >= "2025-07-15") & (s["timestamp"] < "2025-08-15")]

    def amp(g, c):
        gg = g.dropna(subset=[c]).copy()
        gg["d"] = gg["timestamp"].dt.date
        return gg.groupby("d")[c].agg(lambda x: x.max() - x.min()).mean()

    monotone = 0
    evaluated = 0
    for _, g in su.groupby("plot_id"):
        a = [amp(g, f"temp_{d}") for d in (15, 40, 100)]
        if not any(np.isnan(x) for x in a):
            evaluated += 1
            monotone += int(a[0] >= a[1] >= a[2])
    assert evaluated >= 10
    assert monotone / evaluated >= 0.75  # >=75% plots damp with depth


def test_shallow_block_added_later_nulls():
    """15 cm (3rd block, added mid-2025) carries more nulls than the 100 cm
    (1st block), which is complete from deployment. Stated as a relative
    invariant: the absolute 15 cm null rate fell below 5% once the winter
    downloads (all three blocks present) were added."""
    s = _need("sensors.parquet")
    assert s["vwc_100"].isna().mean() < 0.01
    assert s["vwc_15"].isna().mean() > s["vwc_100"].isna().mean() + 0.02


def test_headline_has_no_degenerate_ci():
    from src.config import RESULT_DIR
    p = RESULT_DIR / "headline_summary.csv"
    if not p.exists():
        pytest.skip("headline not built; run `make headline`")
    h = pd.read_csv(p)
    for col in ("ci_low", "ci_high"):
        finite = h[col].dropna()
        assert (finite.abs() <= 1e4).all(), "degenerate CI in headline"
