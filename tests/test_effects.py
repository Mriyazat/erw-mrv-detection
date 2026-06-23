"""Golden-master tests pinning effect-size + reconciliation invariants."""

import numpy as np
import pytest

from src.analysis.effects import compute_dose_response, compute_effects
from src.io.load_resin import load_resin
from src.stats.bootstrap import cohens_d_hedges_g


@pytest.fixture(scope="module")
def resin():
    try:
        return load_resin()
    except Exception as e:
        pytest.skip(f"resin cache unavailable: {e}")


def test_hedges_g_matches_prior_golden(resin):
    """Pinned value reconciled against erw_mrv phase3 (Ca, R1, 15cm, 20 t/ha)."""
    eff = compute_effects(resin, ["ca_ppm"])
    row = eff[(eff["round"] == 1) & (eff["depth_cm"] == 15)
              & (eff["treatment"] == "20")].iloc[0]
    assert row["hedges_g"] == pytest.approx(0.056268, abs=1e-5)


def test_dose_response_units_factor(resin):
    """t/ha slope is exactly 0.1 x the kg/m^2 slope."""
    dr = compute_dose_response(resin, ["ca_ppm"]).dropna(
        subset=["slope_ppm_per_tha", "slope_ppm_per_kgm2"])
    ratio = (dr["slope_ppm_per_tha"] / dr["slope_ppm_per_kgm2"]).round(6)
    assert (ratio == 0.1).all()


def test_hedges_g_small_sample_correction():
    treated = np.array([10.0, 12.0, 11.0, 13.0])
    control = np.array([8.0, 9.0, 7.0, 8.5])
    out = cohens_d_hedges_g(treated, control)
    assert abs(out["hedges_g"]) < abs(out["d"])  # shrinks toward 0
    assert out["mean_diff"] == pytest.approx(treated.mean() - control.mean())


def test_effect_sign_convention():
    """Positive g => treated > control."""
    out = cohens_d_hedges_g(np.array([5.0, 6, 7, 8]), np.array([1.0, 2, 3, 4]))
    assert out["hedges_g"] > 0
