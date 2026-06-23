"""First-principles SNR model for the Simcoe wollastonite + diopside trial.

    Amendment: 50:50 wollastonite (CaSiO3) + diopside (CaMgSi2O6)
    Weathering: dot_M_i = (M_app / 2) * f_i   (f_wo = 1/3, f_di = 1/30 yr^-1)
    Stoichiometry:
        CaSiO3    -> Ca + Si        (nu_Ca=1, nu_Mg=0, nu_Si=1)
        CaMgSi2O6 -> Ca + Mg + 2 Si (nu_Ca=1, nu_Mg=1, nu_Si=2)
    Surface flux:      F_{X,0} = sum_i dot_M_i / MW_i * nu_{X,i}
    Depth attenuation: F_X(z)  = F_{X,0} * exp(-k_r * z)
    Concentration:     dC(z)   = F_X(z) / q
    SNR:               SNR(z)   = F_X(z) / sigma_{F,X}
    Max-detectable:    z_max    = (1/k_r) * ln(F_{X,0} / (SNR* * sigma_{F,X}))

Model is in ANNUAL flux units; resin observations are per-round ppm, so the
empirical-vs-theoretical comparison happens at the SNR / effect-size level.
`k_retention` and the `sigma_F` multiplier are the reconciliation levers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import DEPTHS_M, DOSE_THA, SNR_MODEL

logger = logging.getLogger(__name__)

MW_WOLLASTONITE = SNR_MODEL["mw_wollastonite_g_mol"]
MW_DIOPSIDE     = SNR_MODEL["mw_diopside_g_mol"]
F_RATE_WO       = SNR_MODEL["f_rate_wollastonite"]
F_RATE_DI       = SNR_MODEL["f_rate_diopside"]
Q_DRAINAGE      = SNR_MODEL["q_drainage_m_yr"]

NU_WO = {"Ca": 1, "Mg": 0, "Si": 1}
NU_DI = {"Ca": 1, "Mg": 1, "Si": 2}

# Dose is t/ha in config; the model wants kg/m^2 (1 t/ha = 0.1 kg/m^2).
DOSE_KGM2 = {k: v * 0.1 for k, v in DOSE_THA.items()}


def compute_annual_weathered_mass(m_app_kg_m2: float) -> dict:
    half = m_app_kg_m2 / 2.0
    return {"wo": half * F_RATE_WO, "di": half * F_RATE_DI}


def compute_surface_ion_flux(m_app_kg_m2: float) -> dict:
    dm = compute_annual_weathered_mass(m_app_kg_m2)
    out = {}
    for ion in ("Ca", "Mg", "Si"):
        out[ion] = (
            (dm["wo"] / (MW_WOLLASTONITE * 1e-3)) * NU_WO[ion]
            + (dm["di"] / (MW_DIOPSIDE * 1e-3))   * NU_DI[ion]
        )
    return out


def compute_flux_at_depth(f0: float, k_r: float, z_m: float) -> float:
    return f0 * np.exp(-k_r * z_m)


def compute_concentration_increment(fz: float, q: float = Q_DRAINAGE) -> float:
    return fz / max(q, 1e-9)


def compute_snr(fz: float, sigma: float) -> float:
    if sigma <= 0:
        return float("inf")
    return fz / sigma


def compute_max_detectable_depth(f0: float, k_r: float, sigma: float,
                                 snr_threshold: float = 3.0) -> float:
    ratio = f0 / (snr_threshold * max(sigma, 1e-12))
    if ratio <= 0:
        return 0.0
    return (1.0 / k_r) * np.log(ratio)


def get_theoretical_snr_at_depth(treatment: str, depth_cm: int, ion: str = "Ca",
                                 sigma_multiplier: float = 1.0,
                                 k_retention_override: float | None = None) -> float:
    m_app = DOSE_KGM2.get(treatment, 0.0)
    if m_app == 0:
        return 0.0
    f0 = compute_surface_ion_flux(m_app)[ion]
    k_r = (k_retention_override if k_retention_override is not None
           else SNR_MODEL["k_retention_per_m"][ion])
    fz = compute_flux_at_depth(f0, k_r, DEPTHS_M[depth_cm])
    sigma = SNR_MODEL["sigma_F_mol_m2_yr"][ion] * sigma_multiplier
    return compute_snr(fz, sigma)


def compute_full_snr_table(sigma_multiplier: float = 1.0,
                           k_retention_override: dict | None = None) -> pd.DataFrame:
    sigma_map = {ion: s * sigma_multiplier
                 for ion, s in SNR_MODEL["sigma_F_mol_m2_yr"].items()}
    k_map = dict(SNR_MODEL["k_retention_per_m"])
    if k_retention_override:
        k_map.update(k_retention_override)

    rows = []
    for treatment, m_app in DOSE_KGM2.items():
        for depth_cm, depth_m in DEPTHS_M.items():
            for ion in ("Ca", "Mg", "Si"):
                if m_app == 0:
                    f0 = fz = dc = snr = z_max = 0.0
                else:
                    f0 = compute_surface_ion_flux(m_app)[ion]
                    fz = compute_flux_at_depth(f0, k_map[ion], depth_m)
                    dc = compute_concentration_increment(fz)
                    snr = compute_snr(fz, sigma_map[ion])
                    z_max = compute_max_detectable_depth(
                        f0, k_map[ion], sigma_map[ion], 3.0)
                rows.append({
                    "treatment": treatment, "depth_cm": depth_cm,
                    "depth_m": depth_m, "ion": ion,
                    "F0_mol_m2_yr": round(f0, 4), "Fz_mol_m2_yr": round(fz, 4),
                    "delta_C_mol_m3": round(dc, 4), "SNR": round(snr, 2),
                    "z_max_snr3_m": round(z_max, 2),
                    "sigma_mult": sigma_multiplier, "k_retention": k_map[ion],
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = compute_full_snr_table()
    print(df.pivot_table(values="SNR", index=["treatment", "depth_cm"],
                         columns="ion").to_string())
