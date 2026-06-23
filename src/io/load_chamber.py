"""Standalone eosAC chamber-flux loader.

Chamber data is analysed SEPARATELY from resin / sensor / weather by default
(different season, 1 plot per non-control arm). The 7-day spring-2026 run
(Julian days ~104-111 = Apr 14-21, 2026) covered 6 chambers (mux ports 2-7)
over plots {6W x2, 6E, 7W, 7E x2}.

Both Linear (L) and Exponential (E) flux estimates plus uncertainties and
fit coefficients are retained for the QA gate (flag when L and E disagree).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import CACHE_DIR, CHAMBER_DIR, CHAMBER_PORT_TO_PLOT, TREATMENT_MAP

logger = logging.getLogger(__name__)

CHAMBER_FILE = CHAMBER_DIR / "chamber.xlsx"
CACHE_PATH   = CACHE_DIR / "chamber.parquet"
DEFAULT_YEAR = 2026

COLUMN_MAP = {
    "Julian Day": "julian_day",
    "Chamber Serial": "chamber_serial",
    "Chamber Error Status": "chamber_error",
    "Analyzer Error Status": "analyzer_error",
    "Multiplexer Serial": "mux_serial",
    "Multiplexer Port": "mux_port",
    "Measurement Duration (s)": "duration_s",
    "        Mean CO2 (ppm)": "co2_ppm",
    "        Mean CH4 (ppm)": "ch4_ppm",
    "        Mean N2O (ppm)": "n2o_ppm",
    "        Mean NH3 (ppm)": "nh3_ppm",
    "        Mean H2O (percent)": "h2o_pct",
    "     Chem Detect (0/1)": "chem_detect",
    "    Cav. Pressure (kPa)": "cav_pressure_kpa",
    "Cav. Temperature (K)": "cav_temp_k",
    "    Water Content (fraction)": "water_content",
    "Chmbr. Temperature (K)": "chamber_temp_k",
    " Chmbr. Pressure (kPa)": "chamber_pressure_kpa",
    "    Flux CO2 (L) (umol/m^2/s)": "flux_co2_lin",
    "    Flux CO2 (E) (umol/m^2/s)": "flux_co2_exp",
    "    Flux CH4 (L) (nmol/m^2/s)": "flux_ch4_lin",
    "    Flux CH4 (E) (nmol/m^2/s)": "flux_ch4_exp",
    "    Flux N2O (L) (nmol/m^2/s)": "flux_n2o_lin",
    "    Flux N2O (E) (nmol/m^2/s)": "flux_n2o_exp",
    "    Flux NH3 (L) (umol/m^2/s)": "flux_nh3_lin",
    "    Flux NH3 (E) (umol/m^2/s)": "flux_nh3_exp",
    "  e_Flux CO2 (L) (umol/m^2/s)": "e_flux_co2_lin",
    "  e_Flux CO2 (E) (umol/m^2/s)": "e_flux_co2_exp",
    "  e_Flux CH4 (L) (nmol/m^2/s)": "e_flux_ch4_lin",
    "  e_Flux CH4 (E) (nmol/m^2/s)": "e_flux_ch4_exp",
    "  e_Flux N2O (L) (nmol/m^2/s)": "e_flux_n2o_lin",
    "  e_Flux N2O (E) (nmol/m^2/s)": "e_flux_n2o_exp",
    "  e_Flux NH3 (L) (umol/m^2/s)": "e_flux_nh3_lin",
    "  e_Flux NH3 (E) (umol/m^2/s)": "e_flux_nh3_exp",
    "   AUX Voltage 1 (V)": "aux_v1", "   AUX Voltage 2 (V)": "aux_v2",
    "   AUX Voltage 3 (V)": "aux_v3", "   AUX Current 1 (mA)": "aux_i1_ma",
    "   AUX Current 2 (mA)": "aux_i2_ma",
    "c (Linear Intercept)": "fit_lin_intercept",
    "f (Linear Slope)": "fit_lin_slope",
    "c (Exponential Intercept)": "fit_exp_intercept",
    "d (Exponential Multiplier)": "fit_exp_multiplier",
    "a (Exponential Exponent)": "fit_exp_exponent",
    "MX Analog 1": "mx_analog_1", "MX Analog 2": "mx_analog_2",
    "MX Analog 3": "mx_analog_3", "MX Analog 4": "mx_analog_4",
    "MX Analog 5": "mx_analog_5", "MX Analog 6": "mx_analog_6",
    "MX Analog 7": "mx_analog_7", "MX Analog 8": "mx_analog_8",
    "Deadband Start (pts)": "deadband_start_pts",
    "Deadband End (pts)": "deadband_end_pts",
}

FLUX_GASES = ("co2", "ch4", "n2o", "nh3")


def julian_day_to_timestamp(jd: float, year: int = DEFAULT_YEAR) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=float(jd) - 1.0)


def load_chamber(use_cache: bool = True, year: int = DEFAULT_YEAR) -> pd.DataFrame:
    if use_cache and CACHE_PATH.exists():
        logger.info("Loading cached chamber data from %s", CACHE_PATH)
        return pd.read_parquet(CACHE_PATH)

    logger.info("Reading %s", CHAMBER_FILE.name)
    raw = pd.read_excel(CHAMBER_FILE, sheet_name="in")

    # whitespace-robust header matching (verification hardening)
    norm = {str(c).strip(): c for c in raw.columns}
    keep = {}
    for src_name, out_name in COLUMN_MAP.items():
        if src_name in raw.columns:
            keep[src_name] = out_name
        elif src_name.strip() in norm:
            keep[norm[src_name.strip()]] = out_name
    missing = [v for k, v in COLUMN_MAP.items()
               if v not in keep.values()]
    if missing:
        logger.warning("Chamber file missing %d expected columns: %s",
                       len(missing), missing[:5] + (["..."] if len(missing) > 5 else []))
    df = raw[list(keep.keys())].rename(columns=keep).copy()

    df["timestamp"] = df["julian_day"].apply(
        lambda j: julian_day_to_timestamp(j, year=year))
    df["year_assumed"] = year

    df["plot_id"] = df["mux_port"].map(CHAMBER_PORT_TO_PLOT)
    df["treatment"] = df["plot_id"].map(TREATMENT_MAP)
    if df["plot_id"].isna().any():
        unknown = sorted(df.loc[df["plot_id"].isna(), "mux_port"].unique())
        logger.warning("mux ports without plot mapping: %s", unknown)

    df["valid"] = ((df["chamber_error"].fillna(1) == 0)
                   & (df["analyzer_error"].fillna(1) == 0))

    for gas in FLUX_GASES:
        lin, exp = df.get(f"flux_{gas}_lin"), df.get(f"flux_{gas}_exp")
        if lin is None or exp is None:
            continue
        denom = (lin.abs() + exp.abs()) / 2 + 1e-9
        df[f"flux_{gas}_disagree_pct"] = 100 * (exp - lin).abs() / denom
        df[f"flux_{gas}_qa_ok"] = df[f"flux_{gas}_disagree_pct"].fillna(np.inf) <= 25.0

    qa_ok_cols = [f"flux_{g}_qa_ok" for g in FLUX_GASES if f"flux_{g}_qa_ok" in df.columns]
    if qa_ok_cols:
        df["all_fits_qa_ok"] = df[qa_ok_cols].all(axis=1)

    df = df.sort_values("timestamp").reset_index(drop=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    logger.info("Cached %d chamber rows to %s", len(df), CACHE_PATH)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    df = load_chamber(use_cache=False)
    print(f"\nShape: {df.shape}")
    print(df.groupby("mux_port").agg(plot_id=("plot_id", "first"),
          treatment=("treatment", "first"), n=("timestamp", "count")).to_string())
