"""Project-wide configuration - the single source of truth.

Plot maps, treatment maps, resin schema, sensor columns, deployment dates,
and physical-model parameters all live here. No constant is duplicated
elsewhere.

Locked conventions (from the verification pass):
  * DOSE: canonical unit is t/ha (0 / 20 / 60). kg/m^2 provided alongside
    (1 t/ha = 0.1 kg/m^2). Every dose-response result MUST state its unit.
  * EFFECT SIZE: Hedges' g is the reported effect size (small-sample
    corrected). Cohen's d is computed but secondary.
  * WEATHER CUTOFF: sensor rows past the last weather timestamp are an
    explicit, logged policy (not a silent drop).
"""

from __future__ import annotations

import platform
from pathlib import Path

_HOSTNAME = platform.node().lower()
IS_HPC = any(tag in _HOSTNAME for tag in (
    "alliancecan", "rorqual", "narval", "cedar", "beluga", "graham",
))

if IS_HPC:
    PROJECT_ROOT = Path(
        "/home/mriyazat/links/projects/def-erangauk-ab/mriyazat/erw"
    )
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR    = PROJECT_ROOT / "data"
SENSOR_DIR  = DATA_DIR / "sensors"
WEATHER_DIR = DATA_DIR / "weather"
RESIN_DIR   = DATA_DIR / "resin"
CHAMBER_DIR = DATA_DIR / "chamber"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR  = OUTPUT_DIR / "cache"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "results"
AUDIT_DIR  = OUTPUT_DIR / "audits"
LLM_DIR    = OUTPUT_DIR / "llm"
DOCS_DIR   = PROJECT_ROOT / "docs"

for _d in (CACHE_DIR, FIGURE_DIR, RESULT_DIR, AUDIT_DIR, LLM_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 12-plot wollastonite + diopside trial
# ---------------------------------------------------------------------------
# Each plot is split E / W; the treatment level applies to both halves.
# Plots 1, 2, 9 have loggers in the archive (z6-31197, z6-31238) added ~2025-09-30
# with REDUCED instrumentation and UNKNOWN treatment, so they are intentionally
# NOT ingested here (no dose label -> unusable for the dose-response design).
# They remain available as untreated baseline/forecast series if ever needed.
# See docs/DATA_AUDIT.md.

TREATMENT_MAP: dict[str, str] = {
    "4W": "control", "4E": "control", "6E": "control", "7W": "control",
    "3W": "20", "5E": "20", "7E": "20", "8W": "20",
    "3E": "60", "5W": "60", "6W": "60", "8E": "60",
}

TREATMENT_ORDER = ["control", "20", "60"]

# CANONICAL dose unit: tonnes per hectare.
DOSE_THA: dict[str, float] = {"control": 0.0, "20": 20.0, "60": 60.0}
# Secondary unit: kilograms per square metre (1 t/ha = 0.1 kg/m^2).
DOSE_KGM2: dict[str, float] = {k: v * 0.1 for k, v in DOSE_THA.items()}

CANONICAL_DOSE_UNIT = "t_ha"          # locked convention
CANONICAL_EFFECT_SIZE = "hedges_g"    # locked convention

# Sensor downloads are INCREMENTAL, not cumulative: each ZENTRA export only
# spans since the previous download. The four per-plot downloads are
#   dl20251021 (2025 season) | dl20251130 | dl20260325 | dl20260429
# and together give continuous coverage May 2025 -> Apr 2026. The loader reads
# the full list and dedups by timestamp. (Earlier repos used only the first +
# last file and silently dropped the 2025-10-21 -> 2026-03-25 winter; see
# docs/DATA_AUDIT.md.)
PLOT_FILES = {
    p: sorted(SENSOR_DIR.glob(f"plot_{p}_dl*.xlsx")) for p in TREATMENT_MAP
}
# Backward-compatible aliases: earliest = "2025 season", latest = "2026 spring".
PLOT_FILES_2025 = {p: files[0] for p, files in PLOT_FILES.items() if files}
PLOT_FILES_2026 = {p: files[-1] for p, files in PLOT_FILES.items() if files}

# eosAC chamber multiplexer port -> ERW plot. Spring-2026 campaign only.
# KEPT SEPARATE from resin/sensor analyses by default (different season,
# 1 plot per non-control arm). See docs/CHAMBER_JOIN_DECISION.md.
CHAMBER_PORT_TO_PLOT: dict[int, str] = {
    2: "6W", 3: "6W", 4: "6E", 5: "7W", 6: "7E", 7: "7E",
}

# ---------------------------------------------------------------------------
# Sensor channels (TEROS-12 = VWC + temp + EC; TEROS-21 = matric potential)
# ---------------------------------------------------------------------------
DEPTHS_CM = (15, 40, 100)
DEPTHS_M  = {15: 0.15, 40: 0.40, 100: 1.00}

# CRITICAL depth-mapping correction (verification pass, see
# docs/SENSOR_DEPTH_CORRECTION.md). Sensor blocks appear in the workbook in a
# STABLE order, and soil-thermal physics (diurnal amplitude, single-day curve
# shape, summer/autumn level reversal, rain-infiltration response - all 12
# plots) proves that order is DEEP -> SHALLOW, NOT shallow -> deep. Both prior
# repos (erw_ml, erw_mrv) assigned [15, 40, 100] and therefore had the 15 cm
# and 100 cm sensor labels SWAPPED. Canonical mapping below:
#   block index 0 (1st TEROS block)  -> 100 cm (deepest, most damped)
#   block index 1 (2nd)              ->  40 cm
#   block index 2 (3rd, added later) ->  15 cm (shallowest, largest swing)
# Resin depths (Shallow/Middle/Deep from Sample IDs) are unambiguous and
# UNAFFECTED by this correction.
SENSOR_BLOCK_DEPTH_ORDER = [100, 40, 15]

SENSOR_COLS = {
    "vwc":  [f"vwc_{d}"  for d in DEPTHS_CM],
    "temp": [f"temp_{d}" for d in DEPTHS_CM],
    "ec":   [f"ec_{d}"   for d in DEPTHS_CM],
    "mp":   [f"mp_{d}"   for d in DEPTHS_CM],
}
ALL_SENSOR_COLS: list[str] = sum(SENSOR_COLS.values(), [])

MEASUREMENT_INTERVAL_MIN = 15
STEPS_PER_HOUR = 60 // MEASUREMENT_INTERVAL_MIN
STEPS_PER_DAY  = 24 * STEPS_PER_HOUR

ROLLING_WINDOWS = {"1h": 4, "6h": 24, "24h": 96, "7d": 672, "30d": 2880}

# ---------------------------------------------------------------------------
# UNIBEST PRS resin deployment (Jul 1 -> Oct 16, 2025; 3 rounds, 3 depths)
# ---------------------------------------------------------------------------
RESIN_FILES = {
    "R1_R2": RESIN_DIR / "resin_R1R2.xlsx",
    "R3":    RESIN_DIR / "resin_R3.xlsx",
}

RESIN_DEPTH_MAP = {"Shallow": 15, "Middle": 40, "Deep": 100}

RESIN_ROUND_DATES = {  # (start_inclusive, end_exclusive)
    1: ("2025-07-01", "2025-07-31"),
    2: ("2025-07-31", "2025-08-22"),
    3: ("2025-08-22", "2025-10-16"),
}

RESIN_ANALYTES = [
    "Total N", "NO3-N", "NH4-N", "Al", "B", "Ca", "Cu", "Fe",
    "K", "Mg", "Mn", "Na", "P", "S", "Zn", "pH",
]
RESIN_ANALYTE_COLS = {
    "Total N": "total_n_ppm", "NO3-N": "no3_n_ppm", "NH4-N": "nh4_n_ppm",
    "Al": "al_ppm", "B": "b_ppm", "Ca": "ca_ppm", "Cu": "cu_ppm",
    "Fe": "fe_ppm", "K": "k_ppm", "Mg": "mg_ppm", "Mn": "mn_ppm",
    "Na": "na_ppm", "P": "p_ppm", "S": "s_ppm", "Zn": "zn_ppm", "pH": "ph",
}

# Primary ERW signal channels (validatable against snr_model.py).
# Si NOT in panel: 2M HCl + ion-exchange resin does not liberate silicate Si.
RESIN_PRIMARY_IONS = ["ca_ppm", "mg_ppm"]
RESIN_NEGATIVE_CONTROLS = ["k_ppm", "na_ppm", "no3_n_ppm", "nh4_n_ppm"]
RESIN_SECONDARY_IONS = ["s_ppm", "al_ppm", "mn_ppm", "fe_ppm", "p_ppm",
                        "b_ppm", "cu_ppm", "zn_ppm", "total_n_ppm"]

# Ionic charge (valence) and approximate molar mass (g/mol) for the resin
# analytes - used for the charge-balance / alkalinity budget (C-new-2).
ION_CHARGE = {
    "ca_ppm": +2, "mg_ppm": +2, "k_ppm": +1, "na_ppm": +1,
    "nh4_n_ppm": +1, "no3_n_ppm": -1, "s_ppm": -2, "p_ppm": -1, "al_ppm": +3,
}
ION_MOLAR_MASS = {  # g/mol of the reported element (UNIBEST reports element ppm)
    "ca_ppm": 40.078, "mg_ppm": 24.305, "k_ppm": 39.098, "na_ppm": 22.990,
    "nh4_n_ppm": 14.007, "no3_n_ppm": 14.007, "s_ppm": 32.06,
    "p_ppm": 30.974, "al_ppm": 26.982,
}

# Sample 55303 (4W Deep R1): soil-particle contamination (~6x neighbours).
RESIN_QA_FLAGS = {"55303": "contamination_soil_particle_capsule_4W_deep_R1"}

# ---------------------------------------------------------------------------
# Theoretical SNR model parameters (50:50 wollastonite + diopside)
# ---------------------------------------------------------------------------
SNR_MODEL = {
    "mw_wollastonite_g_mol": 116.16,
    "mw_diopside_g_mol":     216.55,
    "f_rate_wollastonite":   1.0 / 3.0,
    "f_rate_diopside":       1.0 / 30.0,
    "q_drainage_m_yr":       0.30,
    "sigma_F_mol_m2_yr":     {"Ca": 0.12, "Mg": 0.03, "Si": 0.08},
    "k_retention_per_m":     {"Ca": 0.80, "Mg": 0.50, "Si": 0.60},
}

# ---------------------------------------------------------------------------
# Cross-validation policy (from the verification pass)
# ---------------------------------------------------------------------------
# Default spatial holdout unit is the PHYSICAL PLOT (plot_id), not plot_half,
# so the sibling W/E half of a held-out plot never leaks into training.
CV_GROUP_COL = "plot_id"

# Weather coverage policy: sensor rows after the last available weather
# timestamp are retained in the sensor cache but EXCLUDED from any
# sensor+weather join, and this exclusion is logged/recorded - never silent.
WEATHER_JOIN_POLICY = "drop_unmatched_logged"
