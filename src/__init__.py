"""erw - canonical, verified ERW MRV analysis package.

Fresh rebuild of the Simcoe 12-plot wollastonite + diopside trial analysis,
consolidating the two prior repos (erw_ml, erw_mrv) into one tested base.

Design rules baked in from the verification pass:
  * ONE loader per data stream (no legacy column-index loaders).
  * Locked dose convention (t/ha is canonical; kg/m^2 provided alongside).
  * Locked effect-size convention (Hedges' g).
  * Sensor depth mapping is validated, not assumed.
  * Weather-coverage cutoff is an explicit, logged policy.
  * All ML preprocessing fit inside CV folds; plot-level (not plot-half)
    spatial holdout by default.
"""

__version__ = "0.1.0"
