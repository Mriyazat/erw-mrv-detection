
set -euo pipefail

PROJ="${HOME}/links/projects/def-erangauk-ab/${USER}/erw"
cd "${PROJ}"


source scripts/hpc/activate_env.sh

# fail fast if the transferred caches are corrupt
python scripts/hpc/make_manifest.py --check

# deep + foundation models -> outputs/results/cnew_deep_ts_gpu.csv
# (Moirai/uni2ts omitted: its scipy~=1.11.3 pin is unsatisfiable on Alliance.)
#
# Seed-averaging (3 seeds) gives the deep models error bars; the horizon sweep
# (6/24/72 h) shows where they beat persistence. Foundation models are
# deterministic so they run once per horizon. Runtime ~ 3 seeds x 3 horizons;
# budget the 8 h wall time. Drop to --seeds 42 --horizons 24 for a quick rerun.
python scripts/63_cnew_deep_ts.py --gpu \
    --models patchtst,itransformer,nhits,tft,chronos \
    --seeds 42,43,44 \
    --horizons 6,24,72

# Experiment 9: covariate-informed + multivariate Chronos-2 (needs chronos-2 cached).
# Tests whether known-future weather covariates extend skill to 24-72 h.
python scripts/74_cnew_covariate_ts.py --horizons 24,72 \
    --modes univariate,covariate,multivariate

echo "GPU deep-TS tier complete; pull with scripts/hpc/sync.sh pull"
echo "Then on the Mac: python scripts/65_deep_ts_leaderboard.py"
