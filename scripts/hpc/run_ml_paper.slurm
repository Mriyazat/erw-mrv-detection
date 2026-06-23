#!/bin/bash
#SBATCH --job-name=erw_ml_paper
#SBATCH --account=def-erangauk-ab
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ml_paper_%j.out
#
# Companion ML paper: GPU/foundation-model tier for items 6-10.
# Run as a batch job (sbatch) OR copy/paste the python lines into an
# interactive H100 terminal (salloc --gres=gpu:h100:1 ...).
#
# Prereqs (once): caches pushed (scripts/hpc/sync.sh push-data) and the
# foundation-model checkpoints present in ~/hf_models:
#   amazon/chronos-2, amazon/chronos-bolt-base,
#   google/timesfm-2.0-500m-pytorch, Salesforce/moirai-1.1-R-small,
#   time-series-foundation-models/Lag-Llama (lag-llama.ckpt),
#   automl/tabpfn-time-series  (optional models degrade gracefully if absent)

set -euo pipefail
PROJ="${HOME}/links/projects/def-erangauk-ab/${USER}/erw"
cd "${PROJ}"
mkdir -p logs
source scripts/hpc/activate_env.sh
python scripts/hpc/make_manifest.py --check

# item 6: full zero-shot foundation bake-off (6/24/72 h)
python scripts/110_ml_foundation_panel.py --gpu \
    --models chronos,chronos_bolt,timesfm,moirai,lagllama,tabpfn \
    --horizons 6,24,72

# item 7: site-specific training vs zero-shot
python scripts/111_ml_finetune_vs_zeroshot.py --gpu --horizon 24

# item 8: gap-filling (classical + Chronos imputation)
python scripts/112_ml_imputation.py --gpu

# item 9: cross-arm and cross-depth transfer
python scripts/113_ml_transfer.py --gpu --horizon 24

# item 10: self-supervised encoder + LOPO probe
python scripts/114_ml_ssl_probe.py --gpu --epochs 30

echo "ML-paper GPU tier complete."
echo "Pull to Mac:  scripts/hpc/sync.sh pull"
echo "Then locally: python scripts/115_ml_foundation_leaderboard.py"
