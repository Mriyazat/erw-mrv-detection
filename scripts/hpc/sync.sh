#!/usr/bin/env bash
# Mac <-> Rorqual sync for the erw repo.
#
#   ./sync.sh push        # code + caches  Mac -> Rorqual project space
#   ./sync.sh pull        # results + GPU outputs  Rorqual -> Mac
#   ./sync.sh push-data   # large raw/cache -> scratch
#   ./sync.sh verify      # sha256 manifest check on both sides
#
# rsync over ssh; checksum-verified; never deletes on the remote by default.
set -euo pipefail

REMOTE_USER="mriyazat"
REMOTE_HOST="rorqual.alliancecan.ca"
REMOTE_PROJ="~/links/projects/def-erangauk-ab/${REMOTE_USER}/erw"
REMOTE_SCRATCH="~/links/scratch/erw"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

# code/doc sync excludes heavy + transient artifacts
RSYNC_CODE=(rsync -avz --checksum
  --exclude '.venv' --exclude '__pycache__' --exclude '.git'
  --exclude 'outputs/cache/*.parquet' --exclude 'outputs/figures/*'
  --exclude 'catboost_info' --exclude '*.pyc')

case "${1:-}" in
  push)
    "${RSYNC_CODE[@]}" "${LOCAL_ROOT}/" "${REMOTE}:${REMOTE_PROJ}/"
    ;;
  pull)
    rsync -avz --checksum \
      "${REMOTE}:${REMOTE_PROJ}/outputs/results/" "${LOCAL_ROOT}/outputs/results/"
    rsync -avz --checksum \
      "${REMOTE}:${REMOTE_PROJ}/outputs/audits/" "${LOCAL_ROOT}/outputs/audits/"
    ;;
  push-data)
    "${LOCAL_ROOT}/../.venv/bin/python" "${LOCAL_ROOT}/scripts/hpc/make_manifest.py"
    rsync -avz --checksum --mkpath \
      "${LOCAL_ROOT}/outputs/cache/" "${REMOTE}:${REMOTE_SCRATCH}/cache/"
    ;;
  verify)
    echo "[local]"
    "${LOCAL_ROOT}/../.venv/bin/python" "${LOCAL_ROOT}/scripts/hpc/make_manifest.py" --check || true
    echo "[remote]"
    ssh "${REMOTE}" "cd ${REMOTE_PROJ} && python scripts/hpc/make_manifest.py --check" || true
    ;;
  *)
    echo "usage: $0 {push|pull|push-data|verify}"; exit 1 ;;
esac
echo "done: ${1:-}"
