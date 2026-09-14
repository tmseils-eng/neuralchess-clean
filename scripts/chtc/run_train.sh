#!/bin/bash
# Executable wrapper used by scripts/chtc/train.sub.
#
#   $1 = condor cluster id   $2 = process id   $3 = config file
set -euo pipefail

CLUSTER="${1:-0}"
PROCESS="${2:-0}"
CONFIG="${3:-configs/cluster.json}"

echo "host      : $(hostname)"
echo "cpus      : $(nproc)"
echo "python    : $(python3 --version)"
echo "config    : ${CONFIG}"
echo "run seed  : ${PROCESS}"

python3 -m pip install --quiet --no-cache-dir numpy

# One CPU thread per NumPy process: the parallelism comes from the self-play
# workers, and letting BLAS also thread oversubscribes the slot badly.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 -m neuralchess train \
  --config "${CONFIG}" \
  --run-name "chtc_${CLUSTER}_${PROCESS}" \
  --output-dir runs \
  --workers "$(nproc)" \
  --seed "${PROCESS}"

echo "finished; artefacts in runs/chtc_${CLUSTER}_${PROCESS}"
