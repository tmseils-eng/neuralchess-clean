#!/bin/bash
#SBATCH --job-name=neuralchess
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
# Uncomment to train the PyTorch backend on a GPU node:
##SBATCH --gres=gpu:1
#
#   sbatch scripts/slurm/train.sbatch configs/cluster.json
#
# The run resumes from its own checkpoints, so requeue-on-preemption is safe:
#   sbatch --requeue scripts/slurm/train.sbatch
set -euo pipefail

CONFIG="${1:-configs/cluster.json}"
module load python/3.11 2>/dev/null || true

python3 -m venv "${SLURM_TMPDIR:-/tmp}/venv"
source "${SLURM_TMPDIR:-/tmp}/venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -e .

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun python3 -m neuralchess train \
  --config "${CONFIG}" \
  --run-name "slurm_${SLURM_JOB_ID}" \
  --output-dir runs \
  --workers "${SLURM_CPUS_PER_TASK}"
