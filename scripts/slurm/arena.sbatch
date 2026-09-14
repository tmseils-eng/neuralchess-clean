#!/bin/bash
#SBATCH --job-name=nc-arena
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#
# Round-robin every checkpoint of a run against the baseline ladder and fit a
# single Elo scale over the whole thing:
#
#   sbatch scripts/slurm/arena.sbatch runs/laptop
set -euo pipefail

RUN_DIR="${1:?usage: arena.sbatch RUN_DIR}"
export OMP_NUM_THREADS=1

CHECKPOINTS=$(ls "${RUN_DIR}"/checkpoints/iter*.npz | awk 'NR % 5 == 1')
AGENTS="random material alphabeta:2"
for ckpt in ${CHECKPOINTS}; do
  AGENTS="${AGENTS} mcts:${ckpt}"
done

python3 -m neuralchess arena ${AGENTS} \
  --games 20 --simulations 200 --anchor random \
  --pgn "${RUN_DIR}/ladder.pgn"
