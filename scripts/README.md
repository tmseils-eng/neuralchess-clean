# scripts/

| file | what it does |
| --- | --- |
| `run_tests.py` | Runs the test suite with nothing installed but NumPy. `pytest` also works. |
| `chtc/train.sub` | HTCondor submit file for UW-Madison CHTC; fans out one independent run per seed. |
| `chtc/run_train.sh` | Job wrapper: installs NumPy, pins BLAS to one thread, resumes from checkpoints. |
| `slurm/train.sbatch` | Single-node SLURM training job; safe to `--requeue` because runs resume. |
| `slurm/arena.sbatch` | Round-robin every fifth checkpoint against the baseline ladder and fit Elo. |

Both schedulers rely on the same property: `neuralchess train` resumes from
`runs/<name>/checkpoints/best.npz`, so a pre-empted or evicted job loses at
most one iteration of work.
