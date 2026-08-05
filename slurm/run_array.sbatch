#!/bin/bash
# ---------------------------------------------------------------------------
# Full protocol as a SLURM job array: one array task per problem, all methods
# and all seeds inside the task.
#
#     sbatch --array=0-13 slurm/run_array.sbatch
#
# then, once every task has finished:
#
#     python3 aggregate_results.py --root results/full --outdir results/full
#
# CHECK BEFORE THE FIRST SUBMISSION -- these are placeholders, not defaults
# that are known to be right for ixachi:
#   * --partition        : see `sinfo -s`
#   * --cpus-per-task    : do not exceed the cores per node of that partition
#   * --time             : see the partition's TIMELIMIT in `sinfo -s`
#   * the conda module name, if it ever changes
# ---------------------------------------------------------------------------
#SBATCH --job-name=rlrp
#SBATCH --array=0-13
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/rlrp_%A_%a.out
#SBATCH --error=logs/rlrp_%A_%a.err
# #SBATCH --partition=<fill in from `sinfo -s`>

set -euo pipefail

PROBLEMS=(dtlz1 dtlz2 minus-dtlz1 minus-dtlz2 wfg4 wfg9 \
          imop1 imop2 imop3 imop4 imop5 imop6 imop7 imop8)
PROBLEM=${PROBLEMS[$SLURM_ARRAY_TASK_ID]}

# The default system Python is 3.9, and pymoo 0.6.1.6 requires >= 3.10.
module load conda/conda-2026
source activate rlrp

# The repository root IS the package, so its PARENT goes on the import path.
REPO_PARENT=$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)")
export PYTHONPATH="$REPO_PARENT:${PYTHONPATH:-}"
cd "$REPO_PARENT/rlrp_smsemoa"

OUT=results/full
mkdir -p logs "$OUT"

echo "task $SLURM_ARRAY_TASK_ID -> $PROBLEM on $(hostname), \
${SLURM_CPUS_PER_TASK} cpus, started $(date)"

# Per-task outdir so the summary files of different tasks never collide;
# the per-generation histories share one tree (paths already include the
# problem name, so they cannot collide either).
python3 run_full_experiment.py \
    --problems "$PROBLEM" \
    --n_seeds 30 \
    --t_max 100000 \
    --mu 100 \
    --processes "${SLURM_CPUS_PER_TASK}" \
    --eval_every 1 \
    --outdir "$OUT/$PROBLEM" \
    --histdir "$OUT/history"

echo "task $SLURM_ARRAY_TASK_ID finished $(date)"
