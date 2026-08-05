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
# Sized from ixachi as observed on 2026-08:
#   gold5320  nodo[1-8]              52 cores, 256 GB, TIMELIMIT 60 days
#   ixachi    nodo[1-8] + graphos    mixed CPU/GPU nodes
#   gpu       graphos[1-4]           GPU nodes
#
# gold5320 is chosen because it is CPU-only: the `ixachi` partition also spans
# the graphos GPU nodes, and this workload is pure CPU (numpy / pymoo) with no
# GPU code path, so landing on one would occupy an accelerator for exactly
# zero speedup.  Override at submission time if needed:
#     sbatch --partition=<other> --array=0-13 slurm/run_array.sbatch
#
# Half a node per task (26 of 52 cores) so two tasks can share a node and the
# array starts draining without waiting for whole nodes to free up.
# Measured on ixachi: 70 min for one RL-RP run on DTLZ2 at mu=100,
# T_max=100000, eval_every=1.  120 runs per task / 26 cores ~= 5.4 h there,
# and WFG4/WFG9 cost ~1.35x that, so ~7.5 h worst case.  16 h is the margin,
# far inside the 60-day partition limit.
# 2 GB/core is comfortable against the measured ~400 MB peak per run, and well
# inside the 256 GB / 52 cores ~= 4.9 GB per core the nodes actually have.
# ---------------------------------------------------------------------------
#SBATCH --job-name=rlrp
#SBATCH --array=0-13
#SBATCH --partition=gold5320
#SBATCH --cpus-per-task=26
#SBATCH --mem-per-cpu=2G
#SBATCH --time=16:00:00
#SBATCH --output=logs/rlrp_%A_%a.out
#SBATCH --error=logs/rlrp_%A_%a.err

set -euo pipefail

PROBLEMS=(dtlz1 dtlz2 minus-dtlz1 minus-dtlz2 wfg4 wfg9 \
          imop1 imop2 imop3 imop4 imop5 imop6 imop7 imop8)
PROBLEM=${PROBLEMS[$SLURM_ARRAY_TASK_ID]}

# The default system Python is 3.9, and pymoo 0.6.1.6 requires >= 3.10.
module load conda/conda-2026
source activate rlrp

# The parallelism here is one process per run, so every library must stay
# single-threaded: otherwise 26 concurrent processes each ask OpenBLAS for as
# many threads as the node has cores.  The arrays in this workload are small
# enough (mu x m) that BLAS would rarely thread anyway, so this costs nothing
# and removes the possibility of a 26x52 oversubscription.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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
