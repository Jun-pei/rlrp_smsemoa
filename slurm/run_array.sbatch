#!/bin/bash
# ---------------------------------------------------------------------------
# Full protocol as a SLURM job array: one array task per (problem, method)
# pair, 30 seeds inside the task.
#
#     mkdir -p logs
#     sbatch --array=0-55 slurm/run_array.sbatch
#
# then, once every task has finished:
#
#     python3 aggregate_results.py --histdir results/full/history \
#         --outdir results/full
#
# The array is IDEMPOTENT: --skip_existing means a resubmission re-runs only
# the (problem, method, seed) combinations whose history is not yet on disk.
# Submit it again after a partial run and it fills in the gaps.
#
# SIZING, measured on ixachi rather than estimated
# ------------------------------------------------
# The first attempt (14 tasks x 26 cpus, one task per problem) showed that
# throughput is governed by how many tasks land on the same node:
#
#     tasks/node   observed h per run
#         1 (solo)      1.17     <- the calibration run, --cpus-per-task=1
#         2             5.3
#         4            16.0
#
# `sinfo` reports 52 CPUS per node, but those are hyperthreads over 26
# physical cores, so two 26-process tasks already put two processes on every
# core.  The worker count is therefore derived from the PHYSICAL cores in the
# allocation (see PROCS below) instead of from SLURM_CPUS_PER_TASK directly:
# with --cpus-per-task=26 and 2 threads/core that is 13 workers, so two tasks
# sharing a node exactly fill its 26 real cores and never oversubscribe.
#
# EVAL_EVERY defaults to 100, which thins ONLY the external indicators
# (HVR / IGD+ / Eratio against the reference set Z) -- about 83% of the
# runtime.  Every generation is still logged in full.  At T_max = 100000 that
# still leaves 1000 points on the anytime curve, measured to shift D4 by
# ~1e-4 (method differences are ~1e-1) and to leave the final indicators
# bit-identical, since the last generation is always evaluated.
# Override with:  EVAL_EVERY=1 sbatch --array=0-55 slurm/run_array.sbatch
#
# This workload is pure CPU: do not send it to a GPU partition.
# ---------------------------------------------------------------------------
#SBATCH --job-name=rlrp
#SBATCH --array=0-55
#SBATCH --partition=gold5320
#SBATCH --cpus-per-task=26
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rlrp_%A_%a.out
#SBATCH --error=logs/rlrp_%A_%a.err

set -euo pipefail

PROBLEMS=(dtlz1 dtlz2 minus-dtlz1 minus-dtlz2 wfg4 wfg9 \
          imop1 imop2 imop3 imop4 imop5 imop6 imop7 imop8)
METHODS=("SMS-EMOA_balanced" "d-SMS-EMOA" "R2-EMOA" "RL-RP-SMS-EMOA")

IDX=${SLURM_ARRAY_TASK_ID}
PROBLEM=${PROBLEMS[$((IDX / ${#METHODS[@]}))]}
METHOD=${METHODS[$((IDX % ${#METHODS[@]}))]}

# The default system Python is 3.9, and pymoo 0.6.1.6 requires >= 3.10.
module load conda/conda-2026
source activate rlrp

# One process per run, so every library must stay single-threaded: otherwise
# each worker asks OpenBLAS for as many threads as the node has cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Locate the repository.  NOT via ${BASH_SOURCE[0]}: sbatch copies the script
# into a spool directory before executing it, so BASH_SOURCE points at
# /var/spool/... and every path derived from it is wrong.  SLURM_SUBMIT_DIR is
# the directory sbatch was invoked from, which is the repository itself.
REPO=${RLRP_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f "$REPO/run_full_experiment.py" ]]; then
    echo "ERROR: '$REPO' is not the rlrp_smsemoa repository." >&2
    echo "Submit from inside it, or set RLRP_REPO=/path/to/rlrp_smsemoa." >&2
    exit 1
fi
# The repository root IS the package, so its PARENT goes on the import path.
export PYTHONPATH="$(dirname "$REPO"):${PYTHONPATH:-}"
cd "$REPO"

# Workers = PHYSICAL cores in the allocation, not hyperthreads.
THREADS_PER_CORE=$(lscpu | awk -F: '/^Thread\(s\) per core/ {gsub(/ /,"",$2); print $2}')
THREADS_PER_CORE=${THREADS_PER_CORE:-1}
PROCS=$(( ${SLURM_CPUS_PER_TASK:-1} / THREADS_PER_CORE ))
(( PROCS < 1 )) && PROCS=1

OUT=results/full
mkdir -p logs "$OUT"

echo "task $IDX -> $PROBLEM / $METHOD on $(hostname)"
echo "  ${SLURM_CPUS_PER_TASK:-?} logical cpus, ${THREADS_PER_CORE} threads/core"\
     "-> ${PROCS} workers; eval_every=${EVAL_EVERY:-100}; started $(date)"

# Per-task outdir so the summary files of different tasks never collide; the
# per-generation histories share one tree, whose paths already carry the
# problem and method, so they cannot collide either.
python3 run_full_experiment.py \
    --problems "$PROBLEM" \
    --methods "$METHOD" \
    --n_seeds 30 \
    --t_max 100000 \
    --mu 100 \
    --processes "$PROCS" \
    --eval_every "${EVAL_EVERY:-100}" \
    --skip_existing \
    --outdir "$OUT/${PROBLEM}__${METHOD}" \
    --histdir "$OUT/history"

echo "task $IDX finished $(date)"
