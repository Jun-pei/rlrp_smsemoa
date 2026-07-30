"""
run_full_experiment.py
=======================
CLI to reproduce the comparison protocol of Section 5.3 of the proposal:
4 configurations x 30 independent runs per problem, over the 8 problems of
Cuadro 1, plus the sensitivity sweeps of Section 5.3 (W, mu, rho).

IMPORTANT PERFORMANCE NOTE
---------------------------
Cuadro 1 specifies 100,000 *function evaluations* per run. Because this is a
*steady-state* EMOA (one offspring per generation), that is ~100,000
generations. Each generation's SMS-EMOA elimination step calls the exact
hypervolume indicator O(|worst front|) times, and the RL planner's state
computation (g_hat, Riesz energy) costs O(mu^2) for the pairwise distances.
In pure Python/numpy this is too slow to run 8 x 4 x 30 x 100,000 generations
in a chat sandbox -- a single (problem, method, seed) run at mu=100 over
100,000 generations takes on the order of tens of minutes on a laptop, and
the full grid is ~960 such runs. This script is built to run that exact
protocol on a real machine (with --processes > 1 it parallelizes across
seeds); for the sandbox demo, use run_demo.py or pass small --t_max/--mu
overrides here, e.g.:

    python3 run_full_experiment.py --problems dtlz2 --n_seeds 3 \\
        --t_max 1000 --mu 30 --processes 1

For the full Cuadro 1 protocol on your own hardware:

    python3 run_full_experiment.py --problems all --n_seeds 30 \\
        --t_max 100000 --mu 100 --processes 8 --outdir results/

Outputs one CSV per problem (raw per-seed indicator values) plus a combined
``summary.csv`` with means/stds and a ``stats.txt`` with the Wilcoxon
(Bonferroni-corrected), Friedman, and Quade test results requested in
Sec. 5.3.
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from rlrp_smsemoa.problems import PROBLEM_NAMES
from rlrp_smsemoa.experiment import run_single, sample_true_front
from rlrp_smsemoa.stats import wilcoxon_bonferroni, friedman_test, quade_test


def _job(problem_name, method_name, seed, t_max, mu, m, true_front):
    res = run_single(problem_name, method_name, seed, t_max, m=m, mu=mu, true_front=true_front)
    return {k: v for k, v in res.items() if k not in ("X", "F", "history")}


def run_problem(problem_name, methods, n_seeds, t_max, mu, m, processes, true_front_samples):
    true_front = sample_true_front(problem_name, m=m, n_samples=true_front_samples)
    jobs = [(problem_name, method, seed, t_max, mu, m, true_front)
            for method in methods for seed in range(n_seeds)]
    rows = []
    t0 = time.time()
    if processes <= 1:
        for j in jobs:
            rows.append(_job(*j))
            print(f"  [{problem_name}] {j[1]:20s} seed={j[2]:<3d} "
                  f"done ({time.time() - t0:6.1f}s elapsed)", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=processes) as ex:
            futs = {ex.submit(_job, *j): j for j in jobs}
            for fut in as_completed(futs):
                j = futs[fut]
                rows.append(fut.result())
                print(f"  [{problem_name}] {j[1]:20s} seed={j[2]:<3d} "
                      f"done ({time.time() - t0:6.1f}s elapsed)", file=sys.stderr)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", nargs="+", default=["dtlz2"],
                     help=f"Problem names or 'all'. Options: {PROBLEM_NAMES}")
    ap.add_argument("--methods", nargs="+", default=None,
                     help="Subset of methods (default: all 4)")
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--t_max", type=int, default=100_000,
                     help="Generations (~= function evaluations for steady-state EMOA)")
    ap.add_argument("--mu", type=int, default=100)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--true_front_samples", type=int, default=5000)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    problems = PROBLEM_NAMES if args.problems == ["all"] else args.problems
    os.makedirs(args.outdir, exist_ok=True)

    all_dfs = []
    for pname in problems:
        print(f"=== Running {pname} ({args.n_seeds} seeds x "
              f"{len(args.methods) if args.methods else 4} methods, Tmax={args.t_max}) ===")
        df = run_problem(pname, args.methods, args.n_seeds, args.t_max, args.mu, args.m,
                          args.processes, args.true_front_samples)
        df.to_csv(os.path.join(args.outdir, f"results_{pname}.csv"), index=False)
        all_dfs.append(df)

    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv(os.path.join(args.outdir, "all_results.csv"), index=False)

    summary = full.groupby(["problem", "method"])[
        ["hv_ext_far_final", "hv_ext_near_final", "igd_plus_final", "riesz_log_final"]
    ].agg(["mean", "std"])
    summary.to_csv(os.path.join(args.outdir, "summary.csv"))
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # Statistical tests per problem, on the primary indicator (HV, far PR)
    with open(os.path.join(args.outdir, "stats.txt"), "w") as fh:
        for pname in problems:
            sub = full[full["problem"] == pname]
            methods = sorted(sub["method"].unique())
            mat = sub.pivot(index="seed", columns="method", values="hv_ext_far_final")[methods].values
            fh.write(f"\n=== {pname} : HV (external, far PR) ===\n")
            fr = friedman_test(mat, methods)
            fh.write(f"Friedman: stat={fr['stat']:.4f} p={fr['p']:.4g} ranks={fr['avg_ranks']}\n")
            qd = quade_test(mat, methods)
            fh.write(f"Quade:    stat={qd['stat']:.4f} p={qd['p']:.4g} ranks={qd['avg_ranks']}\n")
            wb = wilcoxon_bonferroni(mat, methods)
            for pair, r in wb.items():
                fh.write(f"  Wilcoxon {pair}: p_adj={r['p_adj']:.4g} sig={r['significant']}\n")
    print(f"\nWrote results to {args.outdir}")


if __name__ == "__main__":
    main()
