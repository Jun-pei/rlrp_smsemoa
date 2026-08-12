"""
run_full_experiment.py
======================
Comparison protocol (Sec. 5.3): 4 methods x n_seeds runs per problem, every
generation of every run written to disk, and the statistics computed on the
Definition D1..D9 indicators of performance.py.

Problem-set shortcuts for ``--problems``:
    all      the whole benchmark: DTLZ1/2, Minus-DTLZ1/2, WFG4/9 + IMOP1..IMOP8
    regular  the six DTLZ/WFG problems
    imop     the complete IMOP1..IMOP8 suite

Examples
--------
Smoke test (a couple of minutes)::

    python3 run_full_experiment.py --problems dtlz2 --n_seeds 3 \\
        --t_max 1000 --mu 30 --outdir results/smoke

Full protocol::

    python3 run_full_experiment.py --problems all --n_seeds 30 \\
        --t_max 100000 --mu 100 --processes 32 --outdir results/full

COST MODEL -- READ BEFORE LAUNCHING
-----------------------------------
This is a STEADY-STATE EMOA: one offspring per generation, so T_max
generations == T_max function evaluations.  Per generation the cost is

  (a) SMS-EMOA elimination: O(|worst front|) exact HV calls   <- always paid
  (b) the RL state (energy, curvature, HV): O(mu^2) distances <- RL-RP only
  (c) HVR + IGD+ + Eratio against the reference set Z         <- eval_every

Every generation is logged unconditionally; ``--eval_every`` thins only (c).
The default of 1 evaluates the external indicators at every generation, which
is what "store all the values" asks for and costs roughly the same again as
the search itself.  Raise it (e.g. 100) only when that is genuinely the
bottleneck -- the large-mu sweep of Sec. 5.3(b) is the realistic case.
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

from rlrp_smsemoa.algorithm import METHODS
from rlrp_smsemoa.experiment import (
    add_time_to_target, get_frame, run_single, summary_row_from_history,
)
from rlrp_smsemoa.history_io import history_path
from rlrp_smsemoa.problems import BENCHMARK, DTLZ_WFG_NAMES, IMOP_SUITE, problem_n_obj
from rlrp_smsemoa.stats import friedman_test, quade_test, wilcoxon_bonferroni

#: (indicator column, higher_is_better)
INDICATORS = [("hvr_final", True), ("igd_plus_final", False),
              ("energy_ratio_final", False), ("hvr_anytime", True)]

SHORTCUTS = {"all": BENCHMARK, "regular": DTLZ_WFG_NAMES, "imop": IMOP_SUITE}


def expand(names):
    if len(names) == 1 and names[0] in SHORTCUTS:
        return list(SHORTCUTS[names[0]])
    return names


def _job(kw):
    return run_single(**kw)


def run_problem(pname, methods, n_seeds, args):
    m_eff = problem_n_obj(pname, args.m)
    get_frame(pname, m=m_eff, n_points=args.ref_points)     # warm the cache
    jobs = [dict(problem_name=pname, method_name=meth, seed=s, t_max=args.t_max,
                 histdir=args.histdir, m=m_eff, mu=args.mu,
                 eval_every=args.eval_every, reward_hv=args.reward_hv,
                 zref_max=args.zref_max, geometry_mode=args.geometry_mode,
                 n_bins_fine=args.n_bins_fine, n_bins_coarse=args.n_bins_coarse,
                 action_every=args.action_every)
            for meth in methods for s in range(n_seeds)]
    rows = []
    if args.skip_existing:
        todo = []
        for j in jobs:
            p = history_path(args.histdir, pname, j["method_name"], j["seed"])
            # A run already on disk is not re-run; its summary row is rebuilt
            # from the stored history so the output stays complete.
            if os.path.exists(p):
                rows.append(summary_row_from_history(p))
            else:
                todo.append(j)
        # stdout, alongside the per-problem header: this is a summary, not the
        # high-volume per-run progress that goes to stderr.
        print(f"  [{pname}] resuming: {len(rows)} of {len(jobs)} runs already "
              f"on disk, {len(todo)} to run", flush=True)
        jobs = todo
    t0 = time.time()
    if args.processes <= 1:
        for j in jobs:
            rows.append(_job(j))
            print(f"  [{pname}] {j['method_name']:20s} seed={j['seed']:<3d} "
                  f"({time.time() - t0:7.1f}s)", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=args.processes) as ex:
            futs = {ex.submit(_job, j): j for j in jobs}
            for fut in as_completed(futs):
                j = futs[fut]
                rows.append(fut.result())
                print(f"  [{pname}] {j['method_name']:20s} seed={j['seed']:<3d} "
                      f"({time.time() - t0:7.1f}s)", file=sys.stderr)
    return pd.DataFrame(rows)


def write_stats(full, problems, methods, outdir):
    with open(os.path.join(outdir, "stats.txt"), "w") as fh:
        for pname in problems:
            sub = full[full["problem"] == pname]
            if sub.empty:
                continue
            for col, higher_better in INDICATORS:
                try:
                    piv = sub.pivot(index="seed", columns="method", values=col)
                except ValueError:
                    continue
                cols = [c for c in methods if c in piv.columns]
                mat = piv[cols].to_numpy(dtype=float)
                if mat.shape[0] < 3 or not np.isfinite(mat).all():
                    continue
                # every test below is written for HIGHER-IS-BETTER
                signed = mat if higher_better else -mat
                fh.write(f"\n=== {pname} : {col} "
                         f"({'higher' if higher_better else 'lower'} is better) ===\n")
                fr = friedman_test(signed, cols)
                fh.write(f"Friedman: stat={fr['stat']:.4f} p={fr['p']:.4g} "
                         f"ranks={ {k: round(v, 3) for k, v in fr['avg_ranks'].items()} }\n")
                qd = quade_test(signed, cols)
                fh.write(f"Quade:    stat={qd['stat']:.4f} p={qd['p']:.4g} "
                         f"ranks={ {k: round(v, 3) for k, v in qd['avg_ranks'].items()} }\n")
                for pair, r in wilcoxon_bonferroni(signed, cols).items():
                    fh.write(f"  Wilcoxon {pair[0]} vs {pair[1]}: p_adj={r['p_adj']:.4g} "
                             f"effect={r['effect_size']:+.3f} winner={r['winner']}\n")


def _iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 75) - np.percentile(x, 25)) if x.size else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", nargs="+", default=["all"],
                    help="problem names, or one of: all | regular | imop")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--t_max", type=int, default=100_000)
    ap.add_argument("--mu", type=int, default=100)
    ap.add_argument("--m", type=int, default=3, help="ignored for the IMOP suite")
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--ref_points", type=int, default=5000, help="|Z| for IGD+")
    ap.add_argument("--eval_every", type=int, default=1,
                    help="how often HVR/IGD+/Eratio are computed (see COST MODEL); "
                         "every generation is logged regardless")
    ap.add_argument("--action_every", type=int, default=1,
                    help="planner acts every tau generations (RL-RP only)")
    ap.add_argument("--reward_hv", choices=["fixed", "adaptive"], default="fixed")
    ap.add_argument("--zref_max", type=float, default=10.0)
    ap.add_argument("--geometry_mode", choices=["curvature", "contour", "both"],
                    default="curvature")
    ap.add_argument("--n_bins_fine", type=int, default=5)
    ap.add_argument("--n_bins_coarse", type=int, default=3)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--histdir", default=None,
                    help="where per-generation histories go (default <outdir>/history)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="do not re-run a (problem, method, seed) whose history is "
                         "already in --histdir; rebuild its summary row from the "
                         "stored history instead.  Use this to resume after a job "
                         "hit its wall clock.")
    args = ap.parse_args()

    problems = expand(args.problems)
    methods = args.methods or list(METHODS)
    os.makedirs(args.outdir, exist_ok=True)
    if args.histdir is None:
        args.histdir = os.path.join(args.outdir, "history")
    os.makedirs(args.histdir, exist_ok=True)

    all_dfs = []
    for pname in problems:
        m_eff = problem_n_obj(pname, args.m)
        print(f"=== {pname} (m={m_eff}, {args.n_seeds} seeds x {len(methods)} "
              f"methods, T_max={args.t_max}) ===")
        df = run_problem(pname, methods, args.n_seeds, args)
        df.to_csv(os.path.join(args.outdir, f"results_{pname}.csv"), index=False)
        all_dfs.append(df)

    full = add_time_to_target(pd.concat(all_dfs, ignore_index=True))
    full.to_csv(os.path.join(args.outdir, "all_results.csv"), index=False)

    cols = [c for c, _ in INDICATORS] + ["time_to_target"]
    summary = full.groupby(["problem", "method"])[cols].agg(["median", _iqr])
    summary.to_csv(os.path.join(args.outdir, "summary.csv"))
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    write_stats(full, problems, methods, args.outdir)
    print(f"\nWrote results to {args.outdir} (histories in {args.histdir})")


if __name__ == "__main__":
    main()
