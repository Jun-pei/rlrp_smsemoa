#!/usr/bin/env python3
"""
dump_fronts.py
==============
Re-runs selected (problem, method, seed) cells and saves the FINAL Pareto front
approximation, which the original experiment computed but never persisted.

Because every method is seeded, a re-run of (problem, method, seed) reproduces
exactly the run behind the corresponding row of all_results.csv.  The script
verifies this: it recomputes HVR from the regenerated front and compares it to
the value recorded in all_results.csv, aborting loudly on any mismatch.

Run from the repository root (the directory holding algorithm.py):

    python3 dump_fronts.py --outdir fronts --results results/full/all_results.csv

Default work list: the median-HVR seed of each of the 14 x 4 = 56 cells, plus
the best and worst RL-RP seed on each problem (28 more) for the variability
panel -- 84 runs.  External per-generation indicators are switched off (they
are ~83% of runtime per the SLURM sizing notes and are not needed for a final
front), so each run is well under the 1.17 h of the original calibration.

Parallelism comes in two independent forms, and they compose:

  --processes N   run N cells at once inside this invocation
  --shard/--n_shards   split the work list across separate jobs

Use --processes for one interactive/batch job on a node, and shards on top of
it to spread across nodes.  As with the main experiment, size --processes by
PHYSICAL cores: the nodes report hyperthreads, and oversubscribing them was
measured to cost a factor of three.

Output: one .npz per cell under <outdir>/, plus <outdir>/reference_<problem>.npz
holding the shared reference set, and <outdir>/verify.csv.

    F_raw    (n, m)  final non-dominated objective vectors, raw units
    F_frame  (n, m)  the same, normalised into the shared evaluation frame

``F_raw`` and ``F_frame`` hold the same points in the same order, so row i of
one is row i of the other.  Both are the non-dominated subset of the final
population: the dominated members are of no interest for a front plot, and
keeping them would make the two arrays differ in length.
    Zhat     (N, m)  reference set in that frame  (in reference_<problem>.npz)
    zext     (m,)    external HV reference point, (1 + kappa) * 1
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

PROBS = ["dtlz1", "dtlz2", "minus-dtlz1", "minus-dtlz2", "wfg4", "wfg9"] + \
        [f"imop{i}" for i in range(1, 9)]
METHOD_NAMES = ["SMS-EMOA_balanced", "d-SMS-EMOA", "R2-EMOA", "RL-RP-SMS-EMOA"]

_API: dict | None = None


def api() -> dict:
    """Import the package lazily, once per process.

    The repository uses intra-package relative imports, so it has to be
    imported AS a package -- exactly as run_full_experiment.py does.  The
    package name is taken from the directory this script sits in, so it works
    whatever the checkout is called.  Worker processes call this themselves.
    """
    global _API
    if _API is None:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.dirname(here))
        pkg = os.path.basename(here)
        _API = dict(
            get_problem=importlib.import_module(f"{pkg}.problems").get_problem,
            problem_n_obj=importlib.import_module(f"{pkg}.problems").problem_n_obj,
            METHODS=importlib.import_module(f"{pkg}.algorithm").METHODS,
            get_frame=importlib.import_module(f"{pkg}.experiment").get_frame,
            nondominated=importlib.import_module(f"{pkg}.indicators").nondominated,
        )
    return _API


def run_cell(job: dict) -> dict:
    """Regenerate one (problem, method, seed) and save its final front.

    Runs in a worker process, so it takes only plain values and returns only
    plain values.  The recorded-HVR comparison is left to the parent, which
    already holds the results table.
    """
    a = api()
    p, meth, seed = job["problem"], job["method"], job["seed"]
    m_eff = a["problem_n_obj"](p, 3)
    frame = a["get_frame"](p, m=m_eff)

    t0 = time.time()
    # eval_every is set past t_max so the per-generation external indicators
    # (~83% of runtime, per the SLURM sizing notes) are skipped.  The final
    # generation is always evaluated, and only the final front is wanted here,
    # so this changes nothing about the result.
    _X, F, _hist = a["METHODS"][meth](a["get_problem"](p, m=m_eff),
                                      t_max=job["t_max"], seed=seed,
                                      mu=job["mu"], frame=frame,
                                      eval_every=job["eval_every"])
    dt = time.time() - t0

    # The methods return the whole final POPULATION, which on a steady-state
    # EMOA still contains dominated members.  Store the non-dominated subset,
    # and derive F_frame from that same subset so the two arrays agree row for
    # row (ND filtering is idempotent, so F_frame is unchanged by this).
    F_nd = F[a["nondominated"](F)] if F.shape[0] > 1 else F
    np.savez_compressed(
        os.path.join(job["outdir"], f"{p}__{meth}__seed{seed:03d}.npz"),
        F_raw=F_nd, F_frame=frame.to_frame(F_nd), problem=p, method=meth,
        seed=seed, tag=job["tag"], m=m_eff, hvr=frame.hvr(F))

    return dict(problem=p, method=meth, seed=seed, tag=job["tag"],
                hvr_regenerated=frame.hvr(F), seconds=dt)


def write_reference(outdir: str, problem: str) -> None:
    """Save a problem's shared reference set.  Done in the PARENT: concurrent
    workers writing the same path would race and could leave it truncated."""
    a = api()
    path = os.path.join(outdir, f"reference_{problem}.npz")
    if os.path.exists(path):
        return
    m_eff = a["problem_n_obj"](problem, 3)
    frame = a["get_frame"](problem, m=m_eff)
    np.savez_compressed(path, Zhat=frame.Zhat, zext=frame.zext,
                        ideal=frame.ideal, nadir=frame.nadir,
                        m=m_eff, kappa=frame.kappa)


def build_worklist(results: pd.DataFrame, mode: str):
    """-> list of (problem, method, seed, tag)"""
    work = []
    if mode in ("median", "both"):
        for p in PROBS:
            for m in METHOD_NAMES:
                s = results[(results.problem == p) & (results.method == m)]
                s = s.sort_values("hvr_final").reset_index(drop=True)
                if len(s) == 0:
                    continue
                work.append((p, m, int(s.iloc[len(s) // 2].seed), "median"))
    if mode in ("extremes", "both"):
        for p in PROBS:
            s = results[(results.problem == p) &
                        (results.method == "RL-RP-SMS-EMOA")]
            s = s.sort_values("hvr_final").reset_index(drop=True)
            if len(s) == 0:
                continue
            work.append((p, "RL-RP-SMS-EMOA", int(s.iloc[0].seed), "worst"))
            work.append((p, "RL-RP-SMS-EMOA", int(s.iloc[-1].seed), "best"))
    if mode == "all":
        for p in PROBS:
            for m in METHOD_NAMES:
                for sd in sorted(results[(results.problem == p) &
                                         (results.method == m)].seed.unique()):
                    work.append((p, m, int(sd), "all"))
    # de-duplicate, preserving order
    seen, out = set(), []
    for w in work:
        if w[:3] not in seen:
            seen.add(w[:3])
            out.append(w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="fronts")
    ap.add_argument("--results", default="results/full/all_results.csv")
    ap.add_argument("--mode", default="both",
                    choices=["median", "extremes", "both", "all"])
    ap.add_argument("--t_max", type=int, default=100000)
    ap.add_argument("--mu", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max allowed |HVR_regenerated - HVR_recorded|")
    ap.add_argument("--eval_every", type=int, default=10**9,
                    help="skip per-generation external indicators; the final "
                         "generation is evaluated regardless")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--processes", type=int, default=1,
                    help="cells to regenerate concurrently; size by PHYSICAL "
                         "cores, not the hyperthread count sinfo reports")
    args = ap.parse_args()

    results = pd.read_csv(args.results)
    work = build_worklist(results, args.mode)
    work = [w for i, w in enumerate(work) if i % args.n_shards == args.shard]
    os.makedirs(args.outdir, exist_ok=True)
    print(f"{len(work)} runs on shard {args.shard}/{args.n_shards}, "
          f"{args.processes} process(es)", file=sys.stderr)

    # Reference sets are written HERE, before any worker starts: several
    # workers writing the same reference_<problem>.npz would race.  This also
    # warms the frame cache, which the workers inherit when the start method
    # is fork.
    for p in sorted({w[0] for w in work}):
        write_reference(args.outdir, p)

    jobs = [dict(problem=p, method=meth, seed=seed, tag=tag,
                 outdir=args.outdir, t_max=args.t_max, mu=args.mu,
                 eval_every=args.eval_every)
            for (p, meth, seed, tag) in work]

    def finish(res: dict, i: int) -> dict:
        """Attach the recorded value and the verdict, in the parent."""
        rec = results[(results.problem == res["problem"]) &
                      (results.method == res["method"]) &
                      (results.seed == res["seed"])]
        hvr_old = float(rec.hvr_final.iloc[0]) if len(rec) else np.nan
        delta = abs(res["hvr_regenerated"] - hvr_old) if np.isfinite(hvr_old) else np.nan
        ok = (not np.isfinite(delta)) or delta <= args.tol
        print(f"[{i}/{len(jobs)}] {res['problem']:12s} {res['method']:18s} "
              f"seed{res['seed']:03d} HVR {res['hvr_regenerated']:.6f} "
              f"(rec {hvr_old:.6f}, d={delta:.2e}) "
              f"{'OK' if ok else '** MISMATCH **'}  {res['seconds']/60:.1f} min",
              file=sys.stderr, flush=True)
        return dict(res, hvr_recorded=hvr_old, abs_delta=delta, reproduced=ok)

    rows = []
    if args.processes <= 1:
        for i, job in enumerate(jobs, 1):
            rows.append(finish(run_cell(job), i))
    else:
        with ProcessPoolExecutor(max_workers=args.processes) as ex:
            futs = [ex.submit(run_cell, job) for job in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                rows.append(finish(fut.result(), i))

    bad = sum(1 for r in rows if not r["reproduced"])
    v = pd.DataFrame(rows)
    vpath = os.path.join(args.outdir, f"verify_shard{args.shard}.csv")
    v.to_csv(vpath, index=False)
    print(f"\nwrote {vpath}", file=sys.stderr)
    if bad:
        print(f"WARNING: {bad} run(s) did not reproduce the recorded HVR. "
              f"Do not use these fronts until that is understood.",
              file=sys.stderr)
        sys.exit(1)
    print("all runs reproduced the recorded HVR", file=sys.stderr)


if __name__ == "__main__":
    main()
