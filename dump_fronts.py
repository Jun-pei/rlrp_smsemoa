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
Shard with --shard/--n_shards to run them in parallel.

Output: one .npz per cell under <outdir>/, plus <outdir>/reference_<problem>.npz
holding the shared reference set, and <outdir>/verify.csv.

    F_raw    (n, m)  final non-dominated objective vectors, raw units
    F_frame  (n, m)  the same, normalised into the shared evaluation frame
    Zhat     (N, m)  reference set in that frame  (in reference_<problem>.npz)
    zext     (m,)    external HV reference point, (1 + kappa) * 1
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd

PROBS = ["dtlz1", "dtlz2", "minus-dtlz1", "minus-dtlz2", "wfg4", "wfg9"] + \
        [f"imop{i}" for i in range(1, 9)]
METHOD_NAMES = ["SMS-EMOA_balanced", "d-SMS-EMOA", "R2-EMOA", "RL-RP-SMS-EMOA"]


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
    args = ap.parse_args()

    # The repository uses intra-package relative imports, so it has to be
    # imported AS a package -- exactly as run_full_experiment.py does.  The
    # package name is taken from the directory this script sits in, so it
    # works whatever the checkout is called.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    pkg = os.path.basename(here)
    get_problem = importlib.import_module(f"{pkg}.problems").get_problem
    problem_n_obj = importlib.import_module(f"{pkg}.problems").problem_n_obj
    METHODS = importlib.import_module(f"{pkg}.algorithm").METHODS
    get_frame = importlib.import_module(f"{pkg}.experiment").get_frame

    results = pd.read_csv(args.results)
    work = build_worklist(results, args.mode)
    work = [w for i, w in enumerate(work) if i % args.n_shards == args.shard]
    os.makedirs(args.outdir, exist_ok=True)
    print(f"{len(work)} runs on shard {args.shard}/{args.n_shards}",
          file=sys.stderr)

    rows, bad = [], 0
    for i, (p, meth, seed, tag) in enumerate(work, 1):
        m_eff = problem_n_obj(p, 3)
        frame = get_frame(p, m=m_eff)

        ref_path = os.path.join(args.outdir, f"reference_{p}.npz")
        if not os.path.exists(ref_path):
            np.savez_compressed(ref_path, Zhat=frame.Zhat, zext=frame.zext,
                                ideal=frame.ideal, nadir=frame.nadir,
                                m=m_eff, kappa=frame.kappa)

        t0 = time.time()
        # eval_every is set past t_max so the per-generation external
        # indicators (~83% of runtime, per the SLURM sizing notes) are skipped.
        # The final generation is always evaluated, and only the final front is
        # wanted here, so this changes nothing about the result.
        _X, F, _hist = METHODS[meth](get_problem(p, m=m_eff), t_max=args.t_max,
                                     seed=seed, mu=args.mu, frame=frame,
                                     eval_every=args.eval_every)
        dt = time.time() - t0

        hvr_new = frame.hvr(F)
        rec = results[(results.problem == p) & (results.method == meth) &
                      (results.seed == seed)]
        hvr_old = float(rec.hvr_final.iloc[0]) if len(rec) else np.nan
        delta = abs(hvr_new - hvr_old) if np.isfinite(hvr_old) else np.nan
        ok = (not np.isfinite(delta)) or delta <= args.tol
        if not ok:
            bad += 1

        np.savez_compressed(
            os.path.join(args.outdir, f"{p}__{meth}__seed{seed:03d}.npz"),
            F_raw=F, F_frame=frame.to_frame(F), problem=p, method=meth,
            seed=seed, tag=tag, m=m_eff, hvr=hvr_new)

        rows.append(dict(problem=p, method=meth, seed=seed, tag=tag,
                         hvr_regenerated=hvr_new, hvr_recorded=hvr_old,
                         abs_delta=delta, reproduced=ok, seconds=dt))
        print(f"[{i}/{len(work)}] {p:12s} {meth:18s} seed{seed:03d} "
              f"HVR {hvr_new:.6f} (rec {hvr_old:.6f}, d={delta:.2e}) "
              f"{'OK' if ok else '** MISMATCH **'}  {dt/60:.1f} min",
              file=sys.stderr, flush=True)

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
