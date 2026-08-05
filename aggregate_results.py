"""
aggregate_results.py
====================
Merge the per-problem result files produced by independent
``run_full_experiment.py`` invocations -- typically one scheduler array task
per problem -- into the single set of tables the protocol reports.

    python3 aggregate_results.py --root results/full --outdir results/full

``--root`` is searched recursively for ``results_<problem>.csv``.  Time to
target (Definition D5) is recomputed here rather than reused, because it is
defined relative to the best HVR reached by ANY method on that (problem, seed)
and each array task only ever saw its own problem.
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import glob
import os

import pandas as pd

from rlrp_smsemoa.algorithm import METHODS
from rlrp_smsemoa.experiment import add_time_to_target
from rlrp_smsemoa.run_full_experiment import INDICATORS, _iqr, write_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="directory searched recursively for results_*.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--methods", nargs="+", default=None)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.root, "**", "results_*.csv"),
                             recursive=True))
    if not paths:
        raise SystemExit(f"no results_*.csv under {a.root}")
    print(f"merging {len(paths)} file(s)")

    full = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    dup = full.duplicated(subset=["problem", "method", "seed"]).sum()
    if dup:
        print(f"WARNING: {dup} duplicated (problem, method, seed) rows -- "
              f"keeping the first of each")
        full = full.drop_duplicates(subset=["problem", "method", "seed"])

    os.makedirs(a.outdir, exist_ok=True)
    full = add_time_to_target(full)
    full.to_csv(os.path.join(a.outdir, "all_results.csv"), index=False)

    cols = [c for c, _ in INDICATORS] + ["time_to_target"]
    summary = full.groupby(["problem", "method"])[cols].agg(["median", _iqr])
    summary.to_csv(os.path.join(a.outdir, "summary.csv"))
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    problems = sorted(full["problem"].unique())
    methods = a.methods or [m for m in METHODS if m in set(full["method"])]
    write_stats(full, problems, methods, a.outdir)

    n = full.groupby(["problem", "method"]).size()
    print(f"\n{len(problems)} problems x {len(methods)} methods, "
          f"{n.min()}-{n.max()} seeds each -> {a.outdir}")


if __name__ == "__main__":
    main()
