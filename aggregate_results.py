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

If a task was killed before it finished -- a wall-clock timeout, most likely --
it never wrote its ``results_<problem>.csv``, even though the runs it did
complete are safely on disk as per-generation histories.  ``--histdir`` covers
that case: every summary row is rebuilt from the histories themselves, so
nothing has to be recomputed and no completed run is lost.

    python3 aggregate_results.py --histdir results/full/history \\
        --outdir results/full
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import glob
import os

import pandas as pd

from rlrp_smsemoa.algorithm import METHODS
from rlrp_smsemoa.experiment import (
    add_time_to_target, history_is_complete, load_summary_frame,
    summary_row_from_history,
)
from rlrp_smsemoa.run_full_experiment import INDICATORS, _iqr, write_stats


def _from_summaries(root: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(root, "**", "results_*.csv"),
                             recursive=True))
    if not paths:
        raise SystemExit(f"no results_*.csv under {root} "
                         f"(use --histdir to rebuild from the histories instead)")
    print(f"merging {len(paths)} summary file(s)")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _from_histories(histdir: str, curves: dict) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(histdir, "*", "*", "seed*.csv.gz")))
    if not paths:
        raise SystemExit(f"no seed*.csv.gz under {histdir}")
    print(f"rebuilding {len(paths)} run(s) from their histories")
    rows, bad = [], []
    for i, p in enumerate(paths, 1):
        # Read each history ONCE, and only the columns a summary needs; the
        # hvr curve is kept so Definition D5 does not re-read every file.
        try:
            df = load_summary_frame(p)
        except Exception:
            df = None
        # An interrupted job can leave a truncated history behind; skip it
        # loudly rather than let it poison the aggregate.
        if df is None or not history_is_complete(p, df):
            bad.append(p)
        else:
            row = summary_row_from_history(p, df)
            rows.append(row)
            if "hvr" in df:
                curves[(row["problem"], row["method"], row["seed"])] = \
                    df["hvr"].to_numpy()
        if i % 100 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    if bad:
        print(f"  SKIPPED {len(bad)} incomplete histor{'y' if len(bad) == 1 else 'ies'}:")
        for p in bad[:10]:
            print(f"    {p}")
        print("  re-run those with run_full_experiment.py --skip_existing")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root",
                    help="directory searched recursively for results_*.csv")
    ap.add_argument("--histdir",
                    help="rebuild every summary row from the per-generation "
                         "histories under this directory (use after a timeout)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--methods", nargs="+", default=None)
    a = ap.parse_args()

    if not a.root and not a.histdir:
        raise SystemExit("give --root, --histdir, or both")
    curves: dict = {}
    full = (_from_histories(a.histdir, curves) if a.histdir
            else _from_summaries(a.root))
    dup = full.duplicated(subset=["problem", "method", "seed"]).sum()
    if dup:
        print(f"WARNING: {dup} duplicated (problem, method, seed) rows -- "
              f"keeping the first of each")
        full = full.drop_duplicates(subset=["problem", "method", "seed"])

    os.makedirs(a.outdir, exist_ok=True)
    full = add_time_to_target(full, hvr_curves=curves)
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
