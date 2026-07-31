"""
run_demo.py
===========
Small end-to-end smoke test: the 4 methods on one problem, a handful of seeds,
short runs.  Prints the Definition-D1..D4 indicators and the realised coverage
of the Q-table (see the sample-complexity warning in state.py).

    python3 run_demo.py --problem imop8 --n_seeds 3 --t_max 800 --mu 30
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, time
import pandas as pd
from rlrp_smsemoa.experiment import run_comparison
from rlrp_smsemoa.problems import problem_n_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="dtlz2")
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--t_max", type=int, default=800)
    ap.add_argument("--mu", type=int, default=30)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--histdir", default=None)
    a = ap.parse_args()

    t0 = time.time()
    df = run_comparison(a.problem, n_seeds=a.n_seeds, t_max=a.t_max, mu=a.mu,
                        eval_every=a.eval_every, histdir=a.histdir)
    cols = ["hvr_final", "igd_plus_final", "energy_ratio_final",
            "hvr_anytime", "zref_mean_final"]
    print(f"\n=== {a.problem} (m={problem_n_obj(a.problem)}), "
          f"{a.n_seeds} seeds, T_max={a.t_max}, mu={a.mu} "
          f"[{time.time()-t0:.0f}s] ===")
    print(df.groupby("method")[cols].median().round(4).to_string())

    q = df[df.method == "RL-RP-SMS-EMOA"]
    qc = [c for c in q.columns if c.startswith("qcov_")]
    if qc:
        print("\nCobertura de la tabla Q (RL-RP):")
        print(q[qc].median().round(4).to_string())


if __name__ == "__main__":
    main()
