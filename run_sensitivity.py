"""
run_sensitivity.py
==================
Sensitivity sweeps over the planner's hyper-parameters (Sec. 5.3 a-c), scored
on the Definition D1/D4 indicators.  Only RL-RP-SMS-EMOA is run: the baselines
do not depend on W, rho, sigma0 or the Q-learning parameters.

    python3 run_sensitivity.py --problem imop8 --param rho --values 0.5 0.7 0.9

The three sweeps the proposal commits to::

    --param W    --values 10 20 30 50        (Sec. 5.3 a)
    --param mu   --values 91 100 153 210     (Sec. 5.3 b)
    --param rho  --values 0.5 0.7 0.9        (Sec. 5.3 c)

As everywhere else, every generation of every run is written to ``--histdir``.
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import pandas as pd

from rlrp_smsemoa.experiment import get_frame, run_single
from rlrp_smsemoa.problems import problem_n_obj

SWEEPABLE = ["W", "rho", "mu", "H", "alpha_rl", "gamma", "alpha_reward",
             "eps0", "zref_max", "n_bins_fine", "n_bins_coarse",
             "q_reset_every", "lam", "action_every", "sigma_floor"]
INT_PARAMS = {"W", "mu", "H", "n_bins_fine", "n_bins_coarse", "q_reset_every",
              "action_every"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="dtlz2")
    ap.add_argument("--param", required=True, choices=SWEEPABLE)
    ap.add_argument("--values", nargs="+", type=float, required=True)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--t_max", type=int, default=5000)
    ap.add_argument("--mu", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--outdir", default="results/sensitivity")
    ap.add_argument("--histdir", default=None,
                    help="default <outdir>/history")
    a = ap.parse_args()

    m = problem_n_obj(a.problem)
    frame = get_frame(a.problem, m=m)
    os.makedirs(a.outdir, exist_ok=True)
    histdir = a.histdir or os.path.join(a.outdir, "history")

    rows = []
    for v in a.values:
        val = int(v) if a.param in INT_PARAMS else float(v)
        kw = {a.param: val}
        mu = kw.pop("mu", a.mu)
        for seed in range(a.n_seeds):
            r = run_single(a.problem, "RL-RP-SMS-EMOA", seed, a.t_max,
                           os.path.join(histdir, f"{a.param}={val}"),
                           m=m, mu=mu, frame=frame, eval_every=a.eval_every, **kw)
            r[a.param] = val
            rows.append(r)
        print(f"  {a.param}={val} done", flush=True)

    df = pd.DataFrame(rows)
    path = os.path.join(a.outdir, f"sens_{a.problem}_{a.param}.csv")
    df.to_csv(path, index=False)
    cols = [c for c in ["hvr_final", "igd_plus_final", "hvr_anytime",
                        "zref_mean_final", "qcov_frac_visited", "qcov_frac_blind"]
            if c in df.columns]
    print(f"\n=== {a.problem}: sensibilidad a {a.param} ===")
    print(df.groupby(a.param)[cols].median().round(4).to_string())
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
