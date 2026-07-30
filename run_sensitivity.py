"""
run_sensitivity.py
==================
Sensitivity analyses described in Section 5.3 of the proposal:

  (a) W (sliding-window size) in {10, 20, 30, 50}
  (b) mu (population size) in {91, 100, 153, 210}
       corresponding to Das-Dennis H in {12, 16, 20} for m=3
  (c) rho (adaptation fraction) in {0.5, 0.7, 0.9}

For each sweep we fix the problem to DTLZ2 by default (all 8 problems can be
run by passing --problems all) and run RL-RP-SMS-EMOA under the default
hyperparameters except for the one being varied, using n_seeds independent
replicates.

Outputs one CSV per sweep (e.g. sensitivity_W.csv) and a summary figure.

Usage (fast, for checking):
    python3 run_sensitivity.py --problems dtlz2 --n_seeds 3 \
        --t_max 500 --mu 20 --sweep W

Usage (full):
    python3 run_sensitivity.py --problems all --n_seeds 30 \
        --t_max 100000 --sweep all
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rlrp_smsemoa.problems import get_problem, PROBLEM_NAMES
from rlrp_smsemoa.experiment import run_single, sample_true_front


# Das-Dennis H values -> mu values for m=3 (Sec. 2.2)
_MU_SWEEP = [91, 100, 153, 210]
_W_SWEEP = [10, 20, 30, 50]
_RHO_SWEEP = [0.5, 0.7, 0.9]


def _run_sweep(problem_name, sweep_param, sweep_values, base_kwargs,
               n_seeds, t_max, mu, m, true_front, seeds=None):
    """Run RL-RP-SMS-EMOA varying `sweep_param` over `sweep_values`."""
    rows = []
    seeds = seeds if seeds is not None else list(range(n_seeds))
    for val in sweep_values:
        kw = dict(base_kwargs)
        kw[sweep_param] = val
        # mu is both a constructor param AND controls n_var-related things
        _mu = val if sweep_param == "mu" else mu
        for seed in seeds:
            res = run_single(problem_name, "RL-RP-SMS-EMOA", seed, t_max, m=m, mu=_mu,
                             true_front=true_front, **kw)
            rows.append({
                "problem": problem_name,
                "sweep_param": sweep_param,
                "sweep_value": val,
                "seed": seed,
                "hv_ext_far_final": res["hv_ext_far_final"],
                "hv_ext_near_final": res["hv_ext_near_final"],
                "igd_plus_final": res["igd_plus_final"],
                "riesz_log_final": res["riesz_log_final"],
            })
            print(f"  {problem_name}  {sweep_param}={val}  seed={seed}  "
                  f"HV_far={res['hv_ext_far_final']:.4f}")
    return pd.DataFrame(rows)


def _plot_sweep(df, sweep_param, indicator, outdir):
    """Box plot of an indicator vs sweep values, one line per problem."""
    problems = df["problem"].unique()
    sweep_values = sorted(df["sweep_value"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(problems)))
    for prob, color in zip(problems, palette):
        sub = df[df["problem"] == prob]
        means = [sub[sub["sweep_value"] == v][indicator].mean() for v in sweep_values]
        stes = [sub[sub["sweep_value"] == v][indicator].std() /
                max(np.sqrt(len(sub[sub["sweep_value"] == v])), 1) for v in sweep_values]
        ax.errorbar([str(v) for v in sweep_values], means, yerr=stes,
                    label=prob, color=color, marker="o")
    ax.set_xlabel(sweep_param)
    ax.set_ylabel(f"{indicator} (mean ± SE)")
    ax.set_title(f"Sensitivity: {indicator} vs {sweep_param}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = os.path.join(outdir, f"sensitivity_{sweep_param}_{indicator}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=["dtlz2"],
                     help=f"Problem(s) or 'all'. Options: {PROBLEM_NAMES}")
    ap.add_argument("--sweep", nargs="+", default=["W"],
                     help="Which sweeps to run: W | mu | rho | all")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--t_max", type=int, default=500)
    ap.add_argument("--mu", type=int, default=30,
                     help="Base mu (overridden during the mu sweep)")
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--true_front_samples", type=int, default=2000)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    problems = PROBLEM_NAMES if args.problems == ["all"] else args.problems
    sweeps = ["W", "mu", "rho"] if args.sweep == ["all"] else args.sweep
    os.makedirs(args.outdir, exist_ok=True)

    sweep_config = {
        "W":   (_W_SWEEP,   {}),
        "mu":  (_MU_SWEEP,  {"mu": None}),   # override mu handled in _run_sweep
        "rho": (_RHO_SWEEP, {}),
    }

    all_frames = {}
    for sweep_name in sweeps:
        sweep_values, extra_base = sweep_config[sweep_name]
        frames = []
        for pname in problems:
            tf = sample_true_front(pname, m=args.m, n_samples=args.true_front_samples)
            df = _run_sweep(pname, sweep_name, sweep_values,
                             base_kwargs=extra_base, n_seeds=args.n_seeds,
                             t_max=args.t_max, mu=args.mu, m=args.m, true_front=tf)
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        csv_path = os.path.join(args.outdir, f"sensitivity_{sweep_name}.csv")
        combined.to_csv(csv_path, index=False)
        print(f"[{sweep_name}] CSV saved to {csv_path}")
        all_frames[sweep_name] = combined

        for ind in ["hv_ext_far_final", "igd_plus_final", "riesz_log_final"]:
            ppath = _plot_sweep(combined, sweep_name, ind, args.outdir)
            print(f"[{sweep_name}] Plot saved to {ppath}")

        # Summary table
        summary = combined.groupby(["problem", "sweep_value"])[
            ["hv_ext_far_final", "igd_plus_final", "riesz_log_final"]
        ].agg(["mean", "std"])
        print(summary.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
