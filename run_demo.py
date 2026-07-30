"""
run_demo.py
===========
Quick, self-contained demonstration of RL-RP-SMS-EMOA vs. the three
baselines (Sec. 5.3) at REDUCED scale (small mu / Tmax) so it finishes in
well under a minute on a laptop. It:

  1. Runs the 4 methods on DTLZ2 and Minus-DTLZ1 (one regular, one inverted
     front) with several seeds each.
  2. Plots HV (external, far PR) convergence curves per method.
  3. Plots sigma(t) (regime-selection probabilities) and the z_ref(t)
     trajectory for RL-RP-SMS-EMOA, to inspect the planner's behaviour
     (Sec. 5.2).
  4. Plots the final 3D non-dominated front of each method.
  5. Prints a small summary table + a Friedman test across methods.

For the full Cuadro 1 / Section 5.3 protocol (8 problems, mu=100,
Tmax up to 100000 evaluations, 30 seeds), use run_full_experiment.py
instead -- see the README for expected runtime.

Usage:
    python3 run_demo.py [--outdir DIR]
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from rlrp_smsemoa.problems import get_problem
from rlrp_smsemoa.algorithm import METHODS
from rlrp_smsemoa.experiment import run_comparison, pivot_indicator, sample_true_front
from rlrp_smsemoa.stats import friedman_test


METHOD_COLORS = {
    "SMS-EMOA_nadir": "#999999",
    "SMS-EMOA_balanced": "#5599DD",
    "SMS-EMOA_dynlin": "#DD8844",
    "RL-RP-SMS-EMOA": "#CC3344",
}


def plot_convergence(problem_name, t_max, mu, seeds, outdir):
    problem = get_problem(problem_name, m=3)
    true_front = sample_true_front(problem_name, m=3, n_samples=2000)

    fig, ax = plt.subplots(figsize=(6, 4))
    for method_name, fn in METHODS.items():
        curves = []
        for seed in seeds:
            _, _, hist = fn(problem, t_max=t_max, seed=seed, mu=mu, true_front=true_front)
            curves.append(hist.hv_ext_far)
        curves = np.array(curves)
        mean_curve = curves.mean(axis=0)
        ax.plot(mean_curve, label=method_name, color=METHOD_COLORS[method_name])
    ax.set_xlabel("Generation t")
    ax.set_ylabel("HV (external PR, far)")
    ax.set_title(f"HV convergence -- {problem_name} (mu={mu}, mean of {len(seeds)} seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, f"convergence_{problem_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_planner_behaviour(problem_name, t_max, mu, seed, outdir):
    problem = get_problem(problem_name, m=3)
    _, _, hist = METHODS["RL-RP-SMS-EMOA"](problem, t_max=t_max, seed=seed, mu=mu)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sigma = np.array(hist.sigma)
    axes[0].stackplot(range(len(sigma)), sigma[:, 0], sigma[:, 1], sigma[:, 2],
                       labels=["fine", "balanced", "coarse"],
                       colors=["#88CCEE", "#44AA99", "#CC6677"])
    axes[0].set_xlabel("Generation t (adaptation phase)")
    axes[0].set_ylabel("sigma(t)")
    axes[0].set_title("Replicator-dynamics regime probabilities")
    axes[0].legend(fontsize=8, loc="upper right")

    zref = np.array(hist.zref)
    for d in range(zref.shape[1]):
        axes[1].plot(zref[:, d], label=f"z_ref dim {d+1}")
    axes[1].set_xlabel("Generation t")
    axes[1].set_ylabel("z_ref (normalized space)")
    axes[1].set_title("Reference-point trajectory")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"RL planner behaviour -- {problem_name}")
    fig.tight_layout()
    path = os.path.join(outdir, f"planner_{problem_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_final_fronts(problem_name, t_max, mu, seed, outdir):
    problem = get_problem(problem_name, m=3)
    fig = plt.figure(figsize=(10, 8))
    for idx, (method_name, fn) in enumerate(METHODS.items(), start=1):
        _, F, _ = fn(problem, t_max=t_max, seed=seed, mu=mu)
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=14, color=METHOD_COLORS[method_name])
        ax.set_title(method_name, fontsize=9)
        ax.set_xlabel("f1"); ax.set_ylabel("f2"); ax.set_zlabel("f3")
    fig.suptitle(f"Final non-dominated fronts -- {problem_name}")
    fig.tight_layout()
    path = os.path.join(outdir, f"fronts_{problem_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def summarize(problem_name, t_max, mu, n_seeds, outdir):
    df = run_comparison(problem_name, n_seeds=n_seeds, t_max=t_max, mu=mu,
                         true_front_samples=2000)
    print(f"\n=== {problem_name} (mu={mu}, Tmax={t_max}, {n_seeds} seeds) ===")
    summary = df.groupby("method")[["hv_ext_far_final", "igd_plus_final", "riesz_log_final"]].agg(["mean", "std"])
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    mat, methods = pivot_indicator(df, "hv_ext_far_final")
    fr = friedman_test(mat, methods)
    print(f"\nFriedman test on HV(far) across methods: stat={fr['stat']:.3f}, p={fr['p']:.4g}")
    print("Average ranks (lower = worse, higher = better, since HV is higher-is-better):")
    for m, r in fr["avg_ranks"].items():
        print(f"  {m:20s} {r:.2f}")
    df.to_csv(os.path.join(outdir, f"results_{problem_name}.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--mu", type=int, default=30)
    ap.add_argument("--t_max", type=int, default=400)
    ap.add_argument("--n_seeds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    problems_to_demo = ["dtlz2", "minus-dtlz1"]

    for pname in problems_to_demo:
        summarize(pname, args.t_max, args.mu, args.n_seeds, args.outdir)
        p1 = plot_convergence(pname, args.t_max, args.mu, list(range(args.n_seeds)), args.outdir)
        p2 = plot_planner_behaviour(pname, args.t_max, args.mu, seed=0, outdir=args.outdir)
        p3 = plot_final_fronts(pname, args.t_max, args.mu, seed=0, outdir=args.outdir)
        print(f"Saved: {p1}\n       {p2}\n       {p3}")


if __name__ == "__main__":
    main()
