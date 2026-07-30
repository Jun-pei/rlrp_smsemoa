"""
experiment.py
=============
Glue code to reproduce the comparison protocol of Section 5.3:
4 methods x N independent seeds on a given problem, collecting the final
HV (both external PRs), IGD+, and Riesz energy, ready to feed into stats.py.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .problems import get_problem, PROBLEM_NAMES
from .algorithm import METHODS
from .indicators import estimate_ideal_nadir, normalize


def sample_true_front(problem_name: str, m: int = 3, n_samples: int = 5000, seed: int = 12345):
    """Best-effort sampler of (an approximation to) the true Pareto front,
    used only for IGD+. For DTLZ/WFG/Minus-DTLZ this samples the decision
    space directly on the g=0 manifold where the construction allows it
    (g=0 -> distance variables fixed at their optimal value); for the
    IMOP-like problems it falls back to large-N random sampling + a
    non-dominated filter (their fronts are not analytically known to us).
    """
    rng = np.random.default_rng(seed)
    problem = get_problem(problem_name, m=m)
    key = problem_name.lower()

    if key in ("dtlz1", "minus-dtlz1"):
        n_pos = m - 1
        pos = rng.random((n_samples, n_pos))
        # project onto the simplex scaled to [0, 0.5] (DTLZ1's PF is the
        # hyperplane sum f_i = 0.5)
        w = rng.dirichlet(np.ones(m), size=n_samples) * 0.5
        F = w
    elif key in ("dtlz2", "minus-dtlz2"):
        # PF: sphere of radius 1 in the positive orthant
        ang = rng.random((n_samples, m - 1)) * (np.pi / 2)
        F = np.ones((n_samples, m))
        cum = np.ones(n_samples)
        for i in range(m - 1):
            F[:, i] = cum * np.cos(ang[:, i])
            cum = cum * np.sin(ang[:, i])
        F[:, m - 1] = cum
    else:
        # Fallback: dense random sampling in decision space + ND filter.
        # Not guaranteed to lie exactly on the true PF for WFG4/9 or the
        # IMOP-like problems, but gives a usable IGD+ reference set.
        X = rng.random((n_samples, problem.n_var)) * (problem.xu - problem.xl) + problem.xl
        F = problem.evaluate(X)
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
        F = F[nd]

    if key.startswith("minus-"):
        F = -F
    return F


def run_single(problem_name: str, method_name: str, seed: int, t_max: int,
                m: int = 3, mu: int = 100, true_front=None, **kwargs):
    problem = get_problem(problem_name, m=m)
    fn = METHODS[method_name]
    X, F, hist = fn(problem, t_max=t_max, seed=seed, mu=mu, true_front=true_front, **kwargs)
    return dict(
        problem=problem_name, method=method_name, seed=seed,
        n_final=F.shape[0],
        hv_adaptive_final=hist.hv_adaptive[-1] if hist.hv_adaptive else np.nan,
        hv_ext_far_final=hist.hv_ext_far[-1] if hist.hv_ext_far else np.nan,
        hv_ext_near_final=hist.hv_ext_near[-1] if hist.hv_ext_near else np.nan,
        igd_plus_final=hist.igd_plus[-1] if hist.igd_plus else np.nan,
        riesz_log_final=hist.riesz_log[-1] if hist.riesz_log else np.nan,
        X=X, F=F, history=hist,
    )


def run_comparison(problem_name: str, n_seeds: int = 30, t_max: int = 2000,
                    m: int = 3, mu: int = 100, methods=None, true_front_samples: int = 5000,
                    **kwargs) -> pd.DataFrame:
    """Run all (or a subset of) the 4 methods x n_seeds independent
    executions on a single problem and return a tidy results DataFrame.
    """
    methods = methods or list(METHODS.keys())
    true_front = sample_true_front(problem_name, m=m, n_samples=true_front_samples)
    rows = []
    for method_name in methods:
        for seed in range(n_seeds):
            res = run_single(problem_name, method_name, seed, t_max, m=m, mu=mu,
                              true_front=true_front, **kwargs)
            rows.append({k: v for k, v in res.items() if k not in ("X", "F", "history")})
    return pd.DataFrame(rows)


def pivot_indicator(df: pd.DataFrame, indicator: str, methods=None) -> np.ndarray:
    """[n_seeds, n_methods] matrix of a given indicator, ready for stats.py."""
    methods = methods or sorted(df["method"].unique())
    pivoted = df.pivot(index="seed", columns="method", values=indicator)[methods]
    return pivoted.values, methods
