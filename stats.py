"""
stats.py
========
Statistical analysis tools for Section 5.3 ("Analisis estadistico: test de
Wilcoxon con correccion de Bonferroni para comparaciones por pares; tests
de Friedman y Quade para rankings globales").

Input convention: a 2D array `scores[n_problems_or_runs, n_methods]` of a
single indicator (higher-is-better, e.g. HV), with one row per independent
sample (typically: one row per (problem, seed) combination, or per seed for
a single-problem comparison) and one column per method.
"""

from __future__ import annotations
import itertools
import numpy as np
from scipy import stats as sstats
from .performance import rank_biserial


def wilcoxon_bonferroni(scores: np.ndarray, method_names: list[str],
                        alpha: float = 0.05):
    """Pairwise Wilcoxon signed-rank tests with Bonferroni correction.

    Returns a dict {(name_i, name_j): {"stat":..., "p_raw":..., "p_adj":...}}.
    """
    n_methods = scores.shape[1]
    pairs = list(itertools.combinations(range(n_methods), 2))
    n_comparisons = len(pairs)
    results = {}
    for (i, j) in pairs:
        a, b = scores[:, i], scores[:, j]
        diff = a - b
        if np.allclose(diff, 0):
            stat, p = 0.0, 1.0
        else:
            try:
                stat, p = sstats.wilcoxon(a, b)
            except ValueError:
                stat, p = np.nan, 1.0
        p_adj = min(p * n_comparisons, 1.0)
        eff = rank_biserial(a, b)
        med = float(np.median(a - b))
        results[(method_names[i], method_names[j])] = dict(
            stat=stat, p_raw=p, p_adj=p_adj, significant=p_adj < alpha,
            # Definition D9: significance ALONE is not a claim of superiority.
            effect_size=eff, median_diff=med,
            winner=(method_names[i] if med > 0 else method_names[j]) if p_adj < alpha else None)
    return results


def friedman_test(scores: np.ndarray, method_names: list[str]):
    """Friedman rank test across methods (one row per block/problem)."""
    stat, p = sstats.friedmanchisquare(*[scores[:, j] for j in range(scores.shape[1])])
    avg_ranks = _average_ranks(scores)
    return dict(stat=stat, p=p,
                avg_ranks=dict(zip(method_names, avg_ranks)))


def quade_test(scores: np.ndarray, method_names: list[str]):
    """Quade test (Quade, 1979): like Friedman but weights blocks by how
    much the methods differ within that block (its 'range').

    scores: [n_blocks, n_methods], higher is better.
    """
    n, k = scores.shape
    # within-block ranks of the methods (1 = worst, k = best so that a
    # larger Q (range) block gets to dominate, consistent with Quade's
    # original higher-is-better convention)
    R = np.apply_along_axis(sstats.rankdata, 1, scores)
    # block ranges, then ranked across blocks
    ranges = scores.max(axis=1) - scores.min(axis=1)
    Q = sstats.rankdata(ranges)

    S = Q[:, None] * (R - (k + 1) / 2.0)
    Sj = S.sum(axis=0)

    A = np.sum(S ** 2)
    B = np.sum(Sj ** 2) / n

    if A - B <= 1e-12:
        return dict(stat=0.0, p=1.0, avg_ranks=dict(zip(method_names, R.mean(axis=0))))

    F_Q = (n - 1) * B / (A - B)
    df1, df2 = k - 1, (n - 1) * (k - 1)
    p = 1 - sstats.f.cdf(F_Q, df1, df2)
    return dict(stat=F_Q, p=p, df=(df1, df2),
                avg_ranks=dict(zip(method_names, R.mean(axis=0))))


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    R = np.apply_along_axis(sstats.rankdata, 1, scores)
    return R.mean(axis=0)
