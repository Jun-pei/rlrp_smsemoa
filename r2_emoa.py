"""
r2_emoa.py
==========
R2-EMOA -- the indicator-agnostic control of the comparison.

    H. Trautmann, T. Wagner, D. Brockhoff, "R2-EMOA: Focused Multiobjective
    Search Using R2-Indicator-Based Selection", LION 7, LNCS 7997, pp. 70-74,
    2013.  Extended in D. Brockhoff, T. Wagner, H. Trautmann, "R2 Indicator
    Based Multiobjective Search", Evolutionary Computation 23(3):369-395, 2015.

Selection is driven by the contribution to the unary R2 indicator instead of
the hypervolume contribution.  R2-EMOA has NO reference point to adapt, and
that is exactly what makes it the useful control here: it answers a question
none of the SMS-EMOA variants can -- how much of the performance difference is
about the reference point AT ALL, versus about hypervolume-based selection
being the wrong tool on irregular fronts?  If R2-EMOA beats every SMS-EMOA
variant on Minus-DTLZ / IMOP, then tuning z_ref is optimising the wrong knob,
and that is a more valuable finding than a marginal HV win.

R2 indicator
------------
For a weight set W on the simplex and a utopian point z*,

    R2(A, W, z*) = (1/|W|) * sum_{w in W} min_{a in A} max_i w_i |z*_i - a_i|

(weighted Tchebycheff utility; LOWER is better).  The contribution of a is
R2(A \\ {a}, W, z*) - R2(A, W, z*) >= 0, and the individual with the SMALLEST
contribution is removed.

Complexity: all |A| contributions come out of one |W| x |A| utility matrix in
O(|W| |A| m) total, because removing ``a`` only changes the inner minimum for
those weights where ``a`` is the UNIQUE minimiser, and then only up to the
second-smallest utility.  Unlike exact hypervolume this does not blow up with
m, which is the practical answer to the scalability question.
"""

from __future__ import annotations
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from .core import DEFAULTS, History, baseline_extra, init_population, record
from .indicators import estimate_ideal_nadir, normalize
from .sms_emoa import make_offspring

_nds = NonDominatedSorting()

#: small positive shift making z* strictly utopian (Brockhoff et al.)
UTOPIAN_EPS = 1e-4


# ---------------------------------------------------------------------------
# R2 indicator machinery
# ---------------------------------------------------------------------------
def r2_weights(m: int, n_weights: int = 100) -> np.ndarray:
    """Uniform weight vectors on the (m-1)-simplex (Das-Dennis lattice)."""
    h = 1
    while _dd(m, h + 1) <= max(n_weights, m):
        h += 1
    return get_reference_directions("das-dennis", m, n_partitions=max(h, 1)).astype(float)


def _dd(m: int, h: int) -> int:
    from math import comb
    return comb(h + m - 1, m - 1)


def _utility_matrix(F: np.ndarray, W: np.ndarray, z_star: np.ndarray) -> np.ndarray:
    """U[k, i] = max_j W[k,j] * |z*_j - F[i,j]|  -> shape (|W|, |A|)."""
    D = np.abs(F[None, :, :] - z_star[None, None, :])      # (1, |A|, m)
    return np.max(W[:, None, :] * D, axis=2)               # (|W|, |A|)


def r2_indicator(F: np.ndarray, W: np.ndarray, z_star: np.ndarray) -> float:
    """R2(A, W, z*).  Lower is better."""
    F = np.atleast_2d(F)
    if F.shape[0] == 0:
        return float("inf")
    return float(_utility_matrix(F, W, z_star).min(axis=1).mean())


def r2_contributions(F: np.ndarray, W: np.ndarray, z_star: np.ndarray) -> np.ndarray:
    """R2(A \\ {a_i}) - R2(A) for every i, in one pass."""
    F = np.atleast_2d(F)
    n = F.shape[0]
    if n <= 1:
        return np.zeros(n)
    U = _utility_matrix(F, W, z_star)                      # (|W|, n)
    order = np.argsort(U, axis=1, kind="stable")
    best_i = order[:, 0]
    best = U[np.arange(U.shape[0]), best_i]
    second = U[np.arange(U.shape[0]), order[:, 1]]
    contrib = np.zeros(n)
    # only the unique minimiser of a weight loses anything when removed
    np.add.at(contrib, best_i, second - best)
    return contrib / U.shape[0]


def r2_emoa_eliminate(X, F_raw, W, ideal, nadir):
    """Remove the individual with the smallest R2 contribution from the worst
    non-dominated front -- steady-state, mirroring SMS-EMOA's structure so that
    the two methods differ ONLY in the indicator."""
    fronts = _nds.do(F_raw)
    worst = fronts[-1]
    if len(worst) == 1:
        drop = worst[0]
    else:
        F_norm = normalize(F_raw[worst], ideal, nadir)
        z_star = np.full(F_norm.shape[1], -UTOPIAN_EPS)    # utopian in [0,1]^m
        contrib = r2_contributions(F_norm, W, z_star)
        if not np.any(contrib > 0):
            drop = worst[int(np.argmax(F_norm.sum(axis=1)))]
        else:
            drop = worst[int(np.argmin(contrib))]
    keep = np.ones(X.shape[0], dtype=bool)
    keep[drop] = False
    return X[keep], F_raw[keep]


# ---------------------------------------------------------------------------
# R2-EMOA
# ---------------------------------------------------------------------------
def run_r2_emoa(problem, t_max, seed=0, frame=None, n_weights=100, **kwargs):
    cfg = {**DEFAULTS, **kwargs}
    cfg["_t_max"] = t_max
    rng = np.random.default_rng(seed)
    X, F = init_population(problem, cfg["mu"], rng)
    W = r2_weights(problem.n_obj, n_weights)
    hist = History()
    hist.meta["n_weights"] = int(W.shape[0])
    # logging only: R2-EMOA has no reference point, but the diagnostic column
    # hv_adaptive needs one, so the fixed (1+kappa)*1 is used and flagged here
    z_log = np.full(problem.n_obj, 1.0 + cfg["kappa"])

    for t in range(1, t_max + 1):
        ideal, nadir = estimate_ideal_nadir(F)
        pm = 1.0 / problem.n_var
        child_x = make_offspring(X, problem.xl, problem.xu, cfg["pc"], cfg["eta_c"],
                                 pm, cfg["eta_m"], rng)
        child_f = problem.evaluate(child_x.reshape(1, -1))
        X, F = r2_emoa_eliminate(np.vstack([X, child_x]), np.vstack([F, child_f]),
                                 W, ideal, nadir)
        record(hist, t, F, z_log, frame, cfg, baseline_extra("r2"))
    return X, F, hist
