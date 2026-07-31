"""
sms_emoa.py
===========
Steady-state SMS-EMOA [2] building blocks:

  * SBX crossover (pc=1.0, eta_c=15) and polynomial mutation
    (pm=1/n, eta_m=20), per the proposal's "Algoritmo base" (Sec. 2.2).
  * One steady-state generation step: produce a single offspring, merge
    with the population, and remove the individual with the smallest HV
    contribution from the *worst* non-dominated front (standard, efficient
    SMS-EMOA elimination -- only the worst front needs HV contributions,
    since better fronts are never candidates for removal).
"""

from __future__ import annotations
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from .indicators import hv_contributions, normalize

_nds = NonDominatedSorting()


def sbx_crossover(p1: np.ndarray, p2: np.ndarray, xl: np.ndarray, xu: np.ndarray,
                   pc: float, eta_c: float, rng: np.random.Generator) -> np.ndarray:
    """Simulated Binary Crossover -> returns ONE child (the first of the pair)."""
    n = p1.shape[0]
    c1 = p1.copy()
    if rng.random() > pc:
        return c1
    for k in range(n):
        if rng.random() > 0.5 or abs(p1[k] - p2[k]) < 1e-14:
            continue
        x1, x2 = min(p1[k], p2[k]), max(p1[k], p2[k])
        lo, hi = xl[k], xu[k]
        rand = rng.random()

        beta = 1.0 + (2.0 * (x1 - lo) / max(x2 - x1, 1e-14))
        alpha = 2.0 - beta ** (-(eta_c + 1))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))
        child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        beta = 1.0 + (2.0 * (hi - x2) / max(x2 - x1, 1e-14))
        alpha = 2.0 - beta ** (-(eta_c + 1))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))
        child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

        child1 = min(max(child1, lo), hi)
        child2 = min(max(child2, lo), hi)
        # randomly choose which child's k-th gene we keep, matching the
        # standard SBX implementation used across the EMO literature
        c1[k] = child1 if rng.random() < 0.5 else child2
    return c1


def polynomial_mutation(x: np.ndarray, xl: np.ndarray, xu: np.ndarray,
                         pm: float, eta_m: float, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    n = x.shape[0]
    for k in range(n):
        if rng.random() > pm:
            continue
        lo, hi = xl[k], xu[k]
        if hi <= lo:
            continue
        delta1 = (y[k] - lo) / (hi - lo)
        delta2 = (hi - y[k]) / (hi - lo)
        rand = rng.random()
        mut_pow = 1.0 / (eta_m + 1)
        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1))
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1))
            deltaq = 1.0 - val ** mut_pow
        y[k] = y[k] + deltaq * (hi - lo)
        y[k] = min(max(y[k], lo), hi)
    return y


def make_offspring(P: np.ndarray, xl: np.ndarray, xu: np.ndarray,
                    pc: float, eta_c: float, pm: float, eta_m: float,
                    rng: np.random.Generator) -> np.ndarray:
    """One steady-state offspring via uniform-random parent selection."""
    n_pop = P.shape[0]
    i1, i2 = rng.integers(0, n_pop, size=2)
    child = sbx_crossover(P[i1], P[i2], xl, xu, pc, eta_c, rng)
    child = polynomial_mutation(child, xl, xu, pm, eta_m, rng)
    return child


def sms_emoa_eliminate(X: np.ndarray, F_raw: np.ndarray, zref: np.ndarray,
                        ideal: np.ndarray, nadir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove the least HV-contributing individual from the worst front.

    F_raw is in the problem's native objective space; zref/ideal/nadir are
    given in the *normalized* [0,1]^m space used for the HV calculation, as
    specified in Sec. 2.2 ("El PR se especifica en el espacio objetivo
    normalizado").
    """
    fronts = _nds.do(F_raw)
    worst = fronts[-1]
    if len(worst) == 1:
        drop = worst[0]
    else:
        F_norm = normalize(F_raw[worst], ideal, nadir)
        contrib = hv_contributions(F_norm, zref)
        if not np.any(contrib > 0):
            # No member of the worst front dominates z_ref, so every HV
            # contribution is exactly 0 and argmin would deterministically
            # return index 0 -- i.e. selection would silently degenerate into
            # "always delete the first individual".  Fall back to the standard
            # secondary criterion: drop the point furthest from the ideal in
            # the normalised frame.
            drop = worst[int(np.argmax(F_norm.sum(axis=1)))]
        else:
            drop = worst[int(np.argmin(contrib))]
    keep = np.ones(X.shape[0], dtype=bool)
    keep[drop] = False
    return X[keep], F_raw[keep]
