"""
indicators.py
=============
Implements the indicators of Cuadro 2 / Section 3 of the proposal:

  * Hypervolume (HV), via pymoo's exact HV indicator.
  * HV contributions HVC(x, P, zref) = HV(P, zref) - HV(P \\ {x}, zref),
    used both by the SMS-EMOA elimination step and by the g_hat(t) state
    feature (Eq. 6).
  * IGD+ (Pareto-compliant convergence+diversity indicator), via pymoo.
  * Riesz s-energy E_s(A) (Eq. 4) and its log form E^ln_s (Section 3),
    used for the uniformity feature E_norm(t) (Eq. 5).
  * D(t): mean per-dimension standard deviation of the non-dominated front
    (state feature / auxiliary indicator of Cuadro 2).
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform
from pymoo.indicators.hv import HV as _HV
from pymoo.indicators.igd_plus import IGDPlus as _IGDPlus


def hypervolume(F: np.ndarray, ref_point: np.ndarray) -> float:
    """HV(A, zref). F must be minimization-style objectives dominated by ref_point."""
    if F.shape[0] == 0:
        return 0.0
    mask = np.all(F < ref_point, axis=1)  # points with no HV contribution are harmless either way
    if not np.any(mask):
        return 0.0
    return float(_HV(ref_point=np.asarray(ref_point, dtype=float))(F))


def hv_contributions(F: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
    """HVC(x_i) = HV(F, ref) - HV(F without x_i, ref) for every row of F.

    Exact (inclusion-exclusion) computation: O(n) HV evaluations. Intended
    to be called on small fronts (e.g. the worst non-dominated front during
    SMS-EMOA elimination, or the current non-dominated set And(t) for the
    g_hat(t) feature), not on the full population.
    """
    n = F.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        # single point's contribution is simply its HV against ref
        return np.array([hypervolume(F, ref_point)])
    total = hypervolume(F, ref_point)
    contrib = np.empty(n)
    for i in range(n):
        rest = np.delete(F, i, axis=0)
        contrib[i] = total - hypervolume(rest, ref_point)
    return contrib


def igd_plus(F: np.ndarray, true_front: np.ndarray) -> float:
    """IGD+ of the approximation F w.r.t. a (sampled) true Pareto front."""
    if F.shape[0] == 0:
        return float("inf")
    return float(_IGDPlus(true_front)(F))


def riesz_energy(F: np.ndarray, s: float = 3.0) -> float:
    """E_s(A) = sum_i sum_{j!=i} ||a_i - a_j||^{-s}  (Eq. 4)."""
    n = F.shape[0]
    if n < 2:
        return 0.0
    d = pdist(F, metric="euclidean")
    d = np.maximum(d, 1e-12)
    # pdist gives each unordered pair once; the double sum in Eq.4 counts
    # both (i,j) and (j,i), i.e. a factor of 2.
    return float(2.0 * np.sum(d ** (-s)))


def riesz_energy_log(F: np.ndarray, s: float = 3.0) -> float:
    """E^ln_s(A) = ln(1 + E_s(A))."""
    return float(np.log1p(riesz_energy(F, s=s)))


def mean_dispersion(F: np.ndarray) -> float:
    """D(t): mean (over objective dimensions) of the per-dimension std dev."""
    if F.shape[0] < 2:
        return 0.0
    return float(np.mean(np.std(F, axis=0)))


def estimate_ideal_nadir(F: np.ndarray):
    """Ideal/nadir estimated from a non-dominated set (used for normalization)."""
    ideal = np.min(F, axis=0)
    nadir = np.max(F, axis=0)
    return ideal, nadir


def normalize(F: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    span = np.maximum(nadir - ideal, 1e-12)
    return (F - ideal) / span
