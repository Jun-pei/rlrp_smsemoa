"""
indicators.py
=============
Indicators used by the algorithm (state features, SMS-EMOA selection) and by
the performance assessment (Sec. 5).

Changes w.r.t. the first version, all motivated by the review comments:

* ``hypervolume`` now actually *discards* points that do not dominate z_ref
  (the old code computed the mask and then ignored it) and can fall back to a
  Monte-Carlo estimator, so the code no longer depends on exact HV being
  affordable -- this is what makes m > 3 feasible (see ``backend``).
* ``estimate_ideal_nadir`` estimates the nadir from the **non-dominated
  front**, not from the whole population.  Using the max over the whole
  population makes the normalisation depend on the worst dominated
  individuals, so [0,1]^m was not really the normalised objective space.
* ``riesz_energy`` is reported **per ordered pair**, so the value no longer
  changes just because |A| changed; this is what makes the E-ratio state
  feature comparable across generations.
* NEW ``estimate_curvature_p`` / ``geometry_gamma``: an explicit, measurable
  descriptor of the *geometry* of the current front (convex / linear /
  concave), replacing the implicit contour-vs-interior heuristic.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist
from pymoo.indicators.hv import HV as _HV
from pymoo.indicators.igd_plus import IGDPlus as _IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

_nds = NonDominatedSorting()

#: Above this many objectives exact HV (WFG algorithm) becomes the bottleneck
#: of the whole run; see ``hypervolume(..., backend=...)``.
EXACT_HV_MAX_OBJ = 5


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------
def hypervolume(F: np.ndarray, ref_point: np.ndarray, backend: str = "auto",
                n_mc: int = 100_000, rng: np.random.Generator | None = None) -> float:
    """HV(A, z_ref) for minimisation.

    Parameters
    ----------
    backend : {"auto", "exact", "mc"}
        ``"auto"`` uses the exact indicator for m <= EXACT_HV_MAX_OBJ and the
        Monte-Carlo estimator above that.
    n_mc : Monte-Carlo sample size (only used by the "mc" backend).
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    z = np.asarray(ref_point, dtype=float)
    if F.shape[0] == 0:
        return 0.0
    F = F[np.all(F < z, axis=1)]          # only dominating points contribute
    if F.shape[0] == 0:
        return 0.0

    m = F.shape[1]
    if backend == "auto":
        backend = "exact" if m <= EXACT_HV_MAX_OBJ else "mc"
    if backend == "exact":
        return float(_HV(ref_point=z)(F))
    return _hv_monte_carlo(F, z, n_mc, rng)


def _hv_monte_carlo(F: np.ndarray, z: np.ndarray, n_mc: int,
                    rng: np.random.Generator | None) -> float:
    """Unbiased MC estimate of HV: sample the box [ideal, z] and count the
    fraction of samples dominated by at least one point of F."""
    rng = rng or np.random.default_rng(0)
    lo = F.min(axis=0)
    vol_box = float(np.prod(np.maximum(z - lo, 0.0)))
    if vol_box <= 0.0:
        return 0.0
    S = rng.random((n_mc, F.shape[1])) * (z - lo) + lo
    hit = np.zeros(n_mc, dtype=bool)
    for i in range(0, F.shape[0], 64):                 # blocked, bounded memory
        blk = F[i:i + 64]
        hit |= np.any(np.all(blk[None, :, :] <= S[:, None, :], axis=2), axis=1)
    return vol_box * float(hit.mean())


def hv_contributions(F: np.ndarray, ref_point: np.ndarray, **kw) -> np.ndarray:
    """HVC(x_i) = HV(F, z) - HV(F \\ {x_i}, z) for every row of F."""
    n = F.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([hypervolume(F, ref_point, **kw)])
    total = hypervolume(F, ref_point, **kw)
    contrib = np.empty(n)
    for i in range(n):
        contrib[i] = total - hypervolume(np.delete(F, i, axis=0), ref_point, **kw)
    return contrib


# ---------------------------------------------------------------------------
# IGD+
# ---------------------------------------------------------------------------
def igd_plus(F: np.ndarray, reference_set: np.ndarray) -> float:
    """IGD+ of the approximation F w.r.t. the reference set Z.

    NOTE: IGD+ is *not* scale invariant.  To compare across problems, both F
    and Z must live in the same frame; ``performance.py`` always normalises
    with the true ideal/nadir of the problem before calling this function.
    """
    if reference_set is None or F is None or np.size(F) == 0:
        return float("inf")
    return float(_IGDPlus(np.asarray(reference_set, dtype=float))(np.atleast_2d(F)))


# ---------------------------------------------------------------------------
# Riesz s-energy (uniformity)
# ---------------------------------------------------------------------------
def riesz_energy(F: np.ndarray, s: float = 3.0, per_pair: bool = True) -> float:
    """E_s(A) = sum_{i != j} ||a_i - a_j||^{-s}.

    With ``per_pair=True`` the sum is divided by n(n-1), i.e. the *average*
    pairwise energy.  This matters here: the raw sum scales like n^2, so a
    change in |A| alone would move the state feature E_norm even when the
    geometry of the distribution is unchanged.
    """
    n = F.shape[0]
    if n < 2:
        return 0.0
    d = np.maximum(pdist(F, metric="euclidean"), 1e-12)
    total = 2.0 * float(np.sum(d ** (-s)))     # pdist counts each pair once
    return total / (n * (n - 1)) if per_pair else total


def riesz_energy_log(F: np.ndarray, s: float = 3.0, per_pair: bool = True) -> float:
    """E^ln_s(A) = ln(1 + E_s(A)).  Lower = more uniform."""
    return float(np.log1p(riesz_energy(F, s=s, per_pair=per_pair)))


# ---------------------------------------------------------------------------
# Spread / diversity
# ---------------------------------------------------------------------------
def mean_dispersion(F: np.ndarray) -> float:
    """D(A): mean over objectives of the per-objective standard deviation.

    On a set already mapped into [0,1]^m, D(A) in [0, 0.5]; the maximum 0.5 is
    attained by the degenerate two-cluster distribution with half the points
    at 0 and half at 1.  ``normalised_dispersion`` rescales it into [0,1] with
    that absolute bound.
    """
    if F.shape[0] < 2:
        return 0.0
    return float(np.mean(np.std(F, axis=0)))


def normalised_dispersion(F_norm: np.ndarray) -> float:
    """D_norm(A) = D(A) / 0.5, clipped to [0,1].  F_norm must be in [0,1]^m."""
    return float(min(mean_dispersion(F_norm) / 0.5, 1.0))


def extent(F_norm: np.ndarray) -> float:
    """Fraction of the normalised range actually covered, mean over objectives."""
    if F_norm.shape[0] < 2:
        return 0.0
    return float(np.mean(np.clip(F_norm.max(axis=0) - F_norm.min(axis=0), 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Geometry of the front
# ---------------------------------------------------------------------------
def estimate_curvature_p(F_norm: np.ndarray, p_lo: float = 0.05, p_hi: float = 20.0,
                         n_iter: int = 60) -> float:
    """Estimate the curvature exponent p of the front  sum_i f_i^p = 1.

    The front of a normalised m-objective problem is well approximated by the
    level set of an L_p "norm":  p < 1 convex, p = 1 linear, p > 1 concave
    (DTLZ1 -> p ~ 1, DTLZ2 -> p ~ 2, Minus-DTLZ2 -> p < 1).  We fit p by
    bisection on

        h(p) = median_j ( sum_i f_ij^p ) - 1,

    which is non-increasing in p for points inside the unit box, so bisection
    converges.  Only points with at least one non-zero coordinate are used.

    Returns the fitted p, clipped to [p_lo, p_hi].
    """
    F = np.clip(np.atleast_2d(np.asarray(F_norm, dtype=float)), 0.0, 1.0)
    F = F[F.sum(axis=1) > 1e-9]
    if F.shape[0] < 3:
        return 1.0

    def h(p):
        return float(np.median(np.sum(F ** p, axis=1)) - 1.0)

    lo, hi = p_lo, p_hi
    if h(lo) < 0.0:          # even the smallest exponent undershoots -> convex
        return p_lo
    if h(hi) > 0.0:          # never reaches 1 -> strongly concave
        return p_hi
    for _ in range(n_iter):
        mid = np.sqrt(lo * hi)      # geometric bisection: p lives on a log scale
        if h(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def geometry_gamma(F_norm: np.ndarray) -> float:
    """Bounded geometry descriptor gamma = p / (1 + p) in (0, 1).

        gamma < 1/2  -> convex front
        gamma = 1/2  -> linear front  (p = 1)
        gamma > 1/2  -> concave front

    Bounded by construction, hence directly discretisable into state bins with
    no running normalisation.
    """
    p = estimate_curvature_p(F_norm)
    return float(p / (1.0 + p))


def compute_ghat(F_nd: np.ndarray, zref: np.ndarray, eta: float = 1e-6, **kw) -> float:
    """Contour-vs-interior share of the HV contribution (kept as a diagnostic).

    Complements the curvature estimate: it says *where on the front* the HV
    pressure currently sits, whereas gamma says *what shape* the front has.
    """
    n = F_nd.shape[0]
    if n < 3:
        return 0.5
    c = F_nd.mean(axis=0)
    d = np.linalg.norm(F_nd - c, axis=1)
    contour = d >= np.median(d)
    if not np.any(contour) or np.all(contour):
        return 0.5
    hvc = hv_contributions(F_nd, zref, **kw)
    mb, mi = hvc[contour].mean(), hvc[~contour].mean()
    return float(mb / (mb + mi + eta))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def nondominated(F: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated front of F."""
    return _nds.do(np.atleast_2d(F), only_non_dominated_front=True)


def estimate_ideal_nadir(F: np.ndarray, from_front: bool = True):
    """Estimate (z_ideal, z_nadir), the frame that maps objectives onto [0,1]^m.

    ``from_front=True`` (default, and the textbook definition) takes the nadir
    as the componentwise maximum over the **non-dominated front**.  Taking the
    maximum over the whole population -- what the previous version did -- lets
    arbitrarily bad dominated individuals dictate the scale, so the
    "normalised" space was neither stable across generations nor bounded by
    the front.
    """
    F = np.atleast_2d(F)
    ideal = F.min(axis=0)
    if from_front and F.shape[0] > 1:
        nadir = F[nondominated(F)].max(axis=0)
    else:
        nadir = F.max(axis=0)
    return ideal, np.maximum(nadir, ideal + 1e-12)


def normalize(F: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    """Affine map f -> (f - z_ideal) / (z_nadir - z_ideal)."""
    span = np.maximum(np.asarray(nadir, dtype=float) - np.asarray(ideal, dtype=float), 1e-12)
    return (np.atleast_2d(F) - ideal) / span
