"""
performance.py
==============
FORMAL DEFINITION OF "PERFORMANCE" (review comment: *falta definir
matematicamente el desempeno de los algoritmos*).

Everything below is stated so that it can be copied into the thesis as a
definition, and is implemented here exactly as stated.

------------------------------------------------------------------------------
0. Setting
------------------------------------------------------------------------------
Let P be a test problem with m objectives, Pareto front PF(P), and a finite
**reference set** Z ⊂ PF(P) (Section 2 below).  Let

    z*    = ( min_{z in Z} z_1 , ... , min_{z in Z} z_m )      (true ideal)
    z^nad = ( max_{z in Z} z_1 , ... , max_{z in Z} z_m )      (true nadir)

Every set is mapped into the **common normalised frame** by

    n(a) = ( a - z* ) / ( z^nad - z* )        (componentwise)

so that n(Z) ⊂ [0,1]^m with equality attained on every axis.  Write
Ahat = n(ND(A)) for the normalised non-dominated subset of an approximation
set A, and Zhat = n(Z).

CRITICAL: the frame (z*, z^nad) and the external reference point below are
properties of the PROBLEM, identical for every algorithm and every run.  They
are NOT the algorithm's own adaptive z_ref.  Measuring HV with the adaptive
z_ref would make "performance" trivially maximisable by inflating z_ref, which
is exactly the failure mode the reference-point bound in rl_planner.py guards
against.

------------------------------------------------------------------------------
1. Primary quality indicators
------------------------------------------------------------------------------
Fix the external reference point  z^ext = (1 + kappa) * 1 ,  kappa = 0.1.

(D1) HYPERVOLUME RATIO   -- convergence + spread, higher is better, in [0,1]:

         HVR(A) = HV( Ahat , z^ext ) / HV( Zhat , z^ext )

     Normalising by the HV of the reference set makes HVR comparable ACROSS
     problems (raw HV is not: its magnitude depends on m and on the geometry
     of PF(P)), and gives it the absolute meaning "fraction of the attainable
     hypervolume that this run captured".

(D2) INVERTED GENERATIONAL DISTANCE PLUS -- lower is better, in [0, sqrt(m)]:

         IGD+(A) = (1/|Z|) * sum_{z in Zhat} min_{a in Ahat} d+(z, a),
         d+(z,a) = || ( max(a_1 - z_1, 0), ... , max(a_m - z_m, 0) ) ||_2

     Weakly Pareto compliant (Ishibuchi et al. 2015), unlike plain IGD.

(D3) UNIFORMITY -- lower is better:

         Eratio(A) = E^ln_s( Ahat ) / E^ln_s( U_{|Ahat|} )

     where E^ln_s(S) = ln(1 + (1/(|S|(|S|-1))) sum_{i != j} ||s_i - s_j||^{-s})
     is the log average-pairwise Riesz s-energy (s = m by default) and
     U_{|Ahat|} is a |Ahat|-point maximally-uniform subset of Zhat, obtained
     by greedy energy-minimising selection.  Eratio ~ 1 means the run's
     distribution is as uniform as an ideal distribution of the same size on
     the same front; Eratio > 1 means it is more clustered.  Dividing by the
     reference energy is what removes the dependence on m, on |A| and on the
     scale of the front, none of which the raw energy controls for.

------------------------------------------------------------------------------
2. Reference set Z  (review comment: *IGD+ - conjunto de referencia? cual?*)
------------------------------------------------------------------------------
Z is ALWAYS an analytical or provably-dense sample of PF(P), never a sample of
the algorithms' own output (which would make IGD+ depend on the methods being
compared):

  * DTLZ1, DTLZ2, WFG4, WFG9 : pymoo's closed-form ``pareto_front(ref_dirs)``
    evaluated on a Das-Dennis simplex lattice with H_Z divisions (default
    H_Z = 99 for m = 3, |Z| = 5050).
  * Minus-DTLZ1, Minus-DTLZ2 : the SCALED negation -(1 + g_max) * Z of the
    corresponding DTLZ set.  Not simply -Z: since f = (1 + g)*h(x_pos),
    minimising -f requires g at its MAXIMUM, not at 0.  The factor is 3.5 for
    Minus-DTLZ2 and 1102.30 for Minus-DTLZ1 (see problems.MinusWrapper).
  * IMOP1..IMOP8 : the analytical ``GetOptimum`` sampler of the original
    PlatEMO implementation, ported in ``imop.py`` and verified against a dense
    non-dominated sample of the g = 0 manifold (max deviation < 2e-2, and
    < 6e-3 for all but IMOP8).

Z is never drawn by random decision-space sampling followed by a non-dominated
filter: that is not a sample of PF(P) -- random sampling of a 14-variable WFG
problem essentially never reaches g = 0 -- and IGD+ measured against such a set
is not IGD+.  ``tests_reference_sets.py`` checks every Z that is used.

|Z| must be reported with the results: IGD+ is a sample statistic of Z, so
values obtained with different |Z| are NOT comparable.  It is stored in each
run's ``.meta.json`` as ``n_ref``.

------------------------------------------------------------------------------
3. Anytime performance
------------------------------------------------------------------------------
Since all per-generation values are stored, performance is defined not only at
the final generation but over the whole trajectory.  For an indicator I in
{HVR, IGD+, Eratio} recorded at generations t = 1..T:

(D4) ANYTIME SCORE  (for the higher-is-better HVR):

         AT(A) = (1/T) * sum_{t=1}^{T} HVR( A(t) )      in [0,1]

     i.e. the normalised area under the HVR-vs-generation curve.  This is the
     quantity a reference-point *schedule* is actually supposed to improve: a
     fixed-reference-point baseline and RL-RP can perfectly well end at the
     same final HVR while differing substantially in how fast they got there.

(D5) TIME TO TARGET:

         T_q(A) = min { t : HVR(A(t)) >= q * HVR_max }  (infinity if never),

     with q = 0.95 by default and HVR_max the best HVR reached by ANY method
     on that (problem, seed).

------------------------------------------------------------------------------
4. Aggregation over runs and problems, and what "better" means
------------------------------------------------------------------------------
Let R be the number of independent runs (R = 30) and let I_{a,p,r} be an
indicator for algorithm a on problem p in run r.

(D6) Per-problem summary: the median over runs, med_r I_{a,p,r}, plus the
     interquartile range.  Median, not mean: HV/IGD+ distributions over seeds
     are routinely skewed and the mean is not robust.

(D7) Per-problem ranking: rank the algorithms on each (p, r) block, then
     average over r.

(D8) Global ranking: the Friedman and Quade average ranks over all blocks
     (p, r), as in Section 5.3.

(D9) ALGORITHM a IS SAID TO OUTPERFORM ALGORITHM b ON PROBLEM p IFF the
     two-sided Wilcoxon signed-rank test over the R paired runs rejects
     equality at the Bonferroni-corrected level alpha / C (alpha = 0.05,
     C = number of pairwise comparisons) AND the sign of the median difference
     favours a.  Statistical significance is reported together with the effect
     size (the rank-biserial correlation r = 1 - 2U/(n(n+1)/2)), because with
     R = 30 paired runs a difference in the fourth decimal of HV can be
     "significant" and simultaneously irrelevant.
------------------------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np

from .indicators import (
    hypervolume, igd_plus, riesz_energy_log, normalize, nondominated,
)

#: kappa of Definition D1
KAPPA_EXT = 0.1
#: q of Definition D5
TARGET_Q = 0.95


class ReferenceFrame:
    """The problem-level frame (z*, z^nad, Z, z^ext) of Sections 0-2.

    Built ONCE per problem and shared by every algorithm and every run, which
    is what makes the resulting numbers comparable.
    """

    def __init__(self, reference_set: np.ndarray, kappa: float = KAPPA_EXT,
                 riesz_s: float | None = None, name: str = ""):
        Z = np.atleast_2d(np.asarray(reference_set, dtype=float))
        self.name = name
        self.Z_raw = Z
        self.m = Z.shape[1]
        self.ideal = Z.min(axis=0)
        self.nadir = Z.max(axis=0)
        self.Zhat = normalize(Z, self.ideal, self.nadir)
        self.kappa = kappa
        self.zext = np.full(self.m, 1.0 + kappa)
        self.riesz_s = float(riesz_s if riesz_s is not None else self.m)
        self.hv_ref = hypervolume(self.Zhat, self.zext)
        self._energy_cache: dict[int, float] = {}

    # -- helpers -----------------------------------------------------------
    def to_frame(self, F: np.ndarray, only_nd: bool = True) -> np.ndarray:
        F = np.atleast_2d(np.asarray(F, dtype=float))
        if only_nd and F.shape[0] > 1:
            F = F[nondominated(F)]
        return normalize(F, self.ideal, self.nadir)

    def reference_energy(self, n: int, rng_seed: int = 0) -> float:
        """E^ln_s of a maximally-uniform n-point subset of Zhat (Def. D3)."""
        n = int(min(max(n, 2), self.Zhat.shape[0]))
        if n in self._energy_cache:
            return self._energy_cache[n]
        U = _greedy_uniform_subset(self.Zhat, n, seed=rng_seed)
        val = riesz_energy_log(U, s=self.riesz_s)
        self._energy_cache[n] = val
        return val

    # -- D1, D2, D3 --------------------------------------------------------
    def hvr(self, F: np.ndarray) -> float:
        if self.hv_ref <= 0:
            return float("nan")
        return float(hypervolume(self.to_frame(F), self.zext) / self.hv_ref)

    def igd_plus(self, F: np.ndarray) -> float:
        return igd_plus(self.to_frame(F), self.Zhat)

    def energy_ratio(self, F: np.ndarray) -> float:
        A = self.to_frame(F)
        if A.shape[0] < 2:
            return float("nan")
        ref = self.reference_energy(A.shape[0])
        if ref <= 1e-12:
            return float("nan")
        return float(riesz_energy_log(A, s=self.riesz_s) / ref)

    def evaluate(self, F: np.ndarray) -> dict:
        """All three primary indicators of Section 1 for one approximation set."""
        return dict(hvr=self.hvr(F), igd_plus=self.igd_plus(F),
                    energy_ratio=self.energy_ratio(F))


# ---------------------------------------------------------------------------
# D4, D5: anytime performance
# ---------------------------------------------------------------------------
def anytime_score(hvr_curve) -> float:
    """AT(A) of Definition D4: mean HVR over all recorded generations."""
    v = np.asarray(hvr_curve, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def time_to_target(hvr_curve, hvr_max: float, q: float = TARGET_Q) -> float:
    """T_q(A) of Definition D5 (1-indexed generation; inf if never reached)."""
    v = np.asarray(hvr_curve, dtype=float)
    if not np.isfinite(hvr_max) or hvr_max <= 0:
        return float("inf")
    hit = np.flatnonzero(v >= q * hvr_max)
    return float(hit[0] + 1) if hit.size else float("inf")


# ---------------------------------------------------------------------------
# D9: effect size to accompany the significance test
# ---------------------------------------------------------------------------
def rank_biserial(a, b) -> float:
    """Matched-pairs rank-biserial correlation in [-1,1] for Wilcoxon.

    +1: a beats b in every pair;  0: no consistent difference.
    """
    from scipy.stats import rankdata
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


# ---------------------------------------------------------------------------
# Uniform subset used by Definition D3
# ---------------------------------------------------------------------------
def _greedy_uniform_subset(Z: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Greedy max-min (farthest-point) subset -- a cheap surrogate for the
    energy-minimising n-point subset of Z, and the standard construction for
    "ideally distributed" reference sets."""
    rng = np.random.default_rng(seed)
    N = Z.shape[0]
    if n >= N:
        return Z
    if N > 20000:                                # subsample for tractability
        Z = Z[rng.choice(N, 20000, replace=False)]
        N = Z.shape[0]
    idx = [int(rng.integers(N))]
    d = np.linalg.norm(Z - Z[idx[0]], axis=1)
    for _ in range(n - 1):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(Z - Z[j], axis=1))
    return Z[idx]
