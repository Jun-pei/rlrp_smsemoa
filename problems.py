"""
problems.py
===========
Problem registry and, crucially, the **reference sets Z** used for IGD+.

Suites available
----------------
    DTLZ1, DTLZ2                 (pymoo, m free)
    Minus-DTLZ1, Minus-DTLZ2     (negation, Ishibuchi et al.)
    WFG4, WFG9                   (pymoo, m free)
    IMOP1 ... IMOP8              (faithful port of PlatEMO, see imop.py;
                                  m fixed by the benchmark: 2,2,2,3,3,3,3,3)

``BENCHMARK`` is the set actually run: the six regular problems of Cuadro 1
plus the COMPLETE IMOP suite (all eight problems, not only IMOP3 and IMOP8).

Reference sets (review comment: *IGD+ - conjunto de referencia? cual usas?*)
---------------------------------------------------------------------------
``reference_set(name, m, n_points)`` returns an ANALYTICAL sample of PF(P):

  * DTLZ / WFG : pymoo's closed-form ``pareto_front(ref_dirs)`` evaluated on a
    Das-Dennis simplex lattice.  For m = 3 the default H_Z = 99 gives
    |Z| = 5050 points.
  * Minus-DTLZ : the negation of the corresponding DTLZ reference set, which
    is exactly PF of the minus problem by construction.
  * IMOP       : the ``GetOptimum`` sampler of the original MATLAB code,
    ported in imop.py.

The previous implementation instead used a Dirichlet/spherical *guess* for
DTLZ and, for WFG and IMOP, **random sampling of decision space followed by a
non-dominated filter**.  The latter is not a sample of the Pareto front: for a
14-variable WFG problem, uniform random sampling essentially never reaches
g = 0, so IGD+ was being measured against a set floating well above the true
front -- the reported IGD+ values were therefore not IGD+ at all.  Every
reference set produced here is checked in ``tests_reference_sets.py``.
"""

from __future__ import annotations
import numpy as np
from pymoo.problems import get_problem as _pymoo_get_problem
from pymoo.util.ref_dirs import get_reference_directions


# --------------------------------------------------------------------------
# Generic problem interface
# --------------------------------------------------------------------------
class Problem:
    """Minimal interface: ``evaluate(X) -> F`` with X, F 2-D arrays."""

    def __init__(self, n_var: int, n_obj: int, xl=0.0, xu=1.0, name: str = ""):
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = np.full(n_var, xl, dtype=float) if np.isscalar(xl) else np.asarray(xl, dtype=float)
        self.xu = np.full(n_var, xu, dtype=float) if np.isscalar(xu) else np.asarray(xu, dtype=float)
        self.name = name

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def pareto_front(self, n_points: int = 5000) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f"Problem({self.name}, n_var={self.n_var}, n_obj={self.n_obj})"


class PymooWrapper(Problem):
    """Wraps a pymoo Problem behind the ``evaluate(X) -> F`` interface."""

    def __init__(self, pymoo_problem, name: str):
        super().__init__(pymoo_problem.n_var, pymoo_problem.n_obj,
                         pymoo_problem.xl, pymoo_problem.xu, name)
        self._p = pymoo_problem

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        out: dict = {}
        self._p._evaluate(np.atleast_2d(X), out)
        return np.atleast_2d(out["F"])

    def pareto_front(self, n_points: int = 5000) -> np.ndarray:
        rd = _lattice(self.n_obj, n_points)
        try:
            return np.atleast_2d(self._p.pareto_front(rd))
        except TypeError:                       # problems that take no ref_dirs
            return np.atleast_2d(self._p.pareto_front())


class MinusWrapper(Problem):
    """Negates the objectives of a base problem -> inverted ("minus") front.

    THE PARETO FRONT OF A MINUS-DTLZ PROBLEM IS **NOT** THE NEGATED PARETO
    FRONT OF THE BASE PROBLEM.  This was a bug in the previous version and it
    silently corrupted every IGD+ value ever reported on Minus-DTLZ1/2.

    For the multiplicative DTLZ family, f(x) = (1 + g(x_dist)) * h(x_pos), so
    minimising -f means maximising f, which requires g to be at its MAXIMUM,
    not at 0.  Hence

        PF(Minus-DTLZ) = -(1 + g_max) * PF(DTLZ) ,

    and the scale factor is very far from 1:

        Minus-DTLZ2 (n=12, k=10): 1 + g_max = 3.5
        Minus-DTLZ1 (n= 7, k= 5): 1 + g_max = 1102.30

    So the old reference set was wrong by a factor of 3.5 on Minus-DTLZ2 and
    by three orders of magnitude on Minus-DTLZ1; the approximation sets
    dominated it entirely, which is exactly why IGD+ came out as 0.

    The factor is recovered numerically (g is separable and symmetric in the
    distance variables, so a 1-D scan of their common value is exact), and
    the construction is verified in ``tests_reference_sets.py``: points built
    at g_max are never dominated by 2x10^4 random samples.
    """

    def __init__(self, base: Problem, name: str):
        super().__init__(base.n_var, base.n_obj, base.xl, base.xu, name)
        self._base = base
        self._scale = None

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return -self._base.evaluate(X)

    @property
    def scale(self) -> float:
        """(1 + g_max) / (1 + g_min), found by a 1-D scan of the distance vars."""
        if self._scale is None:
            m, n = self.n_obj, self.n_var
            v = np.linspace(0.0, 1.0, 20001)
            X = np.zeros((v.size, n))
            X[:, m - 1:] = v[:, None]          # DTLZ: x_pos then x_dist
            best = self._base.evaluate(X).max(axis=1)
            lo = max(float(best.min()), 1e-12)
            self._scale = float(best.max() / lo)
        return self._scale

    def pareto_front(self, n_points: int = 5000) -> np.ndarray:
        return -self.scale * self._base.pareto_front(n_points)


def _lattice(m: int, n_points: int) -> np.ndarray:
    """Das-Dennis simplex lattice with roughly ``n_points`` directions."""
    n_part = 1
    while _dd_size(m, n_part + 1) <= n_points:
        n_part += 1
    return get_reference_directions("das-dennis", m, n_partitions=max(n_part, 1))


def _dd_size(m: int, h: int) -> int:
    from math import comb
    return comb(h + m - 1, m - 1)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
#: n_var per Cuadro 1
_DTLZ_WFG_NVAR = {
    "dtlz1": 7, "dtlz2": 12,
    "minus-dtlz1": 7, "minus-dtlz2": 12,
    "wfg4": 14, "wfg9": 14,
}

#: the regular geometries of Cuadro 1 (m = 3)
DTLZ_WFG_NAMES = list(_DTLZ_WFG_NVAR)

#: the COMPLETE IMOP suite -- all eight problems are run, not just IMOP3/IMOP8.
#: IMOP1-3 are bi-objective and IMOP4-8 tri-objective by definition of the
#: benchmark (see ``problem_n_obj``).
IMOP_SUITE = [f"imop{i}" for i in range(1, 9)]

#: everything the experiment protocol runs: 6 regular + 8 irregular = 14
BENCHMARK = DTLZ_WFG_NAMES + IMOP_SUITE

PROBLEM_NAMES = list(BENCHMARK)


def problem_n_obj(name: str, m: int = 3) -> int:
    """Number of objectives actually used for ``name``.

    IMOP is NOT parameterised in m: IMOP1-3 are bi-objective and IMOP4-8 are
    tri-objective in the original definition, so the requested ``m`` is
    ignored (with a clear error if it conflicts) rather than silently
    fabricating a variant that does not exist in the literature.
    """
    key = name.lower()
    if key in IMOP_SUITE:
        from .imop import IMOP_M
        return IMOP_M[key]
    return m


def get_problem(name: str, m: int = 3, n_var: int | None = None) -> Problem:
    """Factory. ``m`` is ignored for the IMOP suite (see ``problem_n_obj``)."""
    key = name.lower()

    if key in IMOP_SUITE:
        from .imop import get_imop
        return get_imop(key, n_var=n_var if n_var is not None else 10)

    if key not in _DTLZ_WFG_NVAR:
        raise ValueError(f"Unknown problem '{name}'. Options: {PROBLEM_NAMES}")
    if n_var is None:
        n_var = _DTLZ_WFG_NVAR[key]

    if key.startswith("minus-"):
        base_key = key[len("minus-"):]
        base = PymooWrapper(_pymoo_get_problem(base_key, n_var=n_var, n_obj=m),
                            base_key.upper())
        return MinusWrapper(base, f"Minus-{base_key.upper()}")

    return PymooWrapper(_pymoo_get_problem(key, n_var=n_var, n_obj=m), key.upper())


# --------------------------------------------------------------------------
# Reference sets for IGD+
# --------------------------------------------------------------------------
def reference_set(name: str, m: int = 3, n_points: int = 5000) -> np.ndarray:
    """Analytical sample Z of PF(P). See the module docstring."""
    return get_problem(name, m=m).pareto_front(n_points)


def make_reference_frame(name: str, m: int = 3, n_points: int = 5000,
                         kappa: float = 0.1):
    """Build the shared ``performance.ReferenceFrame`` for a problem."""
    from .performance import ReferenceFrame
    Z = reference_set(name, m=m, n_points=n_points)
    return ReferenceFrame(Z, kappa=kappa, name=name)
