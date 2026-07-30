"""
problems.py
===========
The eight test problems of Cuadro 1 in the proposal:

    DTLZ1, DTLZ2, Minus-DTLZ1, Minus-DTLZ2, WFG4, WFG9, IMOP3, IMOP8

DTLZ1/2 and WFG4/9 are taken verbatim from pymoo's built-in problem suite
(``pymoo.problems.get_problem``), with the number of decision variables set
to match Cuadro 1 (n=7 for DTLZ1, n=12 for DTLZ2, n=14 for WFG4/9).

Minus-DTLZ1/2 follow the standard "minus" construction of Ishibuchi et al.
(CEC 2018 / PPSN 2018, referenced as [7, 8] in the proposal): the objectives
of the corresponding DTLZ problem are simply negated, f_i^minus(x) =
-f_i^DTLZ(x), which inverts the front (concave <-> convex, and pushes the
HV-optimal distributions towards the boundary instead of the interior).

IMOP3 / IMOP8
-------------
IMPORTANT HONESTY NOTE: the official IMOP suite (Tian et al., IEEE CIM 2019,
ref. [12]) is *not* shipped with pymoo, and its exact closed-form MATLAB
source (PlatEMO ``IMOP3.m`` / ``IMOP8.m``) could not be verified verbatim in
this environment. The official IMOP3 is in fact a 2-objective problem (a
1-D discontinuous curve), while the proposal's Cuadro 1 calls for a
3-objective version (n = (m-1) + l = 2 + 5 = 7), which does not match the
literature's IMOP3. Rather than silently fabricate a "verbatim" port that
might be wrong, this module implements two *custom* 3-objective problems
that reproduce the qualitative property each PlatEMO problem is famous for:

  * ``IMOP3Like``: a spherical (DTLZ2-style) front with a *non-linear
    density-bias* transform x_i' = x_i^a (a << 1) applied to the position
    variables, exactly the mechanism the real IMOP suite uses (parameters
    a1, a2, a3 in the original paper) to concentrate solutions unevenly
    over an otherwise-regular front.
  * ``IMOP8Like``: a *multi-segment* front, built by partitioning decision
    space into three regions that each map to a different local geometry
    (linear / concave / convex), producing genuine discontinuities between
    segments, again matching the qualitative description used in the
    proposal ("convexa, concava y lineal... dentro del mismo problema").

If bit-exact reproduction of the published IMOP benchmark is required
(e.g. for publication), replace ``IMOP3Like``/``IMOP8Like`` with a direct
port of PlatEMO's ``IMOP3.m``/``IMOP8.m`` (https://github.com/BIMK/PlatEMO,
folder ``PlatEMO/Problems/IMOP``) -- the rest of this codebase only depends
on the generic ``Problem`` interface (``n_var``, ``n_obj``, ``xl``, ``xu``,
``evaluate(X)``) and is agnostic to how the front is generated.
"""

from __future__ import annotations
import numpy as np
from pymoo.problems import get_problem as _pymoo_get_problem


# --------------------------------------------------------------------------
# Generic problem interface used throughout the codebase
# --------------------------------------------------------------------------
class Problem:
    """Minimal problem interface: evaluate(X) -> F, with X, F as 2D arrays."""

    def __init__(self, n_var: int, n_obj: int, xl=0.0, xu=1.0, name: str = ""):
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = np.full(n_var, xl, dtype=float) if np.isscalar(xl) else np.asarray(xl, dtype=float)
        self.xu = np.full(n_var, xu, dtype=float) if np.isscalar(xu) else np.asarray(xu, dtype=float)
        self.name = name

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f"Problem({self.name}, n_var={self.n_var}, n_obj={self.n_obj})"


class PymooWrapper(Problem):
    """Wraps a pymoo Problem so it exposes the .evaluate(X) -> F interface."""

    def __init__(self, pymoo_problem, name: str):
        super().__init__(pymoo_problem.n_var, pymoo_problem.n_obj,
                          pymoo_problem.xl, pymoo_problem.xu, name)
        self._p = pymoo_problem

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        out = {}
        self._p._evaluate(X, out)
        return np.atleast_2d(out["F"])


class MinusWrapper(Problem):
    """Negates the objectives of a base problem -> inverted ('minus') front."""

    def __init__(self, base: Problem, name: str):
        super().__init__(base.n_var, base.n_obj, base.xl, base.xu, name)
        self._base = base

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return -self._base.evaluate(X)


# --------------------------------------------------------------------------
# IMOP-style irregular problems (see module docstring for caveats)
# --------------------------------------------------------------------------
class IMOP3Like(Problem):
    """3-objective spherical front with non-linear density bias (a << 1).

    Decision vector x = (x_I, x_II) with |x_I| = m-1 position vars and
    |x_II| = l distance vars (l=5, m=3 -> n=7, matching Cuadro 1).
    """

    def __init__(self, m: int = 3, l: int = 5, a: float = 0.05):
        n = (m - 1) + l
        super().__init__(n, m, 0.0, 1.0, name=f"IMOP3Like(m={m})")
        self.m = m
        self.l = l
        self.a = a

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        m, l, a = self.m, self.l, self.a
        xI = X[:, : m - 1]
        xII = X[:, m - 1:]
        g = np.sum((xII - 0.5) ** 2, axis=1)  # convergence term, 0 on the PF
        # density-bias warp (the mechanism used by the real IMOP suite)
        y = np.clip(xI, 1e-12, 1.0) ** a
        F = np.zeros((X.shape[0], m))
        cum = np.ones(X.shape[0])
        for i in range(m - 1):
            F[:, i] = (1 + g) * cum * np.cos(y[:, i] * np.pi / 2)
            cum = cum * np.sin(y[:, i] * np.pi / 2)
        F[:, m - 1] = (1 + g) * cum
        return F


class IMOP8Like(Problem):
    """3-objective front split into 3 segments with distinct local geometry
    (linear / concave / convex), selected by the first position variable.
    """

    def __init__(self, m: int = 3, l: int = 5):
        n = (m - 1) + l
        super().__init__(n, m, 0.0, 1.0, name=f"IMOP8Like(m={m})")
        self.m = m
        self.l = l

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        m, l = self.m, self.l
        xI = X[:, : m - 1]
        xII = X[:, m - 1:]
        g = np.sum((xII - 0.5) ** 2, axis=1)
        x1, x2 = xI[:, 0], xI[:, 1] if m > 2 else (xI[:, 0], xI[:, 0])
        N = X.shape[0]
        F = np.zeros((N, m))

        seg = np.clip((x1 * 3).astype(int), 0, 2)  # 0,1,2 -> 3 segments

        # Segment 0: LINEAR (simplex) front, offset down
        mask = seg == 0
        if np.any(mask):
            w0 = x2[mask]
            F[mask, 0] = (1 + g[mask]) * w0
            F[mask, 1] = (1 + g[mask]) * (1 - w0) * 0.5
            F[mask, 2] = (1 + g[mask]) * (1 - w0) * 0.5
            F[mask] += 0.0  # base offset

        # Segment 1: CONCAVE (spherical, DTLZ2-style), shifted
        mask = seg == 1
        if np.any(mask):
            w1 = x2[mask] * np.pi / 2
            r = 1 + g[mask]
            F[mask, 0] = r * np.cos(w1) + 0.4
            F[mask, 1] = r * np.sin(w1) + 0.4
            F[mask, 2] = r * 0.3 + 0.4

        # Segment 2: CONVEX front (inverted sphere), shifted further
        mask = seg == 2
        if np.any(mask):
            w2 = x2[mask] * np.pi / 2
            r = 1 + g[mask]
            f0 = r * np.cos(w2)
            f1 = r * np.sin(w2)
            norm = np.sqrt(f0 ** 2 + f1 ** 2) + 1e-9
            F[mask, 0] = (f0 / norm) + 0.8
            F[mask, 1] = (f1 / norm) + 0.8
            F[mask, 2] = r * 0.2 + 0.8

        return F


# --------------------------------------------------------------------------
# Table 1 (Cuadro 1) registry
# --------------------------------------------------------------------------
_TABLE1_NVAR = {
    "dtlz1": 7,
    "dtlz2": 12,
    "dtlz7": 22,   # <-- Add DTLZ7 (pymoo defaults to 22 variables for 3 objectives)
    "minus-dtlz1": 7,
    "minus-dtlz2": 12,
    "wfg4": 14,
    "wfg9": 14,
    "imop3": 7,
    "imop8": 7,
}


def get_problem(name: str, m: int = 3, n_var: int | None = None) -> Problem:
    """Factory matching Cuadro 1 of the proposal.

    Parameters
    ----------
    name : one of dtlz1, dtlz2, minus-dtlz1, minus-dtlz2, wfg4, wfg9, imop3, imop8
    m    : number of objectives (default 3, as in the proposal)
    n_var: override the number of decision variables (defaults to Cuadro 1)
    """
    key = name.lower()
    if n_var is None:
        n_var = _TABLE1_NVAR[key]

    if key == "dtlz1":
        return PymooWrapper(_pymoo_get_problem("dtlz1", n_var=n_var, n_obj=m), "DTLZ1")
    if key == "dtlz2":
        return PymooWrapper(_pymoo_get_problem("dtlz2", n_var=n_var, n_obj=m), "DTLZ2")
    if key == "dtlz7":  # <-- Add this block
        return PymooWrapper(_pymoo_get_problem("dtlz7", n_var=n_var, n_obj=m), "DTLZ7")
    if key == "minus-dtlz1":
        base = PymooWrapper(_pymoo_get_problem("dtlz1", n_var=n_var, n_obj=m), "DTLZ1")
        return MinusWrapper(base, "Minus-DTLZ1")
    if key == "minus-dtlz2":
        base = PymooWrapper(_pymoo_get_problem("dtlz2", n_var=n_var, n_obj=m), "DTLZ2")
        return MinusWrapper(base, "Minus-DTLZ2")
    if key == "wfg4":
        return PymooWrapper(_pymoo_get_problem("wfg4", n_var=n_var, n_obj=m), "WFG4")
    if key == "wfg9":
        return PymooWrapper(_pymoo_get_problem("wfg9", n_var=n_var, n_obj=m), "WFG9")
    if key == "imop3":
        l = n_var - (m - 1)
        return IMOP3Like(m=m, l=l)
    if key == "imop8":
        l = n_var - (m - 1)
        return IMOP8Like(m=m, l=l)

    raise ValueError(f"Unknown problem '{name}'. Options: {list(_TABLE1_NVAR)}")


PROBLEM_NAMES = list(_TABLE1_NVAR.keys())
