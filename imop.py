"""
imop.py
=======
Faithful Python port of the **IMOP1--IMOP8** benchmark suite of

    Y. Tian, R. Cheng, X. Zhang, M. Li, Y. Jin, "Diversity assessment of
    multi-objective evolutionary algorithms: Performance metric and benchmark
    problems", IEEE Computational Intelligence Magazine, 14(3):61-74, 2019.

transcribed one-to-one from the reference MATLAB implementation shipped with
PlatEMO (``PlatEMO/Problems/Multi-objective optimization/IMOP/IMOP*.m``,
https://github.com/BIMK/PlatEMO).  Both ``CalObj`` (objective function) and
``GetOptimum`` (analytical Pareto-front sampler, used as the IGD+ reference
set Z) are ported.

Default settings, exactly as in PlatEMO
---------------------------------------
    IMOP1, IMOP2, IMOP3 : m = 2,  D = 10,  K = 5,  a1 = 0.05
    IMOP4               : m = 3,  D = 10,  K = 5,  a1 = 0.05
    IMOP5..IMOP8        : m = 3,  D = 10,  K = 5,  a1 = 0.05,  a2 = 10

The number of objectives m is FIXED by the benchmark definition (the suite is
not parameterised in m); asking for a different m raises an error rather than
silently fabricating a variant.  The number of decision variables D is free
(D > K), so the distance-variable count D - K can be varied.

Structure of every IMOP problem
-------------------------------
    y  = shape variables obtained from x_{1..K} through a *density-bias*
         power transform (y = mean(x)^a), which is what makes the density of
         solutions on the front non-uniform;
    g  = sum_{i=K+1}^{D} (x_i - 0.5)^2, the distance function (g = 0 <=> PF).

IMOP1/2 give convex/concave 2-D fronts with strongly biased density, IMOP3 a
disconnected 2-D front, IMOP4 a 3-D "wavy" curve-like front, IMOP5 eight
disconnected patches, IMOP6 a grid-like front with holes, IMOP7 a spherical
front reduced to a thin band, and IMOP8 a highly multimodal degenerate front.
"""

from __future__ import annotations
import numpy as np

from .problems import Problem


def _nd_filter(F: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Keep only the non-dominated rows of F (minimisation).

    Exact, but memory-bounded: pymoo's non-dominated sorting is O(n^2) in both
    time and memory, which is prohibitive for the 10^5-point grids needed to
    resolve the disconnected fronts of IMOP3/6/8.  A globally non-dominated
    point is non-dominated inside *any* subset containing it, so filtering
    block-by-block and then filtering the union of the survivors returns
    exactly the same set as a single global pass.
    """
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    if F.shape[0] == 0:
        return F
    nds = NonDominatedSorting()
    while True:
        if F.shape[0] <= chunk:
            return F[nds.do(F, only_non_dominated_front=True)]
        parts = [F[i:i + chunk] for i in range(0, F.shape[0], chunk)]
        kept = np.vstack([p[nds.do(p, only_non_dominated_front=True)] for p in parts])
        if kept.shape[0] == F.shape[0]:      # nothing eliminated -> converged
            return kept
        F = kept


def _dense_nd(sampler, n_points: int, max_raw: int = 400_000) -> np.ndarray:
    """Grow the raw sample until >= n_points survive the non-dominated filter.

    Needed for IMOP3/6/8, whose *analytical* parameterisation covers a surface
    of which only a small (and highly disconnected) subset is non-dominated:
    generating exactly ``n_points`` raw samples would leave an IGD+ reference
    set an order of magnitude too sparse to be trustworthy.
    """
    raw = max(n_points, 1000)
    best = np.empty((0, 1))
    while raw <= max_raw:
        F = _nd_filter(sampler(raw))
        best = F
        if F.shape[0] >= n_points:
            break
        raw *= 4
    if best.shape[0] > n_points:                      # thin out, keep spread
        sel = np.linspace(0, best.shape[0] - 1, n_points).astype(int)
        best = best[sel]
    return best


class _IMOPBase(Problem):
    """Common scaffolding: bounds [0,1]^D, K shape vars, D-K distance vars."""

    M_FIXED: int = 2

    def __init__(self, n_var: int = 10, K: int = 5, a1: float = 0.05,
                 a2: float = 10.0, m: int | None = None, name: str = ""):
        if m is not None and m != self.M_FIXED:
            raise ValueError(
                f"{name} is defined for m={self.M_FIXED} objectives only "
                f"(IMOP is not parameterised in m); got m={m}.")
        if n_var <= K:
            raise ValueError(f"n_var must be > K={K}; got {n_var}.")
        super().__init__(n_var, self.M_FIXED, 0.0, 1.0, name=name)
        self.K = K
        self.a1 = a1
        self.a2 = a2

    # -- shape variables ---------------------------------------------------
    def _y_single(self, X):
        """y1 = mean(x_1..x_K)^a1   (IMOP1--IMOP4)."""
        return X[:, : self.K].mean(axis=1) ** self.a1

    def _y_pair(self, X):
        """y1 = mean(x_1,x_3,x_5,..)^a1 ; y2 = mean(x_2,x_4,..)^a2  (IMOP5--8)."""
        y1 = X[:, 0: self.K: 2].mean(axis=1) ** self.a1
        y2 = X[:, 1: self.K: 2].mean(axis=1) ** self.a2
        return y1, y2

    def _g(self, X):
        return np.sum((X[:, self.K:] - 0.5) ** 2, axis=1)

    def pareto_front(self, n_points: int = 5000) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# IMOP1--IMOP3 : m = 2
# ---------------------------------------------------------------------------
class IMOP1(_IMOPBase):
    """CONVEX 2-D front, f = (cos^8, sin^8), strong density bias.

    The eighth power pulls the front well inside the linear one (it passes
    through f = (0.0625, 0.0625)); the fitted curvature exponent is p = 0.25.
    """
    M_FIXED = 2

    def __init__(self, **kw):
        super().__init__(name="IMOP1", **kw)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        y1, g = self._y_single(X), self._g(X)
        F = np.empty((X.shape[0], 2))
        F[:, 0] = g + np.cos(y1 * np.pi / 2) ** 8
        F[:, 1] = g + np.sin(y1 * np.pi / 2) ** 8
        return F

    def pareto_front(self, n_points: int = 5000):
        x = np.linspace(0.5 ** 4, 1.0, n_points // 2)
        f1 = np.concatenate([np.flip((1 - x ** 0.25) ** 4), x])
        f2 = np.concatenate([np.flip(x), (1 - x ** 0.25) ** 4])
        return np.column_stack([f1, f2])


class IMOP2(_IMOPBase):
    """CONCAVE 2-D front, f = (cos^0.5, sin^0.5), strong density bias.

    Mirror image of IMOP1: the square root pushes the front outside the linear
    one, and the fitted curvature exponent is p = 4.0.
    """
    M_FIXED = 2

    def __init__(self, **kw):
        super().__init__(name="IMOP2", **kw)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        y1, g = self._y_single(X), self._g(X)
        F = np.empty((X.shape[0], 2))
        F[:, 0] = g + np.cos(y1 * np.pi / 2) ** 0.5
        F[:, 1] = g + np.sin(y1 * np.pi / 2) ** 0.5
        return F

    def pareto_front(self, n_points: int = 5000):
        x = np.linspace(0.0, 0.5 ** 0.25, n_points // 2)
        f1 = np.concatenate([x, np.flip((1 - x ** 4) ** 0.25)])
        f2 = np.concatenate([(1 - x ** 4) ** 0.25, np.flip(x)])
        return np.column_stack([f1, f2])


class IMOP3(_IMOPBase):
    """DISCONNECTED 2-D front (cosine ripple), m = 2 as in the original paper."""
    M_FIXED = 2

    def __init__(self, **kw):
        super().__init__(name="IMOP3", **kw)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        y1, g = self._y_single(X), self._g(X)
        F = np.empty((X.shape[0], 2))
        F[:, 0] = g + (1 + np.cos(y1 * np.pi * 10) / 5 - y1)
        F[:, 1] = g + y1
        return F

    def pareto_front(self, n_points: int = 5000):
        def sampler(n):
            x = np.linspace(0.0, 1.0, n)
            return np.column_stack([1 + np.cos(x * np.pi * 10) / 5 - x, x])
        return _dense_nd(sampler, n_points)


# ---------------------------------------------------------------------------
# IMOP4--IMOP8 : m = 3
# ---------------------------------------------------------------------------
class IMOP4(_IMOPBase):
    """3-D degenerate (curve-like) front with a sinusoidal ripple."""
    M_FIXED = 3

    def __init__(self, **kw):
        super().__init__(name="IMOP4", **kw)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        y1, g = self._y_single(X), self._g(X)
        F = np.empty((X.shape[0], 3))
        F[:, 0] = (1 + g) * y1
        F[:, 1] = (1 + g) * (y1 + np.sin(10 * np.pi * y1) / 10)
        F[:, 2] = (1 + g) * (1 - y1)
        return F

    def pareto_front(self, n_points: int = 5000):
        f1 = np.linspace(0.0, 1.0, n_points)
        F = np.column_stack([f1, f1 + np.sin(10 * np.pi * f1) / 10, 1 - f1])
        return F


class IMOP5(_IMOPBase):
    """Front made of EIGHT disconnected circular patches on a plane."""
    M_FIXED = 3

    def __init__(self, **kw):
        super().__init__(name="IMOP5", **kw)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        y1, y2 = self._y_pair(X)
        g = self._g(X)
        F = np.empty((X.shape[0], 3))
        F[:, 0] = 0.4 * np.cos(np.pi * np.ceil(y1 * 8) / 4) + 0.1 * y2 * np.cos(16 * np.pi * y1)
        F[:, 1] = 0.4 * np.sin(np.pi * np.ceil(y1 * 8) / 4) + 0.1 * y2 * np.sin(16 * np.pi * y1)
        F[:, 2] = 0.5 - F[:, 0] - F[:, 1]
        return F + g[:, None]

    def pareto_front(self, n_points: int = 5000):
        side = int(np.ceil(np.sqrt(n_points / 8 * 1.3)))
        gx, gy = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
        pts = np.column_stack([gx.ravel(order="F"), gy.ravel(order="F")]) - 0.5
        pts = 0.2 * pts[np.sum(pts ** 2, axis=1) <= 0.25]
        ang = np.arange(1, 9) * np.pi / 4
        centres = np.column_stack([0.4 * np.cos(ang), 0.4 * np.sin(ang)])
        R = np.vstack([pts + c for c in centres])
        return np.column_stack([R, 0.5 - R.sum(axis=1)])


class IMOP6(_IMOPBase):
    """Planar front punctured by a periodic grid of HOLES."""
    M_FIXED = 3

    def __init__(self, **kw):
        super().__init__(name="IMOP6", **kw)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        y1, y2 = self._y_pair(X)
        g = self._g(X)
        r = np.maximum(0.0, np.minimum(np.sin(3 * np.pi * y1) ** 2,
                                       np.sin(3 * np.pi * y2) ** 2) - 0.05)
        F = np.empty((X.shape[0], 3))
        F[:, 0] = (1 + g) * y1 + np.ceil(r)
        F[:, 1] = (1 + g) * y2 + np.ceil(r)
        F[:, 2] = (0.5 + g) * (2 - y1 - y2) + np.ceil(r)
        return F

    def pareto_front(self, n_points: int = 5000):
        def sampler(n):
            side = int(np.ceil(np.sqrt(n)))
            gx, gy = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
            R = np.column_stack([gx.ravel(order="F"), gy.ravel(order="F")])
            r = np.maximum(0.0, np.minimum(np.sin(3 * np.pi * R[:, 0]) ** 2,
                                           np.sin(3 * np.pi * R[:, 1]) ** 2) - 0.05)
            return np.column_stack([R, 1 - R.sum(axis=1) / 2]) + np.ceil(r)[:, None]
        return _dense_nd(sampler, n_points)


class IMOP7(_IMOPBase):
    """Spherical front reduced to a thin BAND around the f1=f2=f3 diagonal."""
    M_FIXED = 3

    def __init__(self, **kw):
        super().__init__(name="IMOP7", **kw)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        y1, y2 = self._y_pair(X)
        g = self._g(X)
        F = np.empty((X.shape[0], 3))
        F[:, 0] = (1 + g) * np.cos(y1 * np.pi / 2) * np.cos(y2 * np.pi / 2)
        F[:, 1] = (1 + g) * np.cos(y1 * np.pi / 2) * np.sin(y2 * np.pi / 2)
        F[:, 2] = (1 + g) * np.sin(y1 * np.pi / 2)
        r = np.minimum(np.minimum(np.abs(F[:, 0] - F[:, 1]),
                                  np.abs(F[:, 1] - F[:, 2])),
                       np.abs(F[:, 2] - F[:, 0]))
        return F + (10 * np.maximum(0.0, r - 0.1))[:, None]

    def pareto_front(self, n_points: int = 5000):
        from pymoo.util.ref_dirs import get_reference_directions
        # PlatEMO uses UniformPoint(N,3) == Das-Dennis simplex lattice
        n_part = 2
        while (n_part + 1) * (n_part + 2) // 2 < n_points:
            n_part += 1
        R = get_reference_directions("das-dennis", 3, n_partitions=n_part).astype(float)
        R = R / np.linalg.norm(R, axis=1, keepdims=True)
        r = np.minimum(np.minimum(np.abs(R[:, 0] - R[:, 1]),
                                  np.abs(R[:, 1] - R[:, 2])),
                       np.abs(R[:, 2] - R[:, 0]))
        return R[r <= 0.1]


class IMOP8(_IMOPBase):
    """Highly MULTIMODAL front: f3 = 3 - sum f_i (1 + sin(19 pi f_i))."""
    M_FIXED = 3

    def __init__(self, **kw):
        super().__init__(name="IMOP8", **kw)

    def evaluate(self, X):
        X = np.atleast_2d(X)
        y1, y2 = self._y_pair(X)
        g = self._g(X)
        F = np.empty((X.shape[0], 3))
        F[:, 0] = y1
        F[:, 1] = y2
        inner = F[:, :2] / (1 + g)[:, None]
        F[:, 2] = (1 + g) * (3 - np.sum(inner * (1 + np.sin(19 * np.pi * F[:, :2])), axis=1))
        return F

    def pareto_front(self, n_points: int = 5000):
        def sampler(n):
            side = int(np.ceil(np.sqrt(n)))
            gx, gy = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
            R = np.column_stack([gx.ravel(order="F"), gy.ravel(order="F")])
            f3 = 3 - np.sum(R * (1 + np.sin(19 * np.pi * R)), axis=1)
            return np.column_stack([R, f3])
        return _dense_nd(sampler, n_points)


IMOP_CLASSES = {
    "imop1": IMOP1, "imop2": IMOP2, "imop3": IMOP3, "imop4": IMOP4,
    "imop5": IMOP5, "imop6": IMOP6, "imop7": IMOP7, "imop8": IMOP8,
}

IMOP_NAMES = list(IMOP_CLASSES)

#: number of objectives fixed by the benchmark definition
IMOP_M = {k: cls.M_FIXED for k, cls in IMOP_CLASSES.items()}


def get_imop(name: str, n_var: int = 10, K: int = 5) -> _IMOPBase:
    key = name.lower()
    if key not in IMOP_CLASSES:
        raise ValueError(f"Unknown IMOP problem '{name}'. Options: {IMOP_NAMES}")
    return IMOP_CLASSES[key](n_var=n_var, K=K)
