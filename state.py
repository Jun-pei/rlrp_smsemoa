"""
state.py
========
State s_t = (D_norm, HV_norm, t/T_max, gamma, E_norm) in [0,1]^5, discretised
into a finite index for tabular Q-learning.

EVERY FEATURE IS BOUNDED BY AN ABSOLUTE CONSTANT
------------------------------------------------
None of the features is normalised by a running maximum over the run.  A
running maximum is a statistic of the *trajectory*, not of the population, so
the map (population -> state) would change as the run progressed -- the same
front could be state 137 at t = 1000 and state 42 at t = 50000 -- making the
decision process non-stationary, and tabular Q-learning is only guaranteed to
converge on a stationary MDP.  Instead:

    D_norm(t)  = D(A_nd(t)) / (1/2)                       in [0,1]
                 (1/2 is the exact upper bound of the mean per-objective
                 standard deviation inside [0,1]^m; see indicators.py)
    HV_norm(t) = HV(A_nd(t), z_ref(t)) / prod_i z_ref,i   in [0,1]
                 (normalised by the volume of the box, not by the best HV
                 seen so far)
    gamma(t)   = p/(1+p) in (0,1) from the fitted curvature exponent of
                 sum_i f_i^p = 1  (indicators.estimate_curvature_p)
    E_norm(t)  = clip(E^ln_s(t)/E^ln_s(t-1), 0, 2)/2      in [0,1]
                 (1/2 = no change; > 1/2 = uniformity got worse)

Everything is computed on the objective space already mapped to [0,1]^m by the
ideal/nadir estimated from the non-dominated front.

GEOMETRY: gamma, NOT the contour/interior ratio
-----------------------------------------------
``gamma`` measures the *shape* of the front:

    gamma < 1/2 convex,  gamma = 1/2 linear,  gamma > 1/2 concave.

Measured on the analytical fronts of the benchmark (value of p, |Z| ~ 2000):

    DTLZ1 1.00   DTLZ2 2.00   WFG4 2.01   WFG9 2.00
    Minus-DTLZ1 3.88   Minus-DTLZ2 2.43
    IMOP1 0.25   IMOP2 4.00   IMOP3 0.83   IMOP4 1.59
    IMOP5 1.86   IMOP6 1.75   IMOP7 2.00   IMOP8 1.58

The alternative descriptor of Eq. (6) -- the ratio of HV contributions between
the outer and the inner half of the front -- measures *where the selection
pressure currently is*, not what shape the front has, and it depends on z_ref,
the very thing the planner moves, so it confounds cause and effect.  It is
kept available as ``geometry_mode="contour"`` (and ``"both"``) for the ablation
promised in Sec. 6.1, but it is not the default.

``iota`` (invertedness) is computed and logged as a diagnostic; adding it to
the state would multiply |S| by n_bins_coarse, which the sample-complexity
note below argues against.

Default discretisation
----------------------
    D_norm, HV_norm        -> 5 bins each
    t/T_max, gamma, E_norm -> 3 bins each
    |S| = 5*5*3*3*3 = 675

SAMPLE COMPLEXITY (read this before adding features)
----------------------------------------------------
|S| x K x 2m = 675 x 3 x 6 = 12,150 Q-entries for m = 3, filled during at most
rho*T_max adaptation steps -- on the order of 1-2 updates per entry at
T_max = 100,000.  The table is therefore nearly untrained, and measurements on
DTLZ2 show only 19-43 distinct states are ever visited (effective count 5-8).
``StateEncoder.n_states`` and the planner's ``q_coverage()`` report the
realised numbers so this stays measurable rather than assumed; ``n_bins_fine=3``
(-> |S| = 243) is the knob if coverage is too low.  See section 12 of
RESPUESTAS.md: |S| turned out not to be the binding constraint.
"""

from __future__ import annotations
import numpy as np

from .indicators import (
    compute_ghat, estimate_curvature_p, extent, hypervolume, nondominated,
    normalised_dispersion, riesz_energy_log,
)


def invertedness(F_norm: np.ndarray, q: float = 5.0) -> float:
    """iota in [0,1]: how far the front sits from the axes of the unit box.

    For a *regular* front the extreme (corner) solutions lie on the axes, so
    min_j sum_i f_ij = 1.  For an *inverted* front (Minus-DTLZ) every solution
    is pushed towards the nadir corner and the minimum sum exceeds 1.
    A low percentile is used instead of the true minimum for robustness.

        iota = clip( pct_q(sum_i f_i) - 1, 0, m-1 ) / (m-1)

    Measured: DTLZ1/DTLZ2 ~ 0.00, Minus-DTLZ1 ~ 0.50, Minus-DTLZ2 ~ 0.13.
    """
    F = np.atleast_2d(F_norm)
    m = F.shape[1]
    if F.shape[0] < 2 or m < 2:
        return 0.0
    s = np.percentile(F.sum(axis=1), q)
    return float(np.clip(s - 1.0, 0.0, m - 1.0) / (m - 1.0))


class StateEncoder:
    """Maps (front, z_ref, t) -> continuous features and a discrete index.

    Parameters
    ----------
    geometry_mode : {"curvature", "contour", "both"}
        Which geometry descriptor enters the state.  "curvature" (default)
        uses gamma; "contour" reproduces the old g_hat behaviour; "both"
        multiplies the state count by ``n_bins_coarse``.
    """

    def __init__(self, n_bins_fine: int = 5, n_bins_coarse: int = 3,
                 geometry_mode: str = "curvature", riesz_s: float = 3.0,
                 hv_kwargs: dict | None = None):
        if geometry_mode not in ("curvature", "contour", "both"):
            raise ValueError(f"unknown geometry_mode {geometry_mode!r}")
        self.n_bins_fine = n_bins_fine
        self.n_bins_coarse = n_bins_coarse
        self.geometry_mode = geometry_mode
        self.riesz_s = riesz_s
        self.hv_kwargs = hv_kwargs or {}
        self.prev_eln = None

    def reset(self):
        self.prev_eln = None

    # -- features ----------------------------------------------------------
    def compute(self, F_norm: np.ndarray, zref: np.ndarray, t: int, t_max: int,
                use_front_only: bool = True):
        """Return ``(features, state_index)``.

        ``F_norm`` must already be the population mapped into [0,1]^m by the
        ideal/nadir of the non-dominated front (see indicators.normalize).
        """
        F_norm = np.atleast_2d(F_norm)
        A = F_norm[nondominated(F_norm)] if (use_front_only and F_norm.shape[0] > 1) else F_norm

        # --- HV_norm: normalised by the volume of the box [0, z_ref] --------
        hv_raw = hypervolume(A, zref, **self.hv_kwargs)
        box = float(np.prod(np.maximum(zref, 1e-12)))
        hv_norm = float(np.clip(hv_raw / box, 0.0, 1.0))

        # --- D_norm: absolute bound 1/2 (no running maximum) ----------------
        d_norm = normalised_dispersion(A)

        # --- geometry --------------------------------------------------------
        p_hat = estimate_curvature_p(A)
        gamma = p_hat / (1.0 + p_hat)
        iota = invertedness(A)
        ghat = compute_ghat(A, zref, **self.hv_kwargs) if self.geometry_mode in ("contour", "both") else float("nan")

        # --- E_norm: ratio of consecutive log-energies, mapped to [0,1] ------
        # E_norm = 1/2 means "no change"; < 1/2 more uniform than the previous
        # generation, > 1/2 less uniform.  Eq. (5) as written clips the ratio
        # at 1.0, which makes a DEGRADATION of uniformity unobservable -- the
        # planner could never see that a move had made the distribution worse,
        # so the two-sided mapping is used instead.
        eln = riesz_energy_log(A, s=self.riesz_s)
        if self.prev_eln is None or self.prev_eln <= 1e-12:
            e_norm, ratio = 0.5, 1.0
        else:
            ratio = eln / self.prev_eln
            e_norm = float(np.clip(ratio, 0.0, 2.0) / 2.0)
        self.prev_eln = max(eln, 1e-12)

        frac_t = t / max(t_max, 1)

        feats = dict(Dnorm=d_norm, HVnorm=hv_norm, frac_t=frac_t, gamma=gamma,
                     Enorm=e_norm, iota=iota, ghat=ghat, p_hat=p_hat,
                     hv_raw=hv_raw, eln=eln, e_ratio=ratio,
                     extent=extent(A), n_nd=int(A.shape[0]))
        return feats, self.discretize(feats)

    # -- discretisation ----------------------------------------------------
    def discretize(self, f: dict) -> int:
        nf, nc = self.n_bins_fine, self.n_bins_coarse
        idx = _bin(f["Dnorm"], nf)
        idx = idx * nf + _bin(f["HVnorm"], nf)
        idx = idx * nc + _bin(f["frac_t"], nc)
        if self.geometry_mode in ("curvature", "both"):
            idx = idx * nc + _bin(f["gamma"], nc)
        if self.geometry_mode in ("contour", "both"):
            idx = idx * nc + _bin(f["ghat"], nc)
        idx = idx * nc + _bin(f["Enorm"], nc)
        return int(idx)

    @property
    def n_states(self) -> int:
        n_geo = 2 if self.geometry_mode == "both" else 1
        return (self.n_bins_fine ** 2) * (self.n_bins_coarse ** (2 + n_geo))

    @property
    def feature_names(self):
        return ["Dnorm", "HVnorm", "frac_t", "gamma", "Enorm", "iota", "ghat",
                "p_hat", "extent", "n_nd"]


def _bin(x: float, n_bins: int) -> int:
    """Uniform binning of x in [0,1] into n_bins."""
    if not np.isfinite(x):
        return 0
    return int(min(max(x, 0.0) * n_bins, n_bins - 1))
