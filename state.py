"""
state.py
========
State s_t in R^5 = (Dnorm(t), HVnorm(t), t/Tmax, ghat(t), Enorm(t))  (Sec. 3)

  * Dnorm(t): mean per-dim std-dev of And(t), normalized by the max
    observed so far.
  * HVnorm(t): current HV (w.r.t. the *adaptive* zref) divided by the best
    HV observed so far.
  * ghat(t): contour-vs-interior HV-contribution ratio (Eq. 6).
  * Enorm(t): ratio of consecutive log-Riesz energies (Eq. 5); < 1 means
    the distribution became more uniform since the previous generation.

Discretization (Section 3, "Discretizacion del estado"):
  Dnorm, HVnorm        -> 5 bins each  (B_D, B_H in {0,...,4})
  t/Tmax, ghat, Enorm   -> 3 bins each  (B_t, B_g, B_E in {0,1,2})
  total states = 5*5*3*3*3 = 675
"""

from __future__ import annotations
import numpy as np
from .indicators import (
    mean_dispersion, riesz_energy_log, hv_contributions, hypervolume,
)


def compute_ghat(F_nd: np.ndarray, zref: np.ndarray, eta: float = 1e-6) -> float:
    """g_hat(t), Eq. 6: contour-vs-interior share of HV contribution."""
    n = F_nd.shape[0]
    if n < 3:
        return 0.5
    c = F_nd.mean(axis=0)
    d = np.linalg.norm(F_nd - c, axis=1)
    d_med = np.median(d)
    contour_mask = d >= d_med
    interior_mask = ~contour_mask
    if not np.any(contour_mask) or not np.any(interior_mask):
        return 0.5
    hvc = hv_contributions(F_nd, zref)
    mean_b = hvc[contour_mask].mean()
    mean_i = hvc[interior_mask].mean()
    return float(mean_b / (mean_b + mean_i + eta))


class StateEncoder:
    """Tracks running statistics needed to build s_t and discretize it."""

    def __init__(self, n_bins_continuous=5, n_bins_coarse=3):
        self.n_bins_continuous = n_bins_continuous
        self.n_bins_coarse = n_bins_coarse
        self.best_hv = 1e-12
        self.max_dispersion = 1e-12
        self.prev_eln = None  # E^ln_s(And(t-1))

    def reset(self):
        self.best_hv = 1e-12
        self.max_dispersion = 1e-12
        self.prev_eln = None

    def compute(self, F_nd: np.ndarray, zref: np.ndarray, t: int, t_max: int,
                riesz_s: float = 3.0):
        """Returns (raw_features_dict, discrete_state_index)."""
        hv_now = hypervolume(F_nd, zref)
        self.best_hv = max(self.best_hv, hv_now, 1e-12)
        hv_norm = hv_now / self.best_hv

        disp = mean_dispersion(F_nd)
        self.max_dispersion = max(self.max_dispersion, disp, 1e-12)
        d_norm = disp / self.max_dispersion

        ghat = compute_ghat(F_nd, zref)

        eln = riesz_energy_log(F_nd, s=riesz_s)
        if self.prev_eln is None or self.prev_eln <= 1e-12:
            e_norm = 1.0
        else:
            e_norm = min(eln / self.prev_eln, 1.0)
        self.prev_eln = max(eln, 1e-12)

        frac_t = t / max(t_max, 1)

        features = dict(Dnorm=d_norm, HVnorm=hv_norm, frac_t=frac_t,
                         ghat=ghat, Enorm=e_norm, hv_raw=hv_now)
        state_idx = self.discretize(d_norm, hv_norm, frac_t, ghat, e_norm)
        return features, state_idx

    def discretize(self, d_norm, hv_norm, frac_t, ghat, e_norm) -> int:
        nb, nc = self.n_bins_continuous, self.n_bins_coarse
        bD = _bin(d_norm, 0.0, 1.0, nb)
        bH = _bin(hv_norm, 0.0, 1.0, nb)
        bt = _bin(frac_t, 0.0, 1.0, nc)
        bg = _bin(ghat, 0.0, 1.0, nc)
        bE = _bin(e_norm, 0.0, 1.0, nc)  # <1 improvement, ==1 no-change/worse
        # flatten: index = (((bD*nb + bH)*nc + bt)*nc + bg)*nc + bE
        idx = (((bD * nb + bH) * nc + bt) * nc + bg) * nc + bE
        return int(idx)

    @property
    def n_states(self) -> int:
        return (self.n_bins_continuous ** 2) * (self.n_bins_coarse ** 3)


def _bin(x: float, lo: float, hi: float, n_bins: int) -> int:
    x = min(max(x, lo), hi)
    if hi <= lo:
        return 0
    frac = (x - lo) / (hi - lo)
    b = int(frac * n_bins)
    return min(b, n_bins - 1)
