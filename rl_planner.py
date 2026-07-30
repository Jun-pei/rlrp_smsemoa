"""
rl_planner.py
=============
The RL planner of Section 3: replicator dynamics over K=3 magnitude
*regimes* (fine / balanced / coarse, Eq. 9) combined with a Q-learner that
refines the per-dimension (i, s) *sub-action* within the regime selected by
the replicator dynamics (Section 3.3), in the style of NRLPSO [5].

Action space
------------
A pair (i, s), i in {1..m} (dimension), s in {-1,+1} (sign), is encoded as
    j = 2*(i-1) + I[s=+1] + 1   in {1, ..., 2m}
For m=3, 2m=6 sub-actions per regime, 18 (k,j) pairs total.

Reference-point update (projected onto z_ref >= 1, Sec. 2.2):
    z_ref_i(t+1) = max(1, z_ref_i(t) + s * delta_k)
"""

from __future__ import annotations
import numpy as np
from collections import deque


# Three magnitude regimes (delta_1 fine, delta_2 balanced, delta_3 coarse)
def regime_deltas(H: int = 12):
    return np.array([0.01, 1.0 / H, 1.0])


REGIME_NAMES = ["fine", "balanced", "coarse"]
BALANCED_REGIME = 1  # index of the "balanced" regime (k=2 in 1-indexed paper notation)


def encode_action(i: int, s: int, m: int) -> int:
    """i in [0, m-1] (0-indexed dimension), s in {-1, +1} -> j in [0, 2m-1]."""
    return 2 * i + (1 if s == 1 else 0)


def decode_action(j: int):
    """Inverse of encode_action -> (i, s) with i 0-indexed, s in {-1, +1}."""
    i = j // 2
    s = 1 if (j % 2 == 1) else -1
    return i, s


class ReplicatorQPlanner:
    """Replicator-dynamics regime selector + Q-learning direction refiner."""

    def __init__(self, m: int, n_states: int, H: int = 12, W: int = 20,
                 lam: float = 0.9, alpha_rl: float = 0.1, gamma: float = 0.9,
                 eps0: float = 0.1, eps_min: float = 1e-2,
                 sigma0=(0.10, 0.10, 0.80), seed=None,
                 q_reset_every: int = 1000, q_reset_alpha: float = 0.1):
        self.m = m
        self.n_actions_per_regime = 2 * m
        self.n_states = n_states
        self.H = H
        self.deltas = regime_deltas(H)
        self.W = W
        self.lam = lam
        self.alpha_rl = alpha_rl
        self.gamma = gamma
        self.eps = eps0
        self.eps0 = eps0
        self.eps_min = eps_min
        self.q_reset_every = q_reset_every
        self.q_reset_alpha = q_reset_alpha

        self.sigma = np.array(sigma0, dtype=float)
        self.Q = np.zeros((n_states, 3, self.n_actions_per_regime))
        self.rng = np.random.default_rng(seed)

        # Sliding-window EMA trackers per regime for the payoff function (Eq. 7).
        # NOTE: initialized with a small *positive* optimistic baseline rather
        # than exactly 0. With a literal 0 default, Eq. 9's numerator
        # max(sigma_k,eps0)*u_k is exactly 0 for any regime never yet tried,
        # which (once another regime's payoff turns positive) collapses sigma
        # to a one-hot vector permanently -- no amount of eps0-flooring on
        # sigma can rescue a regime whose payoff itself is exactly zero. This
        # optimistic-initialization fix keeps untried regimes "in the running"
        # early on, consistent with the proposal's stated intent that sigma
        # "se redistribuye automaticamente conforme las estimaciones... maduran".
        self.ema_D = np.full(3, 1e-3)
        self.ema_HV = np.full(3, 1e-3)
        self._window_D = [deque(maxlen=W) for _ in range(3)]
        self._window_HV = [deque(maxlen=W) for _ in range(3)]

        self._gen_count = 0

    # ---- payoff function (Eq. 7-8) -------------------------------------
    def beta(self, d_norm: float) -> float:
        return 0.9 - 0.8 * d_norm

    def update_payoff_window(self, active_regime: int, delta_D: float, delta_HV: float):
        """Record (delta_D, delta_HV) for the regime active at t-1 (Sec. 3.1)."""
        self._window_D[active_regime].append(delta_D)
        self._window_HV[active_regime].append(delta_HV)
        # EMA over the sliding window (recompute from the window so W
        # genuinely bounds how far back the smoothing reaches)
        wD = self._window_D[active_regime]
        wHV = self._window_HV[active_regime]
        self.ema_D[active_regime] = _ema(wD, self.lam)
        self.ema_HV[active_regime] = _ema(wHV, self.lam)

    def payoffs(self, d_norm: float) -> np.ndarray:
        b = self.beta(d_norm)
        return b * self.ema_D + (1 - b) * self.ema_HV

    # ---- replicator dynamics (Eq. 9) ------------------------------------
    def replicator_update(self, u: np.ndarray, eps0: float = 1e-3):
        sigma = np.maximum(self.sigma, eps0)
        weighted = sigma * u
        denom = weighted.sum()
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            return  # degenerate payoff vector -> keep sigma unchanged
        new_sigma = weighted / denom
        new_sigma = np.clip(new_sigma, 1e-6, None)
        new_sigma /= new_sigma.sum()
        self.sigma = new_sigma

    def sample_regime(self) -> int:
        return int(self.rng.choice(3, p=self.sigma))

    # ---- Q-learning over (dimension, sign) within a regime ---------------
    def select_subaction(self, state_idx: int, regime: int) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.n_actions_per_regime))
        return int(np.argmax(self.Q[state_idx, regime]))

    def update_q(self, state_idx: int, regime: int, action: int, reward: float,
                 next_state_idx: int, next_regime: int):
        if regime != BALANCED_REGIME:
            return  # only EXPLOIT (balanced regime) updates Q, Eq. 10
        td_target = reward + self.gamma * np.max(self.Q[next_state_idx, next_regime])
        td_error = td_target - self.Q[state_idx, regime, action]
        self.Q[state_idx, regime, action] += self.alpha_rl * td_error

    def decay_epsilon(self, t: int, t_adapt: int):
        frac = min(t / max(t_adapt, 1), 1.0)
        self.eps = self.eps0 + frac * (self.eps_min - self.eps0)

    def maybe_soft_reset_q(self):
        self._gen_count += 1
        if self.q_reset_every and self._gen_count % self.q_reset_every == 0:
            self.Q *= (1 - self.q_reset_alpha)

    # ---- reference point update -------------------------------------------
    def apply_action(self, zref: np.ndarray, regime: int, action: int) -> np.ndarray:
        i, s = decode_action(action)
        delta = self.deltas[regime]
        zref = zref.copy()
        zref[i] = max(1.0, zref[i] + s * delta)
        return zref


def _ema(window: deque, lam: float) -> float:
    """Exponential moving average computed left-to-right over the window."""
    if len(window) == 0:
        return 0.0
    val = window[0]
    for x in list(window)[1:]:
        val = lam * val + (1 - lam) * x
    return float(val)
