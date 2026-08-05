"""
rl_planner.py
=============
Two-level planner: replicator dynamics over K = 3 magnitude regimes (fine /
balanced / coarse) + a tabular Q-learner that picks the (dimension, sign)
sub-action inside the regime chosen by the replicator dynamics.

Action space
------------
A pair (i, s), i in {1..m}, s in {-1,+1}, encoded as j = 2(i-1) + [s=+1], so
2m sub-actions per regime (6 for m = 3, 18 (k,j) pairs in total).

STEP SIZES ARE IN NORMALISED UNITS
----------------------------------
    delta = (delta_fine, delta_balanced, delta_coarse) = (0.01, 1/H, 1.0)

These are displacements **in the normalised objective space** [0,1]^m, where
the estimated nadir sits at (1,...,1) by construction.  So delta = 0.01 is 1%
of the front's own extent, delta = 1/H = 1/12 ~ 0.083 is the classical
Ishibuchi et al. setting (one spacing of a simplex lattice with H divisions),
and delta = 1.0 is one full nadir-to-ideal range per step.

THE REFERENCE POINT IS BOX-BOUNDED
----------------------------------
The update of Sec. 2.2, ``z_i <- max(1, z_i + s*delta)``, is bounded below but
**unbounded above**.  With the coarse regime (delta = 1) a run can push z_ref
to 10^3 or beyond, at which point the HV landscape is essentially flat, every
solution's contribution is dominated by the box term, and SMS-EMOA degenerates
into a boundary-seeking method that spreads the population onto the extremes.
Measured without the bound, z_ref ended runs at (7.0, 9.0, 9.0) and
(3.9, 1.1, 4.6) -- a random walk with +-1 steps, pinned against nothing.  The
update is therefore projected onto a box

    Z = [z_min, z_max]^m,   z_min = 1 + eps_z (default 1.001),
                            z_max = ZREF_MAX  (default 10)

    z_i(t+1) = clip( z_i(t) + s * delta_k ,  z_min , z_max )

The lower bound is strictly *above* 1 so that the extreme solutions of the
front always keep a strictly positive HV contribution (with z_i = 1 exactly, a
solution attaining the nadir in objective i contributes zero volume in that
dimension and would be eliminated first).  The upper bound z_max = 10 caps the
reachable region at 10x the nadir range, i.e. HV can never be inflated more
than 10^m times the [0,1]^m box, so the search for z_ref cannot run away.

Reachability of the box: from z = 1 the coarse regime reaches z_max = 10 in
ceil((10-1)/1) = 9 steps, the balanced regime in 108 steps, the fine regime in
900 steps -- all negligible against rho*T_max ~ 7x10^4 adaptation generations,
so the bound restricts *where the planner can end up*, not *what it can
explore*.
"""

from __future__ import annotations
import numpy as np
from collections import deque

#: default box for the reference point in NORMALISED objective space
ZREF_MIN_EPS = 1e-3
ZREF_MAX = 10.0

REGIME_NAMES = ["fine", "balanced", "coarse"]
BALANCED_REGIME = 1          # index of the "balanced" regime (k=2, 1-indexed)


def regime_deltas(H: int = 12) -> np.ndarray:
    """(delta_fine, delta_balanced, delta_coarse) in normalised units."""
    return np.array([0.01, 1.0 / H, 1.0])


def encode_action(i: int, s: int, m: int) -> int:
    """i in [0, m-1] (0-indexed objective), s in {-1,+1} -> j in [0, 2m-1]."""
    return 2 * i + (1 if s == 1 else 0)


def decode_action(j: int):
    """Inverse of encode_action -> (i, s), i 0-indexed, s in {-1,+1}."""
    return j // 2, (1 if (j % 2 == 1) else -1)


class ReplicatorQPlanner:
    """Replicator-dynamics regime selector + Q-learning direction refiner."""

    def __init__(self, m: int, n_states: int, H: int = 12, W: int = 20,
                 lam: float = 0.9, alpha_rl: float = 0.1, gamma: float = 0.9,
                 eps0: float = 0.1, eps_min: float = 1e-2,
                 sigma0=(0.10, 0.10, 0.80), seed=None,
                 q_reset_every: int = 1000, q_reset_alpha: float = 0.1,
                 zref_max: float = ZREF_MAX, zref_min_eps: float = ZREF_MIN_EPS,
                 q_update_regimes: str = "all", sigma_floor: float = 0.05):
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
        if q_update_regimes not in ("all", "balanced"):
            raise ValueError("q_update_regimes must be 'all' or 'balanced'")
        self.q_update_regimes = q_update_regimes
        self.sigma_floor = float(sigma_floor)
        #: decisions taken on an all-zero Q row, i.e. chosen uniformly at
        #: random -- the diagnostic that exposed the starvation problem
        self.n_decisions = 0
        self.n_blind_decisions = 0

        # --- reference-point box (review comment) --------------------------
        self.zref_min = 1.0 + zref_min_eps
        self.zref_max = float(zref_max)
        if self.zref_max <= self.zref_min:
            raise ValueError("zref_max must exceed 1 + zref_min_eps")

        self.sigma = np.array(sigma0, dtype=float)
        self.sigma /= self.sigma.sum()
        self.Q = np.zeros((n_states, 3, self.n_actions_per_regime))
        #: how many TD updates each (s,k,j) entry received -- lets the runner
        #: report the realised coverage of the Q-table instead of assuming it
        self.visits = np.zeros_like(self.Q, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

        # Sliding-window EMA trackers per regime for the payoff (Eq. 7).
        # Initialised with a small *positive* optimistic baseline rather than
        # exactly 0: with a literal 0, Eq. 9's numerator max(sigma_k,eps0)*u_k
        # is exactly 0 for any regime never yet tried, which (once another
        # regime's payoff turns positive) collapses sigma to a one-hot vector
        # permanently -- no eps0-flooring of sigma can rescue a regime whose
        # payoff itself is exactly zero.
        self.ema_U = np.full(3, 1e-3)     # uniformity term  (Riesz-energy based)
        self.ema_HV = np.full(3, 1e-3)    # convergence term (HV based)
        self._window_U = [deque(maxlen=W) for _ in range(3)]
        self._window_HV = [deque(maxlen=W) for _ in range(3)]
        self._gen_count = 0

    # ---- payoff function (Eq. 7-8) ---------------------------------------
    def beta(self, d_norm: float) -> float:
        """Weight of the uniformity term; decreases as the front spreads out."""
        return 0.9 - 0.8 * d_norm

    def update_payoff_window(self, active_regime: int, delta_U: float, delta_HV: float):
        """Record the (uniformity, HV) improvement credited to regime k_{t-1}.

        ``delta_U`` is the *improvement in uniformity*, i.e. the decrease of
        the normalised Riesz-energy feature E_norm between two consecutive
        generations (positive = the distribution became more uniform).
        """
        self._window_U[active_regime].append(delta_U)
        self._window_HV[active_regime].append(delta_HV)
        self.ema_U[active_regime] = _ema(self._window_U[active_regime], self.lam)
        self.ema_HV[active_regime] = _ema(self._window_HV[active_regime], self.lam)

    def payoffs(self, d_norm: float) -> np.ndarray:
        """u_k = beta(D_norm) * EMA[delta_U]_k + (1 - beta) * EMA[delta_HV]_k."""
        b = self.beta(d_norm)
        return b * self.ema_U + (1 - b) * self.ema_HV

    # ---- replicator dynamics (Eq. 9) --------------------------------------
    def replicator_update(self, u: np.ndarray, eps0: float = 1e-3):
        """Replicator step, then a MUTATION term that keeps sigma interior.

        The plain replicator map sigma <- (sigma*u)/sum(sigma*u) is a
        multiplicative-weights update on the simplex, and the simplex vertices
        are absorbing: once sigma_k gets small, regime k is essentially never
        sampled, so its payoff EMA freezes at whatever stale value it had and
        it can never come back.  Measured on DTLZ2 this is exactly what
        happens -- sigma collapses to (0,0,1) within a few thousand
        generations and stays there for the rest of the run.

        The fix is the standard replicator-MUTATOR equation: mix in a small
        uniform component eta, which bounds sigma_k >= eta/K for every regime
        and therefore keeps every payoff estimate alive.  eta = 0 recovers the
        original (collapsing) behaviour.
        """
        sigma = np.maximum(self.sigma, eps0)
        weighted = sigma * u
        denom = weighted.sum()
        if np.isfinite(denom) and abs(denom) > 1e-12:
            new_sigma = np.clip(weighted / denom, 1e-6, None)
            new_sigma /= new_sigma.sum()
        else:
            new_sigma = self.sigma            # degenerate payoffs -> keep sigma
        eta = self.sigma_floor
        if eta > 0:
            new_sigma = (1.0 - eta) * new_sigma + eta / new_sigma.size
        self.sigma = new_sigma / new_sigma.sum()

    def sample_regime(self) -> int:
        return int(self.rng.choice(3, p=self.sigma))

    # ---- Q-learning over (objective, sign) within a regime -----------------
    def select_subaction(self, state_idx: int, regime: int) -> int:
        self.n_decisions += 1
        q = self.Q[state_idx, regime]
        if not np.any(q):
            # No experience at all for this (state, regime): the choice is
            # uniformly random no matter what epsilon says.  Counting these
            # is how the starvation problem becomes measurable.
            self.n_blind_decisions += 1
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.n_actions_per_regime))
        best = np.flatnonzero(q == q.max())      # random tie-break: with an
        return int(self.rng.choice(best))        # all-zero row argmax always
                                                 # returns action 0 otherwise

    def update_q(self, state_idx: int, regime: int, action: int, reward: float,
                 next_state_idx: int, next_regime: int):
        if self.q_update_regimes == "balanced" and regime != BALANCED_REGIME:
            # Eq. 10 as literally written updates Q only in the balanced
            # regime.  That leaves Q[:, fine, :] and Q[:, coarse, :] exactly
            # zero FOREVER, so whenever the replicator dynamics favours one of
            # those regimes every direction is picked uniformly at random.
            # Measured on DTLZ2 with T_max = 10,000 this made 96.5% of all
            # decisions blind.  ``q_update_regimes="all"`` (the default here)
            # lets each regime learn from its own experience; Q is indexed by
            # regime anyway, so nothing else in the formulation changes.
            return
        td_target = reward + self.gamma * np.max(self.Q[next_state_idx, next_regime])
        self.Q[state_idx, regime, action] += self.alpha_rl * (
            td_target - self.Q[state_idx, regime, action])
        self.visits[state_idx, regime, action] += 1

    def decay_epsilon(self, t: int, t_adapt: int):
        frac = min(t / max(t_adapt, 1), 1.0)
        self.eps = self.eps0 + frac * (self.eps_min - self.eps0)

    def maybe_soft_reset_q(self):
        self._gen_count += 1
        if self.q_reset_every and self._gen_count % self.q_reset_every == 0:
            self.Q *= (1 - self.q_reset_alpha)

    # ---- reference-point update (BOX-BOUNDED) ------------------------------
    def apply_action(self, zref: np.ndarray, regime: int, action: int) -> np.ndarray:
        """z_i(t+1) = clip(z_i(t) + s * delta_k, z_min, z_max), other coords fixed."""
        i, s = decode_action(action)
        z = np.asarray(zref, dtype=float).copy()
        z[i] = np.clip(z[i] + s * self.deltas[regime], self.zref_min, self.zref_max)
        return z

    # ---- diagnostics --------------------------------------------------------
    def q_coverage(self) -> dict:
        """Realised coverage of the Q-table (see the warning in state.py)."""
        v = (self.visits[:, BALANCED_REGIME, :]
             if self.q_update_regimes == "balanced" else self.visits)
        n_entries = v.size
        return dict(n_entries=int(n_entries),
                    n_visited=int(np.count_nonzero(v)),
                    frac_visited=float(np.count_nonzero(v) / max(n_entries, 1)),
                    total_updates=int(v.sum()),
                    mean_visits=float(v.mean()),
                    median_visits=float(np.median(v)),
                    max_visits=int(v.max()) if v.size else 0,
                    n_decisions=int(self.n_decisions),
                    frac_blind=float(self.n_blind_decisions / max(self.n_decisions, 1)))


def _ema(window: deque, lam: float) -> float:
    """Exponential moving average, computed left-to-right over the window."""
    if len(window) == 0:
        return 0.0
    val = window[0]
    for x in list(window)[1:]:
        val = lam * val + (1 - lam) * x
    return float(val)
