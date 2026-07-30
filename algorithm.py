"""
algorithm.py
============
RL-RP-SMS-EMOA (Algorithm 1 of the proposal) plus the three baseline
reference-point schemes used in the comparison of Section 5.3:

  * SMS-EMOA_nadir     : z_ref fixed = nadir_hat + 0.01 * 1
  * SMS-EMOA_balanced  : z_ref fixed = nadir_hat + (1/H) * 1   [6]
  * SMS-EMOA_dyn-lin   : z_ref grows linearly from nadir_hat+0.01*1 to
                          nadir_hat+1.0*1 over the run               [7]
  * RL-RP-SMS-EMOA     : the proposed method (replicator dynamics +
                          Q-learning planner, two-phase schedule)

All four share the same SMS-EMOA stepping primitives (sms_emoa.py) so that
differences in the results are attributable only to the PR mechanism.

A note on the Q-learning update (Eq. 10 / Algorithm 1, line 25)
-----------------------------------------------------------------
The pseudocode bootstraps Q[s_t, (k_t,j_t)] against Q[s_{t+1}, (k_{t+1}, .)],
i.e. it needs the regime k_{t+1} that the replicator dynamics will choose in
the *next* generation before it can finish updating the value for the
*current* generation. We implement this with a one-step-deferred update:
the (s_t, k_t, j_t, r_t) tuple is stashed, and the TD update is completed
at the start of generation t+1 as soon as (s_{t+1}, k_{t+1}) are known. This
preserves the exact dependency structure of Eq. 10 without changing its
semantics.
"""

from __future__ import annotations
import numpy as np

from .problems import Problem
from .indicators import (
    estimate_ideal_nadir, normalize, hypervolume, igd_plus, riesz_energy_log,
    mean_dispersion,
)
from .sms_emoa import make_offspring, sms_emoa_eliminate
from .state import StateEncoder
from .rl_planner import ReplicatorQPlanner, BALANCED_REGIME


DEFAULTS = dict(
    mu=100, pc=1.0, eta_c=15.0, eta_m=20.0,
    H=12, rho=0.7, W=20, eps0=0.1, eps_min=1e-2,
    sigma0=(0.10, 0.10, 0.80), alpha_reward=1.0, lam=0.9,
    alpha_rl=0.1, gamma=0.9, q_reset_every=1000, q_reset_alpha=0.1,
)


class History:
    """Per-generation diagnostics (Sec. 5.1 / 5.2)."""

    def __init__(self):
        self.hv_adaptive = []      # HV w.r.t. the (possibly adaptive) zref_norm
        self.hv_ext_far = []       # HV w.r.t. fixed external zref (nadir_true+1.1)
        self.hv_ext_near = []      # HV w.r.t. fixed external zref (nadir_true+0.1)
        self.igd_plus = []
        self.riesz_log = []
        self.dispersion = []
        self.zref = []
        self.sigma = []
        self.regime = []
        self.reward = []


def _evaluate_external_indicators(F_raw, true_front, nadir_true):
    hv_far = hypervolume(F_raw, nadir_true + 1.1)
    hv_near = hypervolume(F_raw, nadir_true + 0.1)
    igd = igd_plus(F_raw, true_front) if true_front is not None else float("nan")
    return hv_far, hv_near, igd


def _init_population(problem: Problem, mu: int, rng: np.random.Generator):
    X = rng.random((mu, problem.n_var)) * (problem.xu - problem.xl) + problem.xl
    F = problem.evaluate(X)
    return X, F


def _record_common(hist: History, F_raw, ideal, nadir, zref_norm, true_front, nadir_true, riesz_s=3.0):
    F_norm = normalize(F_raw, ideal, nadir)
    hist.hv_adaptive.append(hypervolume(F_norm, zref_norm))
    hv_far, hv_near, igd = _evaluate_external_indicators(F_raw, true_front, nadir_true)
    hist.hv_ext_far.append(hv_far)
    hist.hv_ext_near.append(hv_near)
    hist.igd_plus.append(igd)
    hist.riesz_log.append(riesz_energy_log(F_norm, s=riesz_s))
    hist.dispersion.append(mean_dispersion(F_norm))


def _sms_emoa_generation(problem, X, F, zref_norm, ideal, nadir, cfg, rng):
    pm = 1.0 / problem.n_var
    child_x = make_offspring(X, problem.xl, problem.xu, cfg["pc"], cfg["eta_c"],
                              pm, cfg["eta_m"], rng)
    child_f = problem.evaluate(child_x.reshape(1, -1))
    X2 = np.vstack([X, child_x])
    F2 = np.vstack([F, child_f])
    return sms_emoa_eliminate(X2, F2, zref_norm, ideal, nadir)


# ============================================================================
# RL-RP-SMS-EMOA  (Algorithm 1)
# ============================================================================
class RLRPSMSEMOA:
    def __init__(self, problem: Problem, t_max: int, seed: int = 0,
                 true_front=None, nadir_true=None, **kwargs):
        self.problem = problem
        self.t_max = t_max
        self.cfg = {**DEFAULTS, **kwargs}
        self.rng = np.random.default_rng(seed)
        self.true_front = true_front
        self.nadir_true = nadir_true
        self.t_adapt = int(np.ceil(self.cfg["rho"] * t_max))

        self.state_encoder = StateEncoder()
        self.planner = ReplicatorQPlanner(
            m=problem.n_obj, n_states=self.state_encoder.n_states,
            H=self.cfg["H"], W=self.cfg["W"], lam=self.cfg["lam"],
            alpha_rl=self.cfg["alpha_rl"], gamma=self.cfg["gamma"],
            eps0=self.cfg["eps0"], eps_min=self.cfg["eps_min"],
            sigma0=self.cfg["sigma0"], seed=seed,
            q_reset_every=self.cfg["q_reset_every"],
            q_reset_alpha=self.cfg["q_reset_alpha"],
        )
        self.history = History()
        self._last_enorm = None
        self._last_hvnorm = None

    def run(self):
        cfg = self.cfg
        mu = cfg["mu"]
        X, F = _init_population(self.problem, mu, self.rng)
        ideal, nadir = estimate_ideal_nadir(F)
        if self.nadir_true is None:
            self.nadir_true = nadir.copy()

        zref_norm = np.ones(self.problem.n_obj)  # zref(0) = nadir_hat + 1 -> (1,...,1) once normalized

        prev_regime = None        # k_{t-1}, for payoff attribution (Sec 3.1)
        pending = None            # (s_idx, k, j, reward) awaiting bootstrap target

        for t in range(1, self.t_max + 1):
            ideal, nadir = estimate_ideal_nadir(F)
            adapting = t <= self.t_adapt

            if adapting:
                feats, s_idx = self.state_encoder.compute(
                    normalize(F, ideal, nadir), zref_norm, t, self.t_max)

                # --- payoff bookkeeping & replicator dynamics (Eq. 7-9) -----
                if prev_regime is not None:
                    delta_d = self._last_enorm - feats["Enorm"]
                    delta_hv = feats["HVnorm"] - self._last_hvnorm
                    self.planner.update_payoff_window(prev_regime, delta_d, delta_hv)
                u = self.planner.payoffs(feats["Dnorm"])
                # Eq. 9's floor max(sigma_k(t), eps0) shares the *same*,
                # linearly-decaying eps0 used for eps-greedy Q action
                # selection (Sec. 2.2, suposicion (iv)): early on this keeps
                # every regime's weight >= eps0*u_k so none can be starved
                # out of exploration; as eps decays the floor relaxes and
                # the replicator dynamics is allowed to fully consolidate.
                self.planner.replicator_update(u, eps0=self.planner.eps)
                k_t = self.planner.sample_regime()

                # --- finish deferred Q-update from generation t-1 now that
                #     (s_t, k_t) -- the bootstrap target -- are known --------
                if pending is not None:
                    s_prev, k_prev, j_prev, r_prev = pending
                    self.planner.update_q(s_prev, k_prev, j_prev, r_prev, s_idx, k_t)
                    pending = None

                j_t = self.planner.select_subaction(s_idx, k_t)
                zref_norm = self.planner.apply_action(zref_norm, k_t, j_t)

                self.history.sigma.append(self.planner.sigma.copy())
                self.history.regime.append(k_t)
                self._last_enorm = feats["Enorm"]
                self._last_hvnorm = feats["HVnorm"]
                prev_regime = k_t
            else:
                k_t = j_t = s_idx = feats = None

            self.history.zref.append(zref_norm.copy())

            # ---- one SMS-EMOA generation -----------------------------------
            X, F = _sms_emoa_generation(self.problem, X, F, zref_norm, ideal, nadir, cfg, self.rng)

            ideal2, nadir2 = estimate_ideal_nadir(F)
            _record_common(self.history, F, ideal2, nadir2, zref_norm,
                            self.true_front, self.nadir_true)

            if adapting:
                # state AFTER this generation's reproduction step, used both
                # as r_t's "next" snapshot and as next iteration's s_{t+1}
                feats_after, _ = self.state_encoder.compute(
                    normalize(F, ideal2, nadir2), zref_norm, t, self.t_max)
                reward = (feats_after["HVnorm"] - feats["HVnorm"]) + \
                         cfg["alpha_reward"] * (feats["Enorm"] - feats_after["Enorm"])
                self.history.reward.append(reward)
                pending = (s_idx, k_t, j_t, reward)

                self.planner.decay_epsilon(t, self.t_adapt)
                self.planner.maybe_soft_reset_q()

        return X, F, self.history


# ============================================================================
# Baselines (Sec. 5.3)
# ============================================================================
def _run_baseline(problem: Problem, t_max: int, mode: str, seed: int = 0,
                   true_front=None, nadir_true=None, **kwargs):
    cfg = {**DEFAULTS, **kwargs}
    rng = np.random.default_rng(seed)
    mu = cfg["mu"]
    X, F = _init_population(problem, mu, rng)
    ideal, nadir = estimate_ideal_nadir(F)
    if nadir_true is None:
        nadir_true = nadir.copy()
    H = cfg["H"]
    history = History()

    for t in range(1, t_max + 1):
        ideal, nadir = estimate_ideal_nadir(F)

        if mode == "nadir":
            zref_norm = np.ones(problem.n_obj) + 0.01
        elif mode == "balanced":
            zref_norm = np.ones(problem.n_obj) + 1.0 / H
        elif mode == "dynlin":
            frac = t / t_max
            offset = 0.01 + frac * (1.0 - 0.01)
            zref_norm = np.ones(problem.n_obj) + offset
        else:
            raise ValueError(mode)

        history.zref.append(zref_norm.copy())
        X, F = _sms_emoa_generation(problem, X, F, zref_norm, ideal, nadir, cfg, rng)

        ideal2, nadir2 = estimate_ideal_nadir(F)
        _record_common(history, F, ideal2, nadir2, zref_norm, true_front, nadir_true)

    return X, F, history


def run_sms_emoa_nadir(problem, t_max, seed=0, **kw):
    return _run_baseline(problem, t_max, "nadir", seed, **kw)


def run_sms_emoa_balanced(problem, t_max, seed=0, **kw):
    return _run_baseline(problem, t_max, "balanced", seed, **kw)


def run_sms_emoa_dynlin(problem, t_max, seed=0, **kw):
    return _run_baseline(problem, t_max, "dynlin", seed, **kw)


def run_rl_rp_sms_emoa(problem, t_max, seed=0, **kw):
    algo = RLRPSMSEMOA(problem, t_max, seed=seed, **kw)
    return algo.run()


METHODS = {
    "SMS-EMOA_nadir": run_sms_emoa_nadir,
    "SMS-EMOA_balanced": run_sms_emoa_balanced,
    "SMS-EMOA_dynlin": run_sms_emoa_dynlin,
    "RL-RP-SMS-EMOA": run_rl_rp_sms_emoa,
}
