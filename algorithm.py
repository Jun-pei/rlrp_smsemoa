"""
algorithm.py
============
RL-RP-SMS-EMOA (Algorithm 1) and the three fixed/scheduled reference-point
baselines it is compared against:

  * SMS-EMOA_nadir    : z_ref = nadir_hat + 0.01 * 1        (constant)
  * SMS-EMOA_balanced : z_ref = nadir_hat + (1/H) * 1       (constant)
  * SMS-EMOA_dynlin   : z_ref grows linearly 0.01 -> 1.0 over the run
  * RL-RP-SMS-EMOA    : the proposed planner

All four share the same SMS-EMOA primitives (sms_emoa.py), so any difference
is attributable to the reference-point mechanism alone.

WHAT CHANGED
------------
1. EVERY GENERATION IS NOW LOGGED (review comment).  ``History`` records one
   row per generation with the state features, the reference point, the
   replicator weights, the chosen (regime, sub-action), the reward, and the
   quality indicators.  ``History.to_dataframe()`` returns a tidy frame and
   ``history_io.py`` writes it to disk.  The expensive external indicators
   (HVR / IGD+ / energy ratio against the true front) are controlled by
   ``eval_every``: with ``eval_every=1`` literally every generation is
   evaluated, which is what was asked for, but note that at T_max = 100,000
   this dominates the runtime -- see the note in run_full_experiment.py.

1b. ``action_every`` (tau) decouples the PLANNER's timescale from the
   generation loop.  This is a substantive fix, not a speed knob.  In a
   steady-state EMOA one generation creates ONE offspring and deletes ONE
   individual, so the one-step reward is dominated by the noise of the
   variation operators, while the effect of moving z_ref only materialises
   over a full population turnover (~mu generations).  Measured on DTLZ2, the
   Spearman correlation between r_t and the actual subsequent change in HVR is
   +0.008 -- i.e. the per-generation reward carries no usable signal.  With
   ``action_every=tau`` the planner acts once every tau generations and the
   reward is measured over that whole window, which raises the signal-to-noise
   ratio by roughly sqrt(tau) and aligns the credit-assignment horizon with
   the action's actual effect.  Recommended: tau ~ mu.

2. The reference point is confined to the box [1+eps, z_max]^m (rl_planner.py).

3. Quality is measured against the PROBLEM-level ``ReferenceFrame``
   (performance.py), i.e. a fixed external reference point and the true
   ideal/nadir, never the algorithm's own moving z_ref.

4. ``reward_hv`` controls which HV enters the reward:
     "fixed"    (default) HV against a FIXED point (1+kappa)*1 in the
                algorithm's own estimated normalised frame;
     "adaptive" HV against the moving z_ref, as originally written.
   The original choice is confounded: HV(A, z_ref)/vol(z_ref) changes when
   z_ref moves even if the population is untouched, so the planner was partly
   rewarded for moving the goalposts rather than for improving the front.
   Both are implemented so the two can be compared empirically.

A note on the Q-learning update (Eq. 10)
----------------------------------------
The pseudocode bootstraps Q[s_t,(k_t,j_t)] against Q[s_{t+1},(k_{t+1}, .)],
so it needs the regime the replicator dynamics will pick at t+1 before it can
finish the update for t.  This is implemented as a one-step-deferred update:
(s_t, k_t, j_t) is stashed and the TD update completed at the start of
generation t+1, once (s_{t+1}, k_{t+1}) and hence r_t are known.  This
preserves the dependency structure of Eq. 10 exactly.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .problems import Problem
from .indicators import (
    estimate_ideal_nadir, normalize, hypervolume, riesz_energy_log,
    mean_dispersion, nondominated, geometry_gamma,
)
from .sms_emoa import make_offspring, sms_emoa_eliminate
from .state import StateEncoder
from .rl_planner import ReplicatorQPlanner, BALANCED_REGIME, REGIME_NAMES, ZREF_MAX


DEFAULTS = dict(
    mu=100, pc=1.0, eta_c=15.0, eta_m=20.0,
    H=12, rho=0.7, W=20, eps0=0.1, eps_min=1e-2,
    sigma0=(0.10, 0.10, 0.80), alpha_reward=1.0, lam=0.9,
    alpha_rl=0.1, gamma=0.9, q_reset_every=1000, q_reset_alpha=0.1,
    zref_max=ZREF_MAX, zref_min_eps=1e-3,
    q_update_regimes="all", sigma_floor=0.05, action_every=1,
    geometry_mode="curvature", n_bins_fine=5, n_bins_coarse=3,
    reward_hv="fixed", kappa=0.1, riesz_s=None,
    eval_every=1, record_every=1,
)


# ===========================================================================
# History
# ===========================================================================
class History:
    """One row per generation. ``rows`` is a list of flat dicts."""

    def __init__(self):
        self.rows: list[dict] = []
        self.meta: dict = {}

    def add(self, **kw):
        self.rows.append(kw)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        for k, v in self.meta.items():
            if np.isscalar(v):
                df.attrs[k] = v
        return df

    # convenience accessors used by the older code / plots
    def series(self, key: str) -> np.ndarray:
        return np.array([r.get(key, np.nan) for r in self.rows], dtype=float)

    def __len__(self):
        return len(self.rows)


def _flat_vec(name: str, v) -> dict:
    return {f"{name}{i + 1}": float(x) for i, x in enumerate(np.asarray(v).ravel())}


# ===========================================================================
# shared machinery
# ===========================================================================
def _init_population(problem: Problem, mu: int, rng: np.random.Generator):
    X = rng.random((mu, problem.n_var)) * (problem.xu - problem.xl) + problem.xl
    return X, problem.evaluate(X)


def _sms_emoa_generation(problem, X, F, zref_norm, ideal, nadir, cfg, rng):
    pm = 1.0 / problem.n_var
    child_x = make_offspring(X, problem.xl, problem.xu, cfg["pc"], cfg["eta_c"],
                             pm, cfg["eta_m"], rng)
    child_f = problem.evaluate(child_x.reshape(1, -1))
    X2 = np.vstack([X, child_x])
    F2 = np.vstack([F, child_f])
    return sms_emoa_eliminate(X2, F2, zref_norm, ideal, nadir)


def _fixed_hv_norm(A_norm: np.ndarray, kappa: float) -> float:
    """HV against the FIXED point (1+kappa)*1 in the estimated normalised
    frame, divided by the volume of that box -> in [0,1], comparable across
    generations because the box never moves."""
    m = A_norm.shape[1]
    z = np.full(m, 1.0 + kappa)
    return float(np.clip(hypervolume(A_norm, z) / ((1.0 + kappa) ** m), 0.0, 1.0))


def _record(hist: History, t: int, F_raw, zref_norm, frame, cfg,
            extra: dict | None = None):
    """Log one generation. ``frame`` may be None (then no external indicators)."""
    ideal, nadir = estimate_ideal_nadir(F_raw)
    A = F_raw[nondominated(F_raw)] if F_raw.shape[0] > 1 else F_raw
    A_norm = normalize(A, ideal, nadir)

    row = dict(t=t, n_nd=int(A.shape[0]))
    row.update(_flat_vec("zref", zref_norm))
    row.update(_flat_vec("ideal", ideal))
    row.update(_flat_vec("nadir", nadir))
    row["hv_adaptive"] = hypervolume(A_norm, zref_norm)
    row["hv_fixed_norm"] = _fixed_hv_norm(A_norm, cfg["kappa"])
    row["dispersion"] = mean_dispersion(A_norm)
    row["riesz_log"] = riesz_energy_log(A_norm, s=cfg["riesz_s"] or A_norm.shape[1])
    row["gamma_geom"] = geometry_gamma(A_norm)

    if frame is not None and (t % cfg["eval_every"] == 0 or t == cfg["_t_max"]):
        row.update(frame.evaluate(A))          # hvr, igd_plus, energy_ratio
    else:
        row.update(dict(hvr=np.nan, igd_plus=np.nan, energy_ratio=np.nan))

    if extra:
        row.update(extra)
    hist.add(**row)


# ===========================================================================
# RL-RP-SMS-EMOA (Algorithm 1)
# ===========================================================================
class RLRPSMSEMOA:
    def __init__(self, problem: Problem, t_max: int, seed: int = 0,
                 frame=None, **kwargs):
        self.problem = problem
        self.t_max = t_max
        self.cfg = {**DEFAULTS, **kwargs}
        self.cfg["_t_max"] = t_max
        self.rng = np.random.default_rng(seed)
        self.frame = frame
        self.t_adapt = int(np.ceil(self.cfg["rho"] * t_max))

        self.state_encoder = StateEncoder(
            n_bins_fine=self.cfg["n_bins_fine"],
            n_bins_coarse=self.cfg["n_bins_coarse"],
            geometry_mode=self.cfg["geometry_mode"],
            riesz_s=self.cfg["riesz_s"] or problem.n_obj,
        )
        self.planner = ReplicatorQPlanner(
            m=problem.n_obj, n_states=self.state_encoder.n_states,
            H=self.cfg["H"], W=self.cfg["W"], lam=self.cfg["lam"],
            alpha_rl=self.cfg["alpha_rl"], gamma=self.cfg["gamma"],
            eps0=self.cfg["eps0"], eps_min=self.cfg["eps_min"],
            sigma0=self.cfg["sigma0"], seed=seed,
            q_reset_every=self.cfg["q_reset_every"],
            q_reset_alpha=self.cfg["q_reset_alpha"],
            zref_max=self.cfg["zref_max"], zref_min_eps=self.cfg["zref_min_eps"],
            q_update_regimes=self.cfg["q_update_regimes"],
            sigma_floor=self.cfg["sigma_floor"],
        )
        self.history = History()

    # -- reward -----------------------------------------------------------
    def _reward(self, prev: dict, cur: dict) -> float:
        key = "hv_fixed" if self.cfg["reward_hv"] == "fixed" else "HVnorm"
        d_hv = cur[key] - prev[key]
        d_uni = prev["Enorm"] - cur["Enorm"]      # >0 : became more uniform
        return float(d_hv + self.cfg["alpha_reward"] * d_uni)

    def run(self):
        cfg, mu = self.cfg, self.cfg["mu"]
        X, F = _init_population(self.problem, mu, self.rng)

        # z_ref(0) = nadir_hat + eps -> just inside the box, at the nadir
        zref_norm = np.full(self.problem.n_obj, self.planner.zref_min)

        prev_feats = None       # s_{t-1} features, for payoffs and reward
        prev_regime = None
        pending = None          # (s_idx, k, j, prev_feats) awaiting r_t and target

        for t in range(1, self.t_max + 1):
            ideal, nadir = estimate_ideal_nadir(F)
            F_norm = normalize(F, ideal, nadir)
            adapting = t <= self.t_adapt

            k_t = j_t = -1
            reward = np.nan
            u = np.full(3, np.nan)
            decision = adapting and (t % cfg["action_every"] == 0)

            if decision:
                feats, s_idx = self.state_encoder.compute(F_norm, zref_norm, t, self.t_max)
                feats["hv_fixed"] = _fixed_hv_norm(
                    F_norm[nondominated(F_norm)] if F_norm.shape[0] > 1 else F_norm,
                    cfg["kappa"])

                # --- payoff bookkeeping + replicator dynamics (Eq. 7-9) -----
                if prev_regime is not None and prev_feats is not None:
                    d_uni = prev_feats["Enorm"] - feats["Enorm"]
                    d_hv = feats["HVnorm"] - prev_feats["HVnorm"]
                    self.planner.update_payoff_window(prev_regime, d_uni, d_hv)
                u = self.planner.payoffs(feats["Dnorm"])
                # Eq. 9's floor max(sigma_k, eps0) shares the same linearly
                # decaying eps0 as the eps-greedy Q selection: early on no
                # regime can be starved of exploration; as eps decays the
                # floor relaxes and the dynamics is allowed to consolidate.
                self.planner.replicator_update(u, eps0=self.planner.eps)
                k_t = self.planner.sample_regime()

                # --- finish the deferred update from generation t-1 ----------
                if pending is not None:
                    s_prev, k_prev, j_prev, f_prev = pending
                    reward = self._reward(f_prev, feats)
                    self.planner.update_q(s_prev, k_prev, j_prev, reward, s_idx, k_t)
                    pending = None

                j_t = self.planner.select_subaction(s_idx, k_t)
                zref_norm = self.planner.apply_action(zref_norm, k_t, j_t)

                pending = (s_idx, k_t, j_t, feats)
                prev_feats, prev_regime = feats, k_t
            else:
                feats, s_idx = {}, -1

            extra = dict(regime=k_t, regime_name=REGIME_NAMES[k_t] if k_t >= 0 else "frozen",
                         subaction=j_t, state_idx=s_idx, reward=reward,
                         eps=self.planner.eps, adapting=bool(adapting))
            extra.update(_flat_vec("sigma", self.planner.sigma))
            extra.update(_flat_vec("payoff", u))
            for key in ("Dnorm", "HVnorm", "Enorm", "gamma", "iota", "p_hat",
                        "extent", "e_ratio"):
                extra[f"s_{key}"] = float(feats.get(key, np.nan))

            X, F = _sms_emoa_generation(self.problem, X, F, zref_norm, ideal, nadir,
                                        cfg, self.rng)

            if t % cfg["record_every"] == 0 or t == self.t_max:
                _record(self.history, t, F, zref_norm, self.frame, cfg, extra)

            if decision:
                self.planner.decay_epsilon(t, self.t_adapt)
                self.planner.maybe_soft_reset_q()

        self.history.meta.update(
            n_states=self.state_encoder.n_states,
            t_adapt=self.t_adapt,
            **{f"qcov_{k}": v for k, v in self.planner.q_coverage().items()})
        return X, F, self.history


# ===========================================================================
# Baselines
# ===========================================================================
def _run_baseline(problem: Problem, t_max: int, mode: str, seed: int = 0,
                  frame=None, **kwargs):
    cfg = {**DEFAULTS, **kwargs}
    cfg["_t_max"] = t_max
    rng = np.random.default_rng(seed)
    X, F = _init_population(problem, cfg["mu"], rng)
    H = cfg["H"]
    hist = History()

    for t in range(1, t_max + 1):
        ideal, nadir = estimate_ideal_nadir(F)

        if mode == "nadir":
            zref_norm = np.ones(problem.n_obj) + 0.01
        elif mode == "balanced":
            zref_norm = np.ones(problem.n_obj) + 1.0 / H
        elif mode == "dynlin":
            zref_norm = np.ones(problem.n_obj) + (0.01 + (t / t_max) * (1.0 - 0.01))
        else:
            raise ValueError(mode)

        X, F = _sms_emoa_generation(problem, X, F, zref_norm, ideal, nadir, cfg, rng)

        if t % cfg["record_every"] == 0 or t == t_max:
            _record(hist, t, F, zref_norm, frame, cfg,
                    dict(regime=-1, regime_name=mode, subaction=-1, state_idx=-1,
                         reward=np.nan, eps=np.nan, adapting=False))

    return X, F, hist


def run_sms_emoa_nadir(problem, t_max, seed=0, **kw):
    return _run_baseline(problem, t_max, "nadir", seed, **kw)


def run_sms_emoa_balanced(problem, t_max, seed=0, **kw):
    return _run_baseline(problem, t_max, "balanced", seed, **kw)


def run_sms_emoa_dynlin(problem, t_max, seed=0, **kw):
    return _run_baseline(problem, t_max, "dynlin", seed, **kw)


def run_rl_rp_sms_emoa(problem, t_max, seed=0, **kw):
    return RLRPSMSEMOA(problem, t_max, seed=seed, **kw).run()


#: ``SMS-EMOA_nadir`` (constant +0.01 offset) has been REPLACED by the two
#: baselines in r2_emoa.py -- it is not what Beume et al. proposed and it was
#: a straw man.  It stays importable under ``LEGACY_METHODS`` so that older
#: result files can still be reproduced.
METHODS = {
    "SMS-EMOA_nadir-adaptive": None,   # filled in below (avoids a circular import)
    "SMS-EMOA_balanced": run_sms_emoa_balanced,
    "SMS-EMOA_dynlin": run_sms_emoa_dynlin,
    "R2-EMOA": None,                   # filled in below
    "RL-RP-SMS-EMOA": run_rl_rp_sms_emoa,
}

LEGACY_METHODS = {"SMS-EMOA_nadir": run_sms_emoa_nadir}


def _register_indicator_baselines():
    from .r2_emoa import run_r2_emoa, run_sms_emoa_nadir_adaptive
    METHODS["SMS-EMOA_nadir-adaptive"] = run_sms_emoa_nadir_adaptive
    METHODS["R2-EMOA"] = run_r2_emoa


_register_indicator_baselines()
