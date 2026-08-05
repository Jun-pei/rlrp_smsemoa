"""
algorithm.py
============
RL-RP-SMS-EMOA (Algorithm 1 of the proposal) and the reference-point baselines
it is compared against.

The four compared configurations
--------------------------------
  ``SMS-EMOA_balanced``  z_ref = (1 + 1/H) * 1, constant.
        H. Ishibuchi, R. Imada, Y. Setoguchi, Y. Nojima, "Reference point
        specification in hypervolume calculation for fair comparison and
        efficient search", GECCO 2017, pp. 585-592.
        1/H is the spacing of a Das-Dennis lattice with H divisions and gives
        equal HV contributions to uniformly distributed solutions on a linear
        front, which is why it is the standard fixed choice.

  ``d-SMS-EMOA``         z_ref = r(t) * 1, r decreasing from 10 to 1.
        H. Ishibuchi, R. Imada, N. Masuyama, Y. Nojima, "Dynamic specification
        of a reference point for hypervolume calculation in SMS-EMOA",
        CEC 2018, pp. 701-708.  Also written SMS-EMOA-DRP.
        See ``run_d_sms_emoa``.

  ``R2-EMOA``            indicator-agnostic control, see r2_emoa.py.

  ``RL-RP-SMS-EMOA``     the proposed planner, ``RLRPSMSEMOA`` below.

All of them share the machinery in core.py (same SBX/PM operators, same
steady-state elimination, same normalisation, same logging), so a difference
between them is a difference in the reference-point / selection mechanism.

Two deviations from the pseudocode, both deliberate
---------------------------------------------------
1. *Deferred Q-update.*  Eq. (10) bootstraps Q[s_t,(k_t,j_t)] against
   Q[s_{t+1},(k_{t+1}, .)], so it needs the regime the replicator dynamics will
   pick at t+1 before the update for t can be finished.  ``(s_t, k_t, j_t)`` is
   therefore stashed and the TD update completed at the start of generation
   t+1, once (s_{t+1}, k_{t+1}) and hence r_t are known.  This preserves the
   dependency structure of Eq. (10) exactly.

2. *``action_every`` (tau).*  The planner may act once every tau generations,
   with the reward measured over the whole window.  In a steady-state EMOA one
   generation creates one offspring and deletes one individual, so a one-step
   reward is dominated by the noise of the variation operators, while moving
   z_ref only takes effect over a full population turnover (~mu generations).
   Measured on DTLZ2, the Spearman correlation between r_t and the subsequent
   change in HVR is +0.008 -- the per-generation reward carries no usable
   signal.  tau = 1 reproduces the pseudocode literally; tau ~ mu aligns the
   credit-assignment horizon with the action's actual effect.
"""

from __future__ import annotations
import numpy as np

from .core import (
    DEFAULTS, History, baseline_extra, fixed_hv_norm, flat_vec,
    init_population, record, sms_emoa_generation,
)
from .indicators import estimate_ideal_nadir, normalize, nondominated
from .problems import Problem
from .r2_emoa import run_r2_emoa
from .rl_planner import ReplicatorQPlanner, REGIME_NAMES
from .state import StateEncoder

#: d-SMS-EMOA reference-point schedule endpoints, in the normalised objective
#: space (Ishibuchi et al., CEC 2018)
DRP_R_START = 10.0
DRP_R_END = 1.0


# ===========================================================================
# RL-RP-SMS-EMOA (Algorithm 1)
# ===========================================================================
class RLRPSMSEMOA:
    """SMS-EMOA whose reference point is moved by a replicator-dynamics +
    Q-learning planner during the first ``rho * t_max`` generations, then held
    fixed for the refinement phase."""

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

    def _reward(self, prev: dict, cur: dict) -> float:
        """r_t = Delta HV + alpha * Delta(uniformity).

        ``reward_hv="fixed"`` (default) measures the HV term against the fixed
        box (1+kappa)*1; ``"adaptive"`` measures it against the moving z_ref,
        as originally written.  Both are kept so the two can be compared.
        """
        key = "hv_fixed" if self.cfg["reward_hv"] == "fixed" else "HVnorm"
        d_hv = cur[key] - prev[key]
        d_uni = prev["Enorm"] - cur["Enorm"]      # > 0 : became more uniform
        return float(d_hv + self.cfg["alpha_reward"] * d_uni)

    def run(self):
        cfg, mu = self.cfg, self.cfg["mu"]
        X, F = init_population(self.problem, mu, self.rng)

        # z_ref(0): just inside the box, at the estimated nadir
        zref_norm = np.full(self.problem.n_obj, self.planner.zref_min)

        prev_feats = None       # features at the previous decision
        prev_regime = None
        pending = None          # (s, k, j, feats) awaiting r_t and its target

        for t in range(1, self.t_max + 1):
            ideal, nadir = estimate_ideal_nadir(F)
            F_norm = normalize(F, ideal, nadir)
            adapting = t <= self.t_adapt

            k_t = j_t = -1
            reward = np.nan
            u = np.full(3, np.nan)
            decision = adapting and (t % cfg["action_every"] == 0)

            if decision:
                feats, s_idx = self.state_encoder.compute(F_norm, zref_norm, t,
                                                          self.t_max)
                feats["hv_fixed"] = fixed_hv_norm(
                    F_norm[nondominated(F_norm)] if F_norm.shape[0] > 1 else F_norm,
                    cfg["kappa"])

                # --- payoffs + replicator dynamics (Eq. 7-9) -----------------
                if prev_regime is not None and prev_feats is not None:
                    d_uni = prev_feats["Enorm"] - feats["Enorm"]
                    d_hv = feats["HVnorm"] - prev_feats["HVnorm"]
                    self.planner.update_payoff_window(prev_regime, d_uni, d_hv)
                u = self.planner.payoffs(feats["Dnorm"])
                # Eq. 9's floor max(sigma_k, eps0) shares the same linearly
                # decaying eps0 as the eps-greedy Q selection: early on no
                # regime can be starved of exploration; as eps decays the floor
                # relaxes and the dynamics is allowed to consolidate.
                self.planner.replicator_update(u, eps0=self.planner.eps)
                k_t = self.planner.sample_regime()

                # --- finish the deferred update from the previous decision ---
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

            extra = dict(regime=k_t,
                         regime_name=REGIME_NAMES[k_t] if k_t >= 0 else "frozen",
                         subaction=j_t, state_idx=s_idx, reward=reward,
                         eps=self.planner.eps, adapting=bool(adapting))
            extra.update(flat_vec("sigma", self.planner.sigma))
            extra.update(flat_vec("payoff", u))
            for key in ("Dnorm", "HVnorm", "Enorm", "gamma", "iota", "p_hat",
                        "extent", "e_ratio"):
                extra[f"s_{key}"] = float(feats.get(key, np.nan))

            X, F = sms_emoa_generation(self.problem, X, F, zref_norm, ideal,
                                       nadir, cfg, self.rng)
            record(self.history, t, F, zref_norm, self.frame, cfg, extra)

            if decision:
                self.planner.decay_epsilon(t, self.t_adapt)
                self.planner.maybe_soft_reset_q()

        self.history.meta.update(
            n_states=self.state_encoder.n_states,
            t_adapt=self.t_adapt,
            **{f"qcov_{k}": v for k, v in self.planner.q_coverage().items()})
        return X, F, self.history


def run_rl_rp_sms_emoa(problem, t_max, seed=0, **kw):
    return RLRPSMSEMOA(problem, t_max, seed=seed, **kw).run()


# ===========================================================================
# Reference-point baselines
# ===========================================================================
def run_sms_emoa_balanced(problem, t_max, seed=0, frame=None, **kwargs):
    """Constant z_ref = (1 + 1/H) * 1 in the normalised objective space
    (Ishibuchi et al., GECCO 2017)."""
    cfg = {**DEFAULTS, **kwargs}
    cfg["_t_max"] = t_max
    rng = np.random.default_rng(seed)
    X, F = init_population(problem, cfg["mu"], rng)
    zref_norm = np.ones(problem.n_obj) + 1.0 / cfg["H"]
    hist = History()
    hist.meta["zref"] = float(1.0 + 1.0 / cfg["H"])

    for t in range(1, t_max + 1):
        ideal, nadir = estimate_ideal_nadir(F)
        X, F = sms_emoa_generation(problem, X, F, zref_norm, ideal, nadir, cfg, rng)
        record(hist, t, F, zref_norm, frame, cfg, baseline_extra("balanced"))
    return X, F, hist


def run_d_sms_emoa(problem, t_max, seed=0, frame=None,
                   r_start=DRP_R_START, r_end=DRP_R_END, **kwargs):
    """d-SMS-EMOA / SMS-EMOA-DRP -- dynamic reference-point specification
    (Ishibuchi, Imada, Masuyama & Nojima, CEC 2018).

    The reference point z_ref = (r, ..., r) in the NORMALISED objective space
    is moved from r = 10 for the initial population down to r = 1 for the final
    one:

        r(t) = r_start + (t - 1) / (T_max - 1) * (r_end - r_start)

    A large r makes the extreme solutions carry most of the hypervolume, so the
    search first spreads towards the *boundary* of the front; a small r makes
    the interior solutions dominate the contribution, so the search then
    concentrates on the *centre*.  Sweeping r from 10 to 1 therefore visits
    both regimes in the order that matters for a non-triangular front, which is
    the whole point of the method: on inverted fronts (Minus-DTLZ) the
    reference-point specification, not the algorithm, decides the shape of the
    final solution set.

    This is the schedule the paper describes (r = 10 at the initial population,
    r = 1 at the final one).  Interpolating *linearly in the generation index*
    is this implementation's reading of "changed from ... to ..."; ``r_start``
    and ``r_end`` are exposed so the schedule can be varied.

    Note that r_end = 1 puts the reference point exactly on the estimated
    nadir, where a solution attaining the nadir in some objective contributes
    zero hypervolume.  That is the paper's specification and it is kept
    faithfully; ``sms_emoa.sms_emoa_eliminate`` has an explicit secondary
    criterion for the resulting all-zero-contribution case.
    """
    cfg = {**DEFAULTS, **kwargs}
    cfg["_t_max"] = t_max
    rng = np.random.default_rng(seed)
    X, F = init_population(problem, cfg["mu"], rng)
    hist = History()
    hist.meta.update(r_start=float(r_start), r_end=float(r_end))

    denom = max(t_max - 1, 1)
    for t in range(1, t_max + 1):
        r = r_start + (t - 1) / denom * (r_end - r_start)
        zref_norm = np.full(problem.n_obj, r)
        ideal, nadir = estimate_ideal_nadir(F)
        X, F = sms_emoa_generation(problem, X, F, zref_norm, ideal, nadir, cfg, rng)
        record(hist, t, F, zref_norm, frame, cfg, baseline_extra("drp"))
    return X, F, hist


#: the four configurations compared in Sec. 5.3
METHODS = {
    "SMS-EMOA_balanced": run_sms_emoa_balanced,
    "d-SMS-EMOA": run_d_sms_emoa,
    "R2-EMOA": run_r2_emoa,
    "RL-RP-SMS-EMOA": run_rl_rp_sms_emoa,
}
