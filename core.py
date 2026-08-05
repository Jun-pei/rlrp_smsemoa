"""
core.py
=======
Machinery shared by every algorithm in this package: the default
configuration, the per-generation ``History``, one steady-state SMS-EMOA
generation, and the logging routine.

Keeping it here is what lets ``algorithm.py`` (RL-RP-SMS-EMOA and the
reference-point baselines) and ``r2_emoa.py`` (the indicator-agnostic control)
share exactly the same variation operators, elimination step, normalisation and
instrumentation, so that any difference between methods is attributable to the
selection / reference-point mechanism alone.

EVERY GENERATION IS LOGGED
--------------------------
``_record`` is called once per generation, with no thinning: the history is a
complete trace of the run, not a final snapshot.  ``eval_every`` thins only the
*expensive external* indicators (HVR / IGD+ / Eratio against the reference set
Z); the cheap per-generation quantities -- z_ref, ideal/nadir, HV, dispersion,
Riesz energy, curvature, regime, sub-action, reward, sigma, payoffs and the
state features -- are always written for every t.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .indicators import (
    estimate_ideal_nadir, normalize, hypervolume, riesz_energy_log,
    mean_dispersion, nondominated, geometry_gamma,
)
from .problems import Problem
from .rl_planner import ZREF_MAX
from .sms_emoa import make_offspring, sms_emoa_eliminate


DEFAULTS = dict(
    # --- SMS-EMOA base (proposal Sec. 2.2) --------------------------------
    mu=100, pc=1.0, eta_c=15.0, eta_m=20.0,
    # --- planner ------------------------------------------------------------
    H=12, rho=0.7, W=20, eps0=0.1, eps_min=1e-2,
    sigma0=(0.10, 0.10, 0.80), alpha_reward=1.0, lam=0.9,
    alpha_rl=0.1, gamma=0.9, q_reset_every=1000, q_reset_alpha=0.1,
    zref_max=ZREF_MAX, zref_min_eps=1e-3,
    q_update_regimes="all", sigma_floor=0.05, action_every=1,
    # --- state encoding -----------------------------------------------------
    geometry_mode="curvature", n_bins_fine=5, n_bins_coarse=3,
    # --- reward / indicators ------------------------------------------------
    reward_hv="fixed", kappa=0.1, riesz_s=None, eval_every=1,
)


class History:
    """Complete per-generation trace of a run: one flat dict per generation."""

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

    def series(self, key: str) -> np.ndarray:
        return np.array([r.get(key, np.nan) for r in self.rows], dtype=float)

    def __len__(self):
        return len(self.rows)


def flat_vec(name: str, v) -> dict:
    """``{'zref1': .., 'zref2': ..}`` -- vectors are stored one column each."""
    return {f"{name}{i + 1}": float(x) for i, x in enumerate(np.asarray(v).ravel())}


def init_population(problem: Problem, mu: int, rng: np.random.Generator):
    X = rng.random((mu, problem.n_var)) * (problem.xu - problem.xl) + problem.xl
    return X, problem.evaluate(X)


def sms_emoa_generation(problem, X, F, zref_norm, ideal, nadir, cfg, rng):
    """One steady-state generation: one offspring in, one individual out."""
    pm = 1.0 / problem.n_var
    child_x = make_offspring(X, problem.xl, problem.xu, cfg["pc"], cfg["eta_c"],
                             pm, cfg["eta_m"], rng)
    child_f = problem.evaluate(child_x.reshape(1, -1))
    return sms_emoa_eliminate(np.vstack([X, child_x]), np.vstack([F, child_f]),
                              zref_norm, ideal, nadir)


def fixed_hv_norm(A_norm: np.ndarray, kappa: float) -> float:
    """HV against the FIXED point (1+kappa)*1 in the estimated normalised frame,
    divided by the volume of that box, hence in [0,1].

    Unlike HV against the moving z_ref, this is comparable across generations:
    HV(A, z_ref)/vol(z_ref) changes when z_ref moves even if the population is
    untouched, so a reward built on it would partly pay the planner for moving
    the goalposts rather than for improving the front.
    """
    m = A_norm.shape[1]
    z = np.full(m, 1.0 + kappa)
    return float(np.clip(hypervolume(A_norm, z) / ((1.0 + kappa) ** m), 0.0, 1.0))


def record(hist: History, t: int, F_raw, zref_norm, frame, cfg,
           extra: dict | None = None):
    """Log generation ``t``.  ``frame`` may be None (then no external indicators)."""
    ideal, nadir = estimate_ideal_nadir(F_raw)
    A = F_raw[nondominated(F_raw)] if F_raw.shape[0] > 1 else F_raw
    A_norm = normalize(A, ideal, nadir)

    row = dict(t=t, n_nd=int(A.shape[0]))
    row.update(flat_vec("zref", zref_norm))
    row.update(flat_vec("ideal", ideal))
    row.update(flat_vec("nadir", nadir))
    row["hv_adaptive"] = hypervolume(A_norm, zref_norm)
    row["hv_fixed_norm"] = fixed_hv_norm(A_norm, cfg["kappa"])
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


def baseline_extra(name: str) -> dict:
    """Planner columns for methods that have no planner (kept so that every
    method writes the same schema)."""
    return dict(regime=-1, regime_name=name, subaction=-1, state_idx=-1,
                reward=np.nan, eps=np.nan, adapting=False)
