"""
experiment.py
=============
Glue code for the comparison protocol: n_methods x n_seeds runs on a problem,
with a single shared ``ReferenceFrame`` so that every number produced is on
the same scale, plus per-generation histories written to disk.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .problems import get_problem, make_reference_frame, problem_n_obj, PROBLEM_NAMES
from .algorithm import METHODS
from .performance import anytime_score, time_to_target
from .history_io import save_history

_FRAME_CACHE: dict = {}


def get_frame(problem_name: str, m: int = 3, n_points: int = 5000, kappa: float = 0.1):
    """Cached per-problem reference frame (building Z is not free)."""
    key = (problem_name.lower(), m, n_points, kappa)
    if key not in _FRAME_CACHE:
        _FRAME_CACHE[key] = make_reference_frame(problem_name, m=m,
                                                 n_points=n_points, kappa=kappa)
    return _FRAME_CACHE[key]


def run_single(problem_name: str, method_name: str, seed: int, t_max: int,
               m: int = 3, mu: int = 100, frame=None, histdir: str | None = None,
               keep_history: bool = False, **kwargs) -> dict:
    """One (problem, method, seed) run.

    Returns the summary row.  If ``histdir`` is given, the full per-generation
    history is written there as a compressed CSV.
    """
    m_eff = problem_n_obj(problem_name, m)
    problem = get_problem(problem_name, m=m_eff)
    frame = frame if frame is not None else get_frame(problem_name, m=m_eff)

    X, F, hist = METHODS[method_name](problem, t_max=t_max, seed=seed, mu=mu,
                                      frame=frame, **kwargs)
    df = hist.to_dataframe()

    hist_path = None
    if histdir:
        hist_path = save_history(df, histdir, problem_name, method_name, seed,
                                 meta=dict(problem=problem_name, method=method_name,
                                           seed=seed, m=m_eff, mu=mu, t_max=t_max,
                                           **hist.meta))

    final = frame.evaluate(F)
    hvr_curve = df["hvr"].to_numpy() if "hvr" in df else np.array([])
    row = dict(
        problem=problem_name, method=method_name, seed=seed, m=m_eff, mu=mu,
        t_max=t_max, n_final=int(F.shape[0]),
        # --- primary indicators, Definitions D1-D3 -----------------------
        hvr_final=final["hvr"], igd_plus_final=final["igd_plus"],
        energy_ratio_final=final["energy_ratio"],
        # --- anytime, Definition D4 ---------------------------------------
        hvr_anytime=anytime_score(hvr_curve),
        # --- diagnostics ---------------------------------------------------
        hv_adaptive_final=float(df["hv_adaptive"].iloc[-1]) if len(df) else np.nan,
        gamma_final=float(df["gamma_geom"].iloc[-1]) if len(df) else np.nan,
        zref_mean_final=float(np.mean([df[c].iloc[-1] for c in df.columns
                                       if c.startswith("zref")])) if len(df) else np.nan,
        n_generations=int(len(df)), history_path=hist_path,
    )
    row.update({k: v for k, v in hist.meta.items() if k.startswith("qcov_")})
    if keep_history:
        row["_history"] = df
        row["_X"], row["_F"] = X, F
    return row


def add_time_to_target(df: pd.DataFrame, histdir: str, q: float = 0.95) -> pd.DataFrame:
    """Definition D5: needs HVR_max across methods, so it is a post-processing
    step over the already-written histories."""
    from .history_io import load_history
    out = []
    for (prob, seed), grp in df.groupby(["problem", "seed"]):
        curves = {}
        for _, r in grp.iterrows():
            if isinstance(r.get("history_path"), str) and os.path.exists(r["history_path"]):
                curves[r["method"]] = load_history(r["history_path"])["hvr"].to_numpy()
        hvr_max = max((np.nanmax(c) for c in curves.values() if np.isfinite(c).any()),
                      default=np.nan)
        for _, r in grp.iterrows():
            r = r.copy()
            c = curves.get(r["method"])
            r["time_to_target"] = (time_to_target(c, hvr_max, q)
                                   if c is not None else np.nan)
            out.append(r)
    return pd.DataFrame(out)


def pivot_indicator(df: pd.DataFrame, indicator: str, methods=None):
    """[n_seeds, n_methods] matrix of an indicator, ready for stats.py."""
    methods = methods or sorted(df["method"].unique())
    piv = df.pivot(index="seed", columns="method", values=indicator)[methods]
    return piv.values, methods


def run_comparison(problem_name: str, n_seeds: int = 30, t_max: int = 2000,
                   m: int = 3, mu: int = 100, methods=None,
                   ref_points: int = 5000, histdir: str | None = None,
                   **kwargs) -> pd.DataFrame:
    methods = methods or list(METHODS)
    m_eff = problem_n_obj(problem_name, m)
    frame = get_frame(problem_name, m=m_eff, n_points=ref_points)
    rows = [run_single(problem_name, meth, seed, t_max, m=m_eff, mu=mu,
                       frame=frame, histdir=histdir, **kwargs)
            for meth in methods for seed in range(n_seeds)]
    return pd.DataFrame(rows)
