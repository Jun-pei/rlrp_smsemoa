"""
experiment.py
=============
Glue for the comparison protocol: n_methods x n_seeds runs on a problem, all
sharing one ``ReferenceFrame`` so every number is on the same scale, with the
COMPLETE per-generation history of each run written to disk.

Writing the history is not optional: ``run_single`` needs a ``histdir`` and
always saves there.  The summary row it returns is a convenience for the
statistics; the file on disk is the actual result.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .algorithm import METHODS
from .history_io import load_history, save_history
from .performance import anytime_score, time_to_target
from .problems import get_problem, make_reference_frame, problem_n_obj

_FRAME_CACHE: dict = {}


def get_frame(problem_name: str, m: int = 3, n_points: int = 5000, kappa: float = 0.1):
    """Cached per-problem reference frame (building Z is not free)."""
    key = (problem_name.lower(), m, n_points, kappa)
    if key not in _FRAME_CACHE:
        _FRAME_CACHE[key] = make_reference_frame(problem_name, m=m,
                                                 n_points=n_points, kappa=kappa)
    return _FRAME_CACHE[key]


def run_single(problem_name: str, method_name: str, seed: int, t_max: int,
               histdir: str, m: int = 3, mu: int = 100, frame=None,
               keep_history: bool = False, **kwargs) -> dict:
    """One (problem, method, seed) run.

    The full per-generation history goes to
    ``<histdir>/<problem>/<method>/seed<NNN>.csv.gz``; the returned dict is the
    summary row used by the statistics.
    """
    m_eff = problem_n_obj(problem_name, m)
    problem = get_problem(problem_name, m=m_eff)
    frame = frame if frame is not None else get_frame(problem_name, m=m_eff)

    X, F, hist = METHODS[method_name](problem, t_max=t_max, seed=seed, mu=mu,
                                      frame=frame, **kwargs)
    df = hist.to_dataframe()
    hist_path = save_history(df, histdir, problem_name, method_name, seed,
                             meta=dict(problem=problem_name, method=method_name,
                                       seed=seed, m=m_eff, mu=mu, t_max=t_max,
                                       n_ref=int(frame.Zhat.shape[0]), **hist.meta))

    final = frame.evaluate(F)
    hvr_curve = df["hvr"].to_numpy() if "hvr" in df else np.array([])
    row = dict(
        problem=problem_name, method=method_name, seed=seed, m=m_eff, mu=mu,
        t_max=t_max, n_final=int(F.shape[0]),
        # --- primary indicators, Definitions D1-D3 --------------------------
        hvr_final=final["hvr"], igd_plus_final=final["igd_plus"],
        energy_ratio_final=final["energy_ratio"],
        # --- anytime, Definition D4 -----------------------------------------
        hvr_anytime=anytime_score(hvr_curve),
        # --- diagnostics ------------------------------------------------------
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


def add_time_to_target(df: pd.DataFrame, q: float = 0.95) -> pd.DataFrame:
    """Definition D5.  Needs HVR_max across methods on the same (problem, seed),
    so it is a post-processing step over the already-written histories."""
    out = []
    for _, grp in df.groupby(["problem", "seed"]):
        curves = {}
        for _, r in grp.iterrows():
            path = r.get("history_path")
            if isinstance(path, str) and os.path.exists(path):
                curves[r["method"]] = load_history(path)["hvr"].to_numpy()
        hvr_max = max((np.nanmax(c) for c in curves.values() if np.isfinite(c).any()),
                      default=np.nan)
        for _, r in grp.iterrows():
            r = r.copy()
            c = curves.get(r["method"])
            r["time_to_target"] = (time_to_target(c, hvr_max, q)
                                   if c is not None else np.nan)
            out.append(r)
    return pd.DataFrame(out)


def run_comparison(problem_name: str, histdir: str, n_seeds: int = 30,
                   t_max: int = 2000, m: int = 3, mu: int = 100, methods=None,
                   ref_points: int = 5000, **kwargs) -> pd.DataFrame:
    """All methods x all seeds on one problem, in-process."""
    methods = methods or list(METHODS)
    m_eff = problem_n_obj(problem_name, m)
    frame = get_frame(problem_name, m=m_eff, n_points=ref_points)
    rows = [run_single(problem_name, meth, seed, t_max, histdir, m=m_eff, mu=mu,
                       frame=frame, **kwargs)
            for meth in methods for seed in range(n_seeds)]
    return pd.DataFrame(rows)
