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
from .history_io import load_history, load_meta, save_history
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


def _last_finite(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return np.nan
    v = df[col].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(v[-1]) if v.size else np.nan


#: the only columns a summary row needs out of a ~40-column history
_SUMMARY_COLS = ("t", "hvr", "igd_plus", "energy_ratio", "hv_adaptive",
                 "gamma_geom")


def _summary_col(name: str) -> bool:
    return name in _SUMMARY_COLS or name.startswith("zref")


def load_summary_frame(path: str) -> pd.DataFrame:
    """Read only what a summary row needs.  Reading all 40 columns of a
    100,000-row history costs about twice as much, and over a full grid of
    1680 runs that is the difference between minutes and an hour."""
    return load_history(path, usecols=_summary_col)


def history_is_complete(path: str, df: pd.DataFrame | None = None) -> bool:
    """Whether a stored history is readable AND covers its whole run.

    A job killed mid-write leaves a truncated or empty ``.csv.gz`` behind, so
    the existence of the file is not evidence that the run finished.  Treating
    it as evidence is exactly what turned one interrupted run into a crash for
    a whole array task.  The history is written in a single call once the run
    ends, so a good file always reaches generation ``t_max``.

    Pass ``df`` to validate a frame that has already been read, instead of
    reading the file a second time.
    """
    if df is None:
        if not os.path.exists(path):
            return False
        try:
            df = load_summary_frame(path)
        except Exception:                 # empty, truncated, bad gzip
            return False
    if df.empty or "t" not in df.columns:
        return False
    t_max = load_meta(path).get("t_max")
    return t_max is None or int(df["t"].iloc[-1]) == int(t_max)


def summary_row_from_history(hist_path: str, df: pd.DataFrame | None = None) -> dict:
    """Rebuild a finished run's summary row from its stored history.

    Nothing has to be recomputed, because every generation was logged: the
    final-population indicators are the last recorded hvr / igd_plus /
    energy_ratio, and the anytime score (D4) is the mean of the hvr column.
    With ``eval_every=1`` these are identical to what ``run_single`` returned,
    since the last row is recorded after the last generation, on the same
    population.

    This is what makes an interrupted batch job recoverable: histories are
    written as each run finishes, whereas ``results_<problem>.csv`` is only
    written once a whole task completes, so a job that hits its wall clock
    keeps all of its data but none of its summaries.
    """
    if df is None:
        df = load_summary_frame(hist_path)
    meta = load_meta(hist_path)

    parts = os.path.normpath(hist_path).split(os.sep)
    row = dict(
        problem=meta.get("problem", parts[-3] if len(parts) >= 3 else None),
        method=meta.get("method", parts[-2] if len(parts) >= 2 else None),
        seed=meta.get("seed", int(parts[-1][4:7]) if parts[-1][:4] == "seed" else -1),
        m=meta.get("m"), mu=meta.get("mu"), t_max=meta.get("t_max"),
        n_final=meta.get("mu"),
        hvr_final=_last_finite(df, "hvr"),
        igd_plus_final=_last_finite(df, "igd_plus"),
        energy_ratio_final=_last_finite(df, "energy_ratio"),
        hvr_anytime=anytime_score(df["hvr"].to_numpy() if "hvr" in df else []),
        hv_adaptive_final=_last_finite(df, "hv_adaptive"),
        gamma_final=_last_finite(df, "gamma_geom"),
        zref_mean_final=float(np.mean([df[c].iloc[-1] for c in df.columns
                                       if c.startswith("zref")])) if len(df) else np.nan,
        n_generations=int(len(df)), history_path=hist_path,
    )
    row.update({k: v for k, v in meta.items() if k.startswith("qcov_")})
    return row


def add_time_to_target(df: pd.DataFrame, q: float = 0.95,
                       hvr_curves: dict | None = None) -> pd.DataFrame:
    """Definition D5.  Needs HVR_max across methods on the same (problem, seed),
    so it is a post-processing step over the already-written histories.

    ``hvr_curves`` maps ``(problem, method, seed) -> hvr array`` for callers
    that have already read the histories; without it each history is read
    again, which over a full grid doubles the cost of aggregating.
    """
    out = []
    for _, grp in df.groupby(["problem", "seed"]):
        curves = {}
        for _, r in grp.iterrows():
            key = (r["problem"], r["method"], r["seed"])
            if hvr_curves is not None and key in hvr_curves:
                curves[r["method"]] = hvr_curves[key]
                continue
            path = r.get("history_path")
            if isinstance(path, str) and os.path.exists(path):
                curves[r["method"]] = load_history(path, usecols=["hvr"])["hvr"].to_numpy()
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
