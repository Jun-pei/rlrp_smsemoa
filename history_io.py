"""
history_io.py
=============
Persistence of the PER-GENERATION history (review comment: *guardar todos los
valores de todas las generaciones*).

One file per (problem, method, seed):

    <histdir>/<problem>/<method>/seed<NNN>.csv.gz

with one row per recorded generation and, at minimum, these columns:

    t, n_nd
    zref1..zrefm, ideal1..idealm, nadir1..nadirm
    hv_adaptive          HV against the algorithm's own z_ref (diagnostic only)
    hv_fixed_norm        HV against the fixed (1+kappa)*1 in the estimated frame
    dispersion, riesz_log, gamma_geom
    hvr, igd_plus, energy_ratio        (Definitions D1-D3; NaN when eval_every>1)
    regime, regime_name, subaction, state_idx, reward, eps, adapting
    sigma1..sigma3, payoff1..payoff3
    s_Dnorm, s_HVnorm, s_Enorm, s_gamma, s_iota, s_p_hat, s_extent, s_e_ratio

Run-level metadata (|S|, t_adapt, Q-table coverage, mu, T_max, ...) is written
next to it as ``seed<NNN>.meta.json``.

SIZE
----
About 60 float columns per row.  Gzipped CSV lands around 25-40 bytes/row, so

    T_max = 100,000, record_every = 1  ->  ~3-4 MB per run
    8 problems x 4 methods x 30 seeds  ->  ~3-4 GB

That is affordable, but ``eval_every`` (how often HVR/IGD+ are computed) is
what actually decides the RUNTIME, not the disk cost -- computing IGD+ against
a 5,000-point reference set 100,000 times per run is far more expensive than
the generation itself.  ``record_every`` thins the log; ``eval_every`` thins
only the expensive indicators while still logging every generation.
"""

from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd


def history_path(histdir: str, problem: str, method: str, seed: int) -> str:
    safe = method.replace("/", "-")
    return os.path.join(histdir, problem, safe, f"seed{seed:03d}.csv.gz")


def save_history(df: pd.DataFrame, histdir: str, problem: str, method: str,
                 seed: int, meta: dict | None = None) -> str:
    path = history_path(histdir, problem, method, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, compression="gzip", float_format="%.8g")
    if meta:
        with open(path.replace(".csv.gz", ".meta.json"), "w") as fh:
            json.dump({k: _jsonable(v) for k, v in meta.items()}, fh, indent=2)
    return path


def load_history(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_meta(path: str) -> dict:
    mpath = path.replace(".csv.gz", ".meta.json")
    if not os.path.exists(mpath):
        return {}
    with open(mpath) as fh:
        return json.load(fh)


def iter_histories(histdir: str, problem: str | None = None,
                   method: str | None = None):
    """Yield ``(problem, method, seed, DataFrame)`` for everything on disk."""
    for root, _, files in os.walk(histdir):
        for fn in sorted(files):
            if not fn.endswith(".csv.gz"):
                continue
            path = os.path.join(root, fn)
            parts = os.path.relpath(path, histdir).split(os.sep)
            if len(parts) < 3:
                continue
            p, meth, seed = parts[0], parts[1], int(parts[2][4:7])
            if problem and p != problem:
                continue
            if method and meth != method.replace("/", "-"):
                continue
            yield p, meth, seed, load_history(path)


def stack_curves(histdir: str, column: str, problem: str) -> pd.DataFrame:
    """Long-format table (method, seed, t, value) for one indicator.

    Convenient for the convergence plots of Section 5: with all generations on
    disk, the anytime behaviour can be re-analysed without re-running anything.
    """
    rows = []
    for p, meth, seed, df in iter_histories(histdir, problem=problem):
        if column not in df:
            continue
        sub = df[["t", column]].dropna()
        rows.append(pd.DataFrame(dict(method=meth, seed=seed, t=sub["t"],
                                      value=sub[column])))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
