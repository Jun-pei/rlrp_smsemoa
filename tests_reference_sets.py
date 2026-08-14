"""
tests_reference_sets.py
=======================
Validation of every reference set Z used for IGD+ (Definition D2 / Section 2
of performance.py), over the whole benchmark: the six DTLZ/WFG problems and
all eight IMOP problems.  Run from inside the repository with::

    python3 tests_reference_sets.py

(or, from the repository's parent directory,
``python3 -m rlrp_smsemoa.tests_reference_sets``).

Two independent checks per problem:

  (A) SOUNDNESS  -- no point of Z is dominated by a large random sample of
      attainable objective vectors.  If Z sat above the true front (the old
      random-sampling fallback), random search would dominate parts of it.

  (B) TIGHTNESS  -- the non-dominated subset of a dense sample of the g = 0
      manifold (or of the whole decision space, for the Minus problems) lies
      close to Z.  If Z sat below the true front, this distance would blow up.

Both are reported rather than asserted with a hard threshold, because the
attainable tolerance differs by problem (IMOP8 is extremely multimodal).
"""

from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import cKDTree

from rlrp_smsemoa.imop import _nd_filter
from rlrp_smsemoa.problems import (
    PROBLEM_NAMES, get_problem, problem_n_obj, reference_set,
)


def _dominated_by(sample: np.ndarray, Z: np.ndarray, block: int = 512) -> int:
    """How many rows of Z are dominated by some row of ``sample``."""
    dom = np.zeros(Z.shape[0], dtype=bool)
    for i in range(0, sample.shape[0], block):
        S = sample[i:i + block]
        le = np.all(S[:, None, :] <= Z[None, :, :], axis=2)
        lt = np.any(S[:, None, :] < Z[None, :, :], axis=2)
        dom |= np.any(le & lt, axis=0)
    return int(dom.sum())


def _optimal_manifold_sample(name: str, p, m: int, n: int, rng) -> np.ndarray:
    """Objective vectors of points lying on the problem's optimal manifold."""
    key = name.lower()

    if key.startswith("imop"):                  # x_dist = 0.5  ->  g = 0
        X = rng.random((n, p.n_var))
        X[:, p.K:] = 0.5
        return p.evaluate(X)

    if key.startswith("minus-"):                # x_dist at the g_MAXIMISER
        base, n_var = p._base, p.n_var
        v = np.linspace(0.0, 1.0, 20001)
        S = np.zeros((v.size, n_var))
        S[:, m - 1:] = v[:, None]
        v_star = v[base.evaluate(S).max(axis=1).argmax()]
        X = rng.random((n, n_var))
        X[:, m - 1:] = v_star
        return p.evaluate(X)

    if key.startswith("wfg"):
        # pymoo's _positional_to_optimal expects the position parameters in
        # UNIT range and applies the xu scaling itself -- passing them already
        # scaled double-scales them and lands far off the front.
        pos = rng.random((n, p._p.k))
        return p.evaluate(p._p._positional_to_optimal(pos))

    X = rng.random((n, p.n_var))                # DTLZ: x_dist = 0.5 -> g = 0
    X[:, m - 1:] = 0.5
    return p.evaluate(X)


def check(name: str, m: int = 3, n_ref: int = 2000, n_sample: int = 8000,
          seed: int = 0) -> dict:
    m_eff = problem_n_obj(name, m)
    p = get_problem(name, m=m_eff)
    Z = reference_set(name, m=m_eff, n_points=n_ref)
    rng = np.random.default_rng(seed)

    X = rng.random((n_sample, p.n_var)) * (p.xu - p.xl) + p.xl
    F_rand = p.evaluate(X)

    # (A) soundness
    n_dom = _dominated_by(F_rand, Z)

    # (B) tightness: dense sample restricted to the OPTIMAL manifold of each
    # family.  Comparing against a plain random sample would be meaningless --
    # random search never reaches g = 0 on a 12/14-variable problem, so the
    # distance would be large no matter how good Z is.
    ND = _nd_filter(_optimal_manifold_sample(name, p, m_eff, n_sample, rng))
    d, _ = cKDTree(Z).query(ND)
    span = np.linalg.norm(Z.max(axis=0) - Z.min(axis=0)) or 1.0

    return dict(problem=name, m=m_eff, n_ref=int(Z.shape[0]),
                dominated_ref_points=n_dom,
                rel_dist_mean=float(d.mean() / span),
                rel_dist_p95=float(np.percentile(d, 95) / span))


def check_determinism(name: str, m: int = 3, n_ref: int = 2000) -> bool:
    """Is Z the same set every time it is built?

    It must be: HVR divides by HV(Z) and IGD+/Eratio are measured against Z, so
    an unstable Z makes those indicators incomparable BETWEEN RUNS -- and the
    experiment builds the frame independently in every worker process.

    pymoo does not give most WFG problems an analytical front; they fall back
    to a sampler that draws 200 x 200 random interior points with an
    OS-entropy seed.  ``problems._pinned_pymoo_rng`` pins it.  This check is
    what catches that pin breaking, e.g. after a pymoo upgrade.
    """
    m_eff = problem_n_obj(name, m)
    a = reference_set(name, m=m_eff, n_points=n_ref)
    b = reference_set(name, m=m_eff, n_points=n_ref)
    return a.shape == b.shape and np.array_equal(a, b)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--determinism", action="store_true",
                    help="also rebuild every Z twice and check it is identical "
                         "(slow; run it after upgrading pymoo)")
    args = ap.parse_args()

    if args.determinism:
        print(f"{'problem':14s} {'Z estable':>10s}")
        bad = []
        for name in PROBLEM_NAMES:
            ok = check_determinism(name)
            if not ok:
                bad.append(name)
            print(f"{name:14s} {str(ok):>10s}{'   <-- NO DETERMINISTA' if not ok else ''}")
        if bad:
            raise SystemExit(f"\nnon-deterministic reference sets: {bad}\n"
                             f"every indicator measured against them is "
                             f"incomparable between runs.")
        print()

    print(f"{'problem':14s} {'m':>2s} {'|Z|':>6s} {'Z dominados':>12s} "
          f"{'d_rel medio':>12s} {'d_rel p95':>10s}")
    for name in PROBLEM_NAMES:
        r = check(name)
        flag = "  <-- REVISAR" if (r["dominated_ref_points"] > 0.01 * r["n_ref"]
                                   or r["rel_dist_p95"] > 0.10) else ""
        print(f"{r['problem']:14s} {r['m']:2d} {r['n_ref']:6d} "
              f"{r['dominated_ref_points']:12d} {r['rel_dist_mean']:12.5f} "
              f"{r['rel_dist_p95']:10.5f}{flag}")


if __name__ == "__main__":
    main()
