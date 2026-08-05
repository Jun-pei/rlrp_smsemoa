# RL-RP-SMS-EMOA

Python implementation of **"Un Método Basado en Aprendizaje por Refuerzo para la
Especificación en Línea del Punto de Referencia en MOEAs Basados en Hipervolumen"**
(CICESE, Dr. Jesús Guillermo Falcón Cardona).

SMS-EMOA whose hypervolume reference point is moved online by a two-level
planner: replicator dynamics chooses the *step size regime* (fine / balanced /
coarse) and a tabular Q-learner chooses the *direction* (which objective, which
sign). See `RESPUESTAS.md` for the answers to the seminar observations and for
what the measurements actually show.

---

## Project structure

```
rlrp_smsemoa/
├── problems.py             # problem registry + analytical reference sets Z
├── imop.py                 # IMOP1..IMOP8, ported from PlatEMO (CalObj + GetOptimum)
├── indicators.py           # HV, HVC, IGD+, Riesz energy, dispersion, curvature
├── sms_emoa.py             # SBX + PM operators, steady-state SMS-EMOA elimination
├── state.py                # 5-feature state encoder -> 675 discrete states
├── rl_planner.py           # replicator dynamics (Eq. 9) + Q-learning (Eq. 10)
├── core.py                 # shared config, History, generation step, logging
├── algorithm.py            # RL-RP-SMS-EMOA + the reference-point baselines
├── r2_emoa.py              # R2-EMOA, the indicator-agnostic control
├── performance.py          # Definitions D1-D9: what "performance" means
├── history_io.py           # per-generation histories on disk
├── experiment.py           # one run / one comparison
├── stats.py                # Wilcoxon+Bonferroni, Friedman, Quade
├── run_full_experiment.py  # the Sec. 5.3 protocol (CLI)
├── run_sensitivity.py      # the Sec. 5.3 a-c sweeps (CLI)
└── tests_reference_sets.py # validation of every Z used for IGD+
```

---

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+, pymoo 0.6.1.6. The repository root *is* the `rlrp_smsemoa`
package, so its parent directory must be on the import path; the CLI scripts
arrange that themselves.

---

## Quick start

```bash
# smoke run: 2 problems x 4 methods x 3 seeds, ~2 minutes
python3 run_full_experiment.py --problems dtlz2 imop3 --n_seeds 3 \
    --t_max 1000 --mu 30 --outdir results/smoke
```

Or with Docker:

```bash
docker compose run --rm smoke     # short check
docker compose run --rm full      # the whole protocol
docker compose run --rm imop      # the IMOP suite on its own
```

---

## The benchmark

Fourteen problems: the six regular geometries of Cuadro 1 plus the **complete**
IMOP suite.

| Problem | Front | m | n |
|---|---|---|---|
| DTLZ1 | linear triangular | 3 | 7 |
| DTLZ2 | concave spherical | 3 | 12 |
| Minus-DTLZ1 | inverted linear | 3 | 7 |
| Minus-DTLZ2 | inverted spherical | 3 | 12 |
| WFG4 | multimodal concave | 3 | 14 |
| WFG9 | degenerate / irregular | 3 | 14 |
| IMOP1 | biased density, convex | 2 | 10 |
| IMOP2 | biased density, concave | 2 | 10 |
| IMOP3 | disconnected | 2 | 10 |
| IMOP4 | degenerate curve-like | 3 | 10 |
| IMOP5 | eight disconnected patches | 3 | 10 |
| IMOP6 | planar with holes | 3 | 10 |
| IMOP7 | thin band | 3 | 10 |
| IMOP8 | highly multimodal | 3 | 10 |

**IMOP is not parameterised in `m`.** IMOP1–3 are bi-objective and IMOP4–8
tri-objective by definition of the benchmark, so `problem_n_obj()` returns the
correct `m` and `get_imop()` raises rather than fabricating a 3-objective
"IMOP3" that does not exist in the literature. Cuadro 1 asked for one; it was
previously supplied by an invented `IMOP3Like` surrogate, which is gone.

---

## The four compared configurations

| Method | Reference point | Source |
|---|---|---|
| `SMS-EMOA_balanced` | fixed, `(1 + 1/H)·1` | Ishibuchi et al., GECCO 2017 |
| `d-SMS-EMOA` | dynamic, `r` from 10 down to 1 | Ishibuchi et al., CEC 2018 |
| `R2-EMOA` | none (R2 selection) | Trautmann, Wagner & Brockhoff, LION 2013 |
| `RL-RP-SMS-EMOA` | learned online | proposed |

All four share the same operators, the same steady-state elimination structure
and the same instrumentation, so any difference between them is a difference in
the reference-point / selection mechanism.

**`d-SMS-EMOA`** (also written SMS-EMOA-DRP) replaces the `SMS-EMOA_nadir`
baseline of the original proposal. A reference point pinned at `nadir + 0.01`
is a straw man: it sits so close to the nadir that the extreme solutions
contribute almost no hypervolume, and it is not what any of the cited papers
propose. d-SMS-EMOA is the published dynamic scheme — `r = 10` at the initial
population, falling to `r = 1` at the final one, so the search first spreads
towards the boundary of the front and then concentrates on its centre.

---

## Full protocol (Sec. 5.3)

```bash
python3 run_full_experiment.py \
    --problems all \
    --n_seeds 30 \
    --t_max 100000 \
    --mu 100 \
    --processes 32 \
    --outdir results/full
```

Outputs in `--outdir`:

| File | Contents |
|---|---|
| `history/<problem>/<method>/seed<NNN>.csv.gz` | **every generation of every run** |
| `history/.../seed<NNN>.meta.json` | run metadata: \|S\|, \|Z\|, `t_adapt`, Q-table coverage |
| `results_<problem>.csv` | one summary row per (method, seed) |
| `all_results.csv` | all of them, plus time-to-target (D5) |
| `summary.csv` | median + IQR per (problem, method) |
| `stats.txt` | Friedman, Quade, Wilcoxon/Bonferroni with effect sizes |

### Everything is stored, not just the final values

One row per generation, unconditionally — about 40 columns: the full `z_ref`,
the estimated ideal/nadir, HV against both the adaptive and a fixed reference
point, dispersion, Riesz energy, fitted curvature, HVR / IGD+ / Eratio, the
chosen regime and sub-action, the state index, the reward, ε, σ, the payoffs
and the state features. That is roughly 3–4 MB per 100 000-generation run.

This is not bookkeeping for its own sake: the finding in section 12 of
`RESPUESTAS.md` — that the per-generation reward carries no signal about the
eventual improvement (Spearman ρ = +0.008) — is invisible in final-value tables
and only shows up in the traces.

`--eval_every` thins **only** the external indicators (HVR / IGD+ / Eratio
against the reference set Z), which are the expensive part. It defaults to 1,
i.e. nothing is thinned; raise it if a large-`mu` sweep makes it the bottleneck.

---

## Running on a cluster

**Python 3.10 or newer is required** — pymoo 0.6.1.6 declares
`requires_python >= 3.10`, so a system default of 3.9 will not work. Load a
newer interpreter (e.g. a conda module) and build an environment:

```bash
module load conda/conda-2026          # ships Python 3.13.11
conda create -y -n rlrp python=3.13
source activate rlrp
pip install -r requirements.txt       # cp313 wheels exist for all four
```

One array task per problem, all methods and seeds inside the task:

```bash
sbatch --array=0-13 slurm/run_array.sbatch
```

The workload is pure CPU (numpy / pymoo); there is no GPU code path, so it
belongs on a CPU-only partition. Peak memory is ~400 MB per concurrent run at
`T_max = 100000` — the per-generation history is held in RAM until the run
ends — hence `--mem-per-cpu=2G`. The partition, core count and wall clock in
the script are set for ixachi's `gold5320` (52-core, 256 GB nodes); override
with `sbatch --partition=... --array=0-13 slurm/run_array.sbatch` elsewhere.

Each task writes its summary CSVs to `results/full/<problem>/` and its
per-generation histories into the shared `results/full/history/` tree. When
every task has finished, merge them:

```bash
python3 aggregate_results.py --root results/full --outdir results/full
```

Time to target (D5) is recomputed during the merge, because it is defined
relative to the best HVR reached by *any* method on that (problem, seed) and
each array task only saw its own problem.

**Sizing.** Measured at `mu=100`, `T_max=100000`, `eval_every=1`, one core per
run: roughly **1 core-hour per run** on a 3-objective DTLZ/WFG problem
(30–49 ms/generation depending on method and problem; the bi-objective IMOPs
are cheaper). The full grid is 14 × 4 × 30 = 1680 runs ≈ **1700 core-hours**,
so ~2 days of wall time on 32 cores. Setting `--eval_every 100` cuts it to
≈ 350 core-hours at the cost of evaluating the external indicators on a
1000-point grid instead of every generation — the per-generation log is
unaffected either way.

---

## Sensitivity analysis (Sec. 5.3 a–c)

```bash
python3 run_sensitivity.py --problem dtlz2 --param W   --values 10 20 30 50
python3 run_sensitivity.py --problem dtlz2 --param mu  --values 91 100 153 210
python3 run_sensitivity.py --problem dtlz2 --param rho --values 0.5 0.7 0.9
```

---

## Validating the reference sets

```bash
python3 -m rlrp_smsemoa.tests_reference_sets
```

Checks, for all fourteen problems, that no point of `Z` is dominated by a large
random sample (soundness) and that a dense sample of the true optimal manifold
lies close to `Z` (tightness). IGD+ against a wrong `Z` is not IGD+, so this is
run before trusting any IGD+ number.

---

## Programmatic use

```python
from rlrp_smsemoa import get_problem, get_frame, run_single, METHODS

# one run, full history written to disk
row = run_single("imop8", "RL-RP-SMS-EMOA", seed=0, t_max=10_000,
                 histdir="results/history", mu=100)
print(row["hvr_final"], row["igd_plus_final"], row["history_path"])

# or drive an algorithm directly
problem = get_problem("minus-dtlz2", m=3)
frame = get_frame("minus-dtlz2", m=3)
X, F, hist = METHODS["d-SMS-EMOA"](problem, t_max=5_000, seed=0, frame=frame)
df = hist.to_dataframe()          # one row per generation
```

### Planner hyper-parameters

```python
from rlrp_smsemoa import run_rl_rp_sms_emoa

X, F, hist = run_rl_rp_sms_emoa(
    problem, t_max=10_000, seed=0,
    mu=100,           # population size
    H=12,             # Das-Dennis parameter; balanced step is delta_2 = 1/H
    rho=0.7,          # fraction of T_max spent adapting z_ref
    W=20,             # sliding window for the payoff EMA
    eps0=0.1, eps_min=0.01,        # eps-greedy schedule
    alpha_rl=0.1, gamma=0.9,       # Q-learning
    lam=0.9,                       # payoff EMA smoothing
    sigma0=(0.10, 0.10, 0.80),     # initial regime probabilities
    alpha_reward=1.0,              # weight of the uniformity term in r_t
    q_reset_every=1000, q_reset_alpha=0.1,
    zref_max=10.0,                 # the reference point lives in [1+eps, 10]^m
    q_update_regimes="all",        # "balanced" reproduces Eq. 10 literally
    sigma_floor=0.05,              # replicator-MUTATOR floor; 0 = plain Eq. 9
    action_every=1,                # planner acts every tau generations
    reward_hv="fixed",             # or "adaptive" (HV against the moving z_ref)
    geometry_mode="curvature",     # or "contour" (Eq. 6), or "both"
    eval_every=1,
)
```

---

## Implementation notes

**Q-learning update order (Eq. 10).** The bootstrap target needs
`max_j Q[s_{t+1}, (k_{t+1}, j)]`, i.e. the regime the replicator dynamics will
pick at `t+1`. The update is therefore deferred by one step: `(s_t, k_t, j_t)`
is stashed and the TD update completed at the start of generation `t+1`. This
preserves the dependency structure of the equation exactly.

**Hypervolume.** pymoo's exact `HV` (WFG algorithm) for the indicator
evaluations and for the elimination step, with a Monte-Carlo backend selected
automatically above 5 objectives, so nothing depends on exact HV staying
affordable.

**Deviations from the pseudocode**, all deliberate and all switchable:
`q_update_regimes="all"`, `sigma_floor > 0`, `action_every > 1`,
`reward_hv="fixed"`, the box bound on `z_ref`, and the two-sided `E_norm`.
Each is justified in `RESPUESTAS.md`, and each has the literal behaviour of the
proposal available as an option.

---

## References

1. Auger et al. (2009). Theory of the hypervolume indicator. FOGA.
2. Beume, Naujoks & Emmerich (2007). SMS-EMOA. Eur. J. Oper. Res. 181(3).
3. Brockhoff (2010). Optimal μ-distributions for HV. SEAL.
4. Das & Dennis (1998). Normal-boundary intersection. SIAM J. Optim.
5. Hamdi et al. (2026). NRLPSO. ICAART.
6. Ishibuchi, Imada, Setoguchi & Nojima (2017). Reference point specification in
   hypervolume calculation. GECCO.
7. Ishibuchi, Imada, Masuyama & Nojima (2018). Dynamic specification of a
   reference point for hypervolume calculation in SMS-EMOA. CEC, pp. 701–708.
8. Ishibuchi, Imada, Masuyama & Nojima (2018). Use of two reference points in
   hypervolume-based EMO. PPSN.
9. Morales-Paredes, Falcón-Cardona et al. (2025). Reference point specification
   in greedy inclusion HV-based subset selection. GECCO.
10. Nemhauser, Wolsey & Fisher (1978). Submodular set functions. Math. Prog.
11. Shang, Ishibuchi et al. (2021). HV-optimal μ-distributions. arXiv:2104.09736.
12. Tian, Cheng, Zhang, Cheng & Jin (2019). Diversity assessment of MOEAs
    (the IMOP suite). IEEE Comput. Intell. Mag. 14(3).
13. Trautmann, Wagner & Brockhoff (2013). R2-EMOA. LION 7, LNCS 7997.
14. Zitzler, Brockhoff & Thiele (2007). The HV indicator revisited. EMO.
