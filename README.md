# RL-RP-SMS-EMOA

Python implementation of **"Un Método Basado en Aprendizaje por Refuerzo para la Especificación en Línea del Punto de Referencia en MOEAs Basados en Hipervolumen"** (CICESE, Dr. Jesús Guillermo Falcón Cardona).

---

## Project structure

```
rlrp_smsemoa/
├── problems.py            # 8 test problems (Cuadro 1): DTLZ1/2, Minus-DTLZ1/2,
│                          #   WFG4, WFG9, IMOP3-like, IMOP8-like
├── indicators.py          # HV, HVC, IGD+, Riesz E_s, dispersion D(t)
├── state.py               # State encoder: 5-dim feature vector -> 675 discrete states
├── rl_planner.py          # Replicator dynamics (Eq. 9) + Q-learning (Eq. 10)
├── sms_emoa.py            # SBX + PM operators, steady-state SMS-EMOA elimination
├── algorithm.py           # Algorithm 1 (RL-RP-SMS-EMOA) + 3 baseline methods
├── experiment.py          # Single-run helper, multi-seed runner, true-front sampler
├── stats.py               # Wilcoxon/Bonferroni, Friedman, Quade tests (Sec. 5.3)
├── run_demo.py            # Quick demonstration (small mu/Tmax, 2 problems)
├── run_full_experiment.py # Full protocol: 8 problems × 4 methods × 30 seeds
├── run_sensitivity.py     # Sensitivity sweeps: W, mu, rho (Sec. 5.3 a-c)
└── requirements.txt
```

---

## Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+, pymoo 0.6.1.6.

---

## Quick start

```bash
python3 run_demo.py --mu 30 --t_max 400 --n_seeds 5 --outdir results/demo
```

Runs 4 methods × 5 seeds on **DTLZ2** and **Minus-DTLZ1** (≈3 min on a laptop), produces:

- `convergence_{problem}.png` — HV (external far PR) convergence curves
- `planner_{problem}.png`     — σ(t) regime probabilities + z_ref(t) trajectory
- `fronts_{problem}.png`      — Final 3D non-dominated fronts, 4-panel
- `results_{problem}.csv`     — Raw per-seed indicator values

---

## Full experiment (Sec. 5.3, 30 seeds)

```bash
# All 8 problems, 30 seeds, Tmax=100 000, mu=100, 8 parallel workers
python3 run_full_experiment.py \
    --problems all \
    --n_seeds 30 \
    --t_max 100000 \
    --mu 100 \
    --processes 8 \
    --outdir results/full
```

> **Expected runtime**: each (problem, method, seed) run at mu=100, Tmax=100 000 takes
> roughly 10-30 min in pure Python. The full 8×4×30=960 runs are feasible on a
> compute node with `--processes 32` overnight. For faster iteration, lower
> `--t_max` (e.g. 5000) to check correctness before committing to the full run.

Outputs:
- `results_{problem}.csv` — Raw per-seed indicators (one row per seed)
- `all_results.csv`       — Concatenation of all problems
- `summary.csv`           — Grouped mean ± std per (problem, method)
- `stats.txt`             — Friedman, Quade, Wilcoxon/Bonferroni results

---

## Sensitivity analysis (Sec. 5.3 a–c)

```bash
# (a) W in {10, 20, 30, 50}
python3 run_sensitivity.py --sweep W --problems all --n_seeds 30 \
    --t_max 100000 --mu 100

# (b) mu in {91, 100, 153, 210}  (Das-Dennis H in {12, 16, 20} for m=3)
python3 run_sensitivity.py --sweep mu --problems all --n_seeds 30 \
    --t_max 100000

# (c) rho in {0.5, 0.7, 0.9}
python3 run_sensitivity.py --sweep rho --problems all --n_seeds 30 \
    --t_max 100000 --mu 100

# Or all at once:
python3 run_sensitivity.py --sweep all --problems all --n_seeds 30 \
    --t_max 100000 --mu 100 --outdir results/sensitivity
```

---

## Using the API programmatically

```python
from problems import get_problem
from algorithm import run_rl_rp_sms_emoa, run_sms_emoa_balanced

problem = get_problem("dtlz2", m=3)

# Run RL-RP-SMS-EMOA with default hyperparameters
X, F, hist = run_rl_rp_sms_emoa(problem, t_max=10_000, seed=0, mu=100)

print(f"Final pop: {F.shape}")
print(f"HV (far PR, final 5 gens): {hist.hv_ext_far[-5:]}")
print(f"Final sigma: {hist.sigma[-1]}")  # regime distribution at end of adaptation

# Compare to balanced baseline
X_b, F_b, hist_b = run_sms_emoa_balanced(problem, t_max=10_000, seed=0, mu=100)
```

### Hyperparameter API

```python
X, F, hist = run_rl_rp_sms_emoa(
    problem, t_max=10_000, seed=0,
    mu=100,          # population size
    H=12,            # Das-Dennis parameter for balanced regime delta_2 = 1/H
    rho=0.7,         # fraction of Tmax allocated to RL adaptation phase
    W=20,            # EMA sliding window size (payoff estimation)
    eps0=0.1,        # initial ε for ε-greedy action selection (decays to eps_min)
    eps_min=0.01,
    alpha_rl=0.1,    # Q-learning step size
    gamma=0.9,       # discount factor
    lam=0.9,         # EMA smoothing coefficient for payoffs
    sigma0=(0.10, 0.10, 0.80),  # initial regime probabilities (fine/balanced/coarse)
    alpha_reward=1.0,           # weight of Riesz-energy improvement in reward
    q_reset_every=1000,         # soft Q-table reset period (during adaptation)
    q_reset_alpha=0.1,          # fraction of Q wiped per reset
)
```

### Accessing the History object

```python
import numpy as np

hist.hv_adaptive    # list[float] — HV w.r.t. adaptive z_ref
hist.hv_ext_far     # list[float] — HV w.r.t. z_ref_true_nadir + 1.1 (Cuadro 2)
hist.hv_ext_near    # list[float] — HV w.r.t. z_ref_true_nadir + 0.1 (robustness check)
hist.igd_plus       # list[float]
hist.riesz_log      # list[float] — log Riesz energy E^ln_s(And(t))
hist.dispersion     # list[float] — D(t), mean per-dim std dev
hist.zref           # list[np.ndarray] — z_ref(t) trajectory
hist.sigma          # list[np.ndarray] — σ(t) (only during adaptation phase)
hist.regime         # list[int]        — k_t chosen at each generation
hist.reward         # list[float]      — r_t (only during adaptation phase)
```

---

## Algorithm overview (Algorithm 1)

```
Inputs: problem, mu=100, Tmax, rho=0.7, W, eps0=0.1, H=12
Outputs: non-dominated set P

1.  t_adapt = ceil(rho * Tmax)
2.  Initialize P with mu random solutions; sigma = (0.10, 0.10, 0.80)
3.  Q[s,(k,j)] = 0  (675 states × 18 actions)
4.  z_ref = nadir_hat + 1 * ones  [satisfies z_ref >= (1,...,1) after normalization]

5.  for t = 1 ... Tmax:
6.    Estimate ideal, nadir from P; normalize objective space
7.    if t <= t_adapt:    [Adaptation phase]
8.      Compute state s_t = (D_norm, HV_norm, t/Tmax, g_hat, E_norm) -> discrete index
9.      Update EMA payoffs u_k(sigma, t)  using sliding window W       [Eq. 7-8]
10.     Update sigma(t) via replicator dynamics                         [Eq. 9]
11.     Sample k_t ~ Categorical(sigma(t))
12.     Select j_t = argmax_j Q[s_t, (k_t, j)] or uniform (eps-greedy)
13.     Complete deferred Q-update from t-1 now that (s_t, k_t) are known [Eq. 10]
14.     Update z_ref_i <- max(1, z_ref_i + s*delta_{k_t})              [Sec. 2.2]
15.   else:               [Refinement phase]
16.     z_ref stays fixed at z_ref(t_adapt)
17.   Generate offspring via SBX (pc=1.0, eta_c=15) + PM (pm=1/n, eta_m=20)
18.   P' = P + {offspring}; eliminate least-HVC individual from worst front
19.   Compute reward r_t = Delta_HV_norm + alpha * Delta_E_norm
20.   Store (s_t, k_t, j_t, r_t) for deferred Q-update at t+1
21.   Decay eps linearly to eps_min
22. return non-dominated P
```

---

## Key implementation decisions

### IMOP3 / IMOP8 (Cuadro 1)
The original IMOP suite (Tian et al., IEEE CIM 2019) is MATLAB-only (PlatEMO).
The proposal's Cuadro 1 calls for 3-objective versions with n=(m-1)+l=7 variables,
which do not match the published IMOP3/8 (2-objective, and 3-objective respectively
but with different variable counts). We implement:

- **IMOP3Like**: spherical front with a density-bias warp $x_i' = x_i^a$ ($a=0.05$),
  faithfully reproducing the *mechanism* of the real IMOP suite (non-uniform
  density over an otherwise-regular front).
- **IMOP8Like**: multi-segment front (3 segments: linear / concave / convex),
  producing genuine discontinuities as in IMOP8.

To replace with the exact PlatEMO implementations, port `IMOP3.m` / `IMOP8.m`
from https://github.com/BIMK/PlatEMO/tree/master/PlatEMO/Problems/IMOP and
adapt them to the `Problem` interface in `problems.py`.

### Q-learning update sequencing (Eq. 10)
The pseudocode's bootstrap target for the update at generation $t$ is
$\max_{j'} Q[s_{t+1}, (k_{t+1}, j')]$, which requires knowing $k_{t+1}$ first.
We use a *deferred* (one-step-lag) update: $(s_t, k_t, j_t, r_t)$ is stashed
and the TD update is completed at the *start* of generation $t+1$ once
$(s_{t+1}, k_{t+1})$ are known.

### HV computation
We use pymoo's exact `HV` indicator (based on the WFG algorithm) for the
external indicator evaluations (Cuadro 2) and our own incremental `hv_contributions`
function (also based on pymoo) for the SMS-EMOA elimination step and the $\hat{g}(t)$
feature (Eq. 6). For very large populations (mu=210, Sec. 2.2 sweep) where HV
becomes a bottleneck, `FV-MOEA` integration is noted in the proposal (Sec. 6.1)
as a future mitigation.

---

## Note on results at reduced scale

The demo (`run_demo.py`) uses mu=30 and Tmax≤500, which is far below the
proposal's mu=100, Tmax=100,000. At this short scale the RL agent is still
exploring (sigma has not yet converged) and the adaptive z_ref has had little
time to home in on the geometry-optimal position. The full 100,000-generation
run at mu=100 is where the planner's benefit is expected to be visible. This is
consistent with the proposal's risk discussion (Sec. 6.1, "No estacionariedad
del espacio de estado") and with the NRLPSO paper's observation that the
replicator dynamics needs enough evaluations to distinguish regime payoffs.

---

## References (from the proposal)

1. Auger et al. (2009). Theory of the hypervolume indicator. FOGA.
2. Beume et al. (2007). SMS-EMOA. Eur. J. Oper. Res.
3. Brockhoff (2010). Optimal μ-distributions for HV. SEAL.
4. Das & Dennis (1998). Normal-boundary intersection. SIAM J. Optim.
5. Hamdi et al. (2026). NRLPSO. ICAART.
6. Ishibuchi et al. (2017). Reference point specification for HV. GECCO.
7. Ishibuchi et al. (2018). Dynamic specification of a reference point. CEC.
8. Ishibuchi et al. (2018). Two reference points in HV-based MOEA. PPSN.
9. Morales-Paredes et al. (2025). Reference point specification in greedy HV. GECCO.
10. Nemhauser et al. (1978). Submodular set functions. Math. Programming.
11. Shang & Ishibuchi et al. (2021). HV-optimal μ-distributions. arXiv:2104.09736.
12. Tian et al. (2019). Diversity assessment of MOEAs. IEEE CIM.
13. Zitzler et al. (2007). The HV indicator revisited. EMO.
