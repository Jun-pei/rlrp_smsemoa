"""RL-RP-SMS-EMOA: RL-based online reference-point specification for HV-based MOEAs."""
from .algorithm import (
    METHODS, RLRPSMSEMOA, run_d_sms_emoa, run_rl_rp_sms_emoa,
    run_sms_emoa_balanced,
)
from .core import DEFAULTS, History
from .experiment import add_time_to_target, get_frame, run_comparison, run_single
from .history_io import iter_histories, load_history, save_history, stack_curves
from .imop import IMOP_M, IMOP_NAMES, get_imop
from .indicators import (
    estimate_curvature_p, estimate_ideal_nadir, extent, geometry_gamma,
    hv_contributions, hypervolume, igd_plus, mean_dispersion, nondominated,
    normalize, normalised_dispersion, riesz_energy, riesz_energy_log,
)
from .performance import (
    KAPPA_EXT, TARGET_Q, ReferenceFrame, anytime_score, rank_biserial,
    time_to_target,
)
from .problems import (
    BENCHMARK, DTLZ_WFG_NAMES, IMOP_SUITE, PROBLEM_NAMES, Problem, get_problem,
    make_reference_frame, problem_n_obj, reference_set,
)
from .r2_emoa import r2_contributions, r2_indicator, r2_weights, run_r2_emoa
from .rl_planner import REGIME_NAMES, ZREF_MAX, ReplicatorQPlanner, regime_deltas
from .state import StateEncoder, invertedness
from .stats import friedman_test, quade_test, wilcoxon_bonferroni

__all__ = [n for n in dir() if not n.startswith("_")]
