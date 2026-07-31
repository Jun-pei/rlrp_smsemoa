"""RL-RP-SMS-EMOA: RL-based online reference-point specification for HV-based MOEAs."""
from .problems import (
    get_problem, reference_set, make_reference_frame, problem_n_obj,
    PROBLEM_NAMES, TABLE1_NAMES, IMOP_SUITE, Problem,
)
from .imop import get_imop, IMOP_NAMES, IMOP_M
from .algorithm import (
    RLRPSMSEMOA, History, METHODS, LEGACY_METHODS, DEFAULTS,
    run_rl_rp_sms_emoa, run_sms_emoa_nadir, run_sms_emoa_balanced, run_sms_emoa_dynlin,
)
from .indicators import (
    hypervolume, hv_contributions, igd_plus, riesz_energy, riesz_energy_log,
    mean_dispersion, normalised_dispersion, extent, estimate_curvature_p,
    geometry_gamma, estimate_ideal_nadir, normalize, nondominated,
)
from .state import StateEncoder, invertedness
from .rl_planner import ReplicatorQPlanner, regime_deltas, REGIME_NAMES, ZREF_MAX
from .performance import (
    ReferenceFrame, anytime_score, time_to_target, rank_biserial,
    KAPPA_EXT, TARGET_Q,
)
from .r2_emoa import (
    run_r2_emoa, run_sms_emoa_nadir_adaptive, r2_indicator, r2_contributions,
    r2_weights,
)
from .stats import wilcoxon_bonferroni, friedman_test, quade_test
from .experiment import (
    run_comparison, run_single, pivot_indicator, get_frame, add_time_to_target,
)
from .history_io import save_history, load_history, iter_histories, stack_curves

__all__ = [n for n in dir() if not n.startswith("_")]
