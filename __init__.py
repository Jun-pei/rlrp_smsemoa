"""RL-RP-SMS-EMOA: RL-based online reference-point specification for hypervolume-based MOEAs."""
from .problems import get_problem, PROBLEM_NAMES, Problem
from .algorithm import (
    RLRPSMSEMOA, History, METHODS, DEFAULTS,
    run_rl_rp_sms_emoa,
    run_sms_emoa_nadir,
    run_sms_emoa_balanced,
    run_sms_emoa_dynlin,
)
from .indicators import hypervolume, igd_plus, riesz_energy, riesz_energy_log, mean_dispersion
from .stats import wilcoxon_bonferroni, friedman_test, quade_test
from .experiment import run_comparison, run_single, sample_true_front, pivot_indicator

__all__ = [
    "get_problem", "PROBLEM_NAMES", "Problem",
    "RLRPSMSEMOA", "History", "METHODS", "DEFAULTS",
    "run_rl_rp_sms_emoa", "run_sms_emoa_nadir", "run_sms_emoa_balanced", "run_sms_emoa_dynlin",
    "hypervolume", "igd_plus", "riesz_energy", "riesz_energy_log", "mean_dispersion",
    "wilcoxon_bonferroni", "friedman_test", "quade_test",
    "run_comparison", "run_single", "sample_true_front", "pivot_indicator",
]
