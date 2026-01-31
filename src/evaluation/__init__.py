"""
Evaluation module for QuantumMind.

Provides baseline ansatzes and metrics for comparing discovered circuits.
"""

from src.evaluation.baselines import (
    compare_baselines,
    efficient_su2_ansatz,
    excitation_preserving_ansatz,
    get_baseline,
    get_circuit_info,
    hardware_efficient_ansatz,
    list_baselines,
    real_amplitudes_ansatz,
    two_local_ansatz,
)
from src.evaluation.metrics import (
    ComparisonMetrics,
    EvaluationMetrics,
    compare_metrics,
    compute_improvement_vs_baseline,
    compute_metrics,
    format_metrics_table,
    significance_test,
    statistical_evaluation,
)

__all__ = [
    # Baselines
    "compare_baselines",
    "efficient_su2_ansatz",
    "excitation_preserving_ansatz",
    "get_baseline",
    "get_circuit_info",
    "hardware_efficient_ansatz",
    "list_baselines",
    "real_amplitudes_ansatz",
    "two_local_ansatz",
    # Metrics
    "ComparisonMetrics",
    "EvaluationMetrics",
    "compare_metrics",
    "compute_improvement_vs_baseline",
    "compute_metrics",
    "format_metrics_table",
    "significance_test",
    "statistical_evaluation",
]
