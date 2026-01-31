"""
Metrics module for QuantumMind.

Provides comprehensive evaluation metrics for comparing quantum circuits
including energy accuracy, efficiency, and statistical analysis.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics for a circuit."""

    # Energy metrics
    energy: float
    energy_error: float
    relative_error: float

    # Circuit metrics
    depth: int
    gate_count: int
    param_count: int
    two_qubit_count: int

    # Efficiency metrics
    energy_per_depth: float
    energy_per_param: float
    energy_per_gate: float

    # Quality metrics
    fidelity: float | None = None
    gradient_variance: float | None = None

    # Optimization metrics
    vqe_iterations: int | None = None
    converged: bool = True

    # Extra details
    details: dict = field(default_factory=dict)


@dataclass
class ComparisonMetrics:
    """Metrics comparing two circuits."""

    # Which is better
    winner: str  # "circuit_a" or "circuit_b"

    # Relative improvements (positive = circuit_a is better)
    energy_improvement: float  # percentage
    depth_improvement: float
    param_improvement: float

    # Absolute differences
    energy_diff: float
    depth_diff: int
    param_diff: int

    # Overall assessment
    is_significant: bool
    summary: str


def compute_metrics(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    exact_energy: float,
    vqe_result: Any = None,
    exact_state: Statevector | None = None,
    gradient_variance: float | None = None,
) -> EvaluationMetrics:
    """
    Compute comprehensive evaluation metrics for a circuit.

    Args:
        circuit: The quantum circuit to evaluate
        hamiltonian: Hamiltonian operator
        exact_energy: Exact ground state energy
        vqe_result: Optional VQE result object
        exact_state: Optional exact ground state for fidelity
        gradient_variance: Optional gradient variance from BP detection

    Returns:
        Complete evaluation metrics
    """
    # Circuit metrics
    depth = circuit.depth()
    gate_count = sum(circuit.count_ops().values())
    param_count = len(circuit.parameters)
    ops = circuit.count_ops()
    two_qubit_count = sum(
        ops.get(g, 0) for g in ["cx", "cz", "swap", "rxx", "ryy", "rzz"]
    )

    # Energy metrics from VQE result
    if vqe_result is not None:
        energy = vqe_result.energy
        vqe_iterations = getattr(vqe_result, "iterations", None)
        converged = getattr(vqe_result, "converged", True)
    else:
        energy = 0.0
        vqe_iterations = None
        converged = True

    energy_error = abs(energy - exact_energy)
    relative_error = energy_error / abs(exact_energy) if exact_energy != 0 else energy_error

    # Efficiency metrics (lower is better)
    energy_per_depth = energy_error / depth if depth > 0 else float("inf")
    energy_per_param = energy_error / param_count if param_count > 0 else float("inf")
    energy_per_gate = energy_error / gate_count if gate_count > 0 else float("inf")

    # Fidelity
    fidelity = None
    if exact_state is not None and vqe_result is not None:
        final_state = getattr(vqe_result, "final_state", None)
        if final_state is not None:
            fidelity = state_fidelity(final_state, exact_state)

    return EvaluationMetrics(
        energy=energy,
        energy_error=energy_error,
        relative_error=relative_error,
        depth=depth,
        gate_count=gate_count,
        param_count=param_count,
        two_qubit_count=two_qubit_count,
        energy_per_depth=energy_per_depth,
        energy_per_param=energy_per_param,
        energy_per_gate=energy_per_gate,
        fidelity=fidelity,
        gradient_variance=gradient_variance,
        vqe_iterations=vqe_iterations,
        converged=converged,
        details={
            "ops": dict(ops),
            "num_qubits": circuit.num_qubits,
        },
    )


def compare_metrics(
    metrics_a: EvaluationMetrics,
    metrics_b: EvaluationMetrics,
    significance_threshold: float = 0.05,
) -> ComparisonMetrics:
    """
    Compare metrics between two circuits.

    Args:
        metrics_a: Metrics for circuit A
        metrics_b: Metrics for circuit B
        significance_threshold: Threshold for significant improvement

    Returns:
        Comparison metrics
    """
    # Energy improvement (positive = A is better = lower error)
    if metrics_b.energy_error > 0:
        energy_improvement = (
            (metrics_b.energy_error - metrics_a.energy_error)
            / metrics_b.energy_error
            * 100
        )
    else:
        energy_improvement = 0.0

    # Depth improvement (positive = A is better = shallower)
    if metrics_b.depth > 0:
        depth_improvement = (metrics_b.depth - metrics_a.depth) / metrics_b.depth * 100
    else:
        depth_improvement = 0.0

    # Parameter improvement (positive = A is better = fewer params)
    if metrics_b.param_count > 0:
        param_improvement = (
            (metrics_b.param_count - metrics_a.param_count) / metrics_b.param_count * 100
        )
    else:
        param_improvement = 0.0

    # Determine winner based on energy error (primary metric)
    if metrics_a.energy_error < metrics_b.energy_error:
        winner = "circuit_a"
    elif metrics_b.energy_error < metrics_a.energy_error:
        winner = "circuit_b"
    else:
        # Tie-breaker: use depth
        winner = "circuit_a" if metrics_a.depth <= metrics_b.depth else "circuit_b"

    # Check significance
    is_significant = abs(energy_improvement) > significance_threshold * 100

    # Generate summary
    if winner == "circuit_a":
        summary = f"Circuit A is better: {abs(energy_improvement):.1f}% lower error"
        if depth_improvement > 0:
            summary += f", {depth_improvement:.1f}% shallower"
    else:
        summary = f"Circuit B is better: {abs(energy_improvement):.1f}% lower error"
        if depth_improvement < 0:
            summary += f", {abs(depth_improvement):.1f}% shallower"

    return ComparisonMetrics(
        winner=winner,
        energy_improvement=energy_improvement,
        depth_improvement=depth_improvement,
        param_improvement=param_improvement,
        energy_diff=metrics_a.energy_error - metrics_b.energy_error,
        depth_diff=metrics_a.depth - metrics_b.depth,
        param_diff=metrics_a.param_count - metrics_b.param_count,
        is_significant=is_significant,
        summary=summary,
    )


def statistical_evaluation(
    results: list[EvaluationMetrics],
) -> dict:
    """
    Compute statistical summary of multiple evaluation runs.

    Args:
        results: List of evaluation metrics from multiple runs

    Returns:
        Dictionary with statistical summary
    """
    if not results:
        return {}

    energies = [r.energy for r in results]
    errors = [r.energy_error for r in results]
    relative_errors = [r.relative_error for r in results]

    return {
        "num_runs": len(results),
        # Energy statistics
        "mean_energy": np.mean(energies),
        "std_energy": np.std(energies),
        "best_energy": min(energies),
        "worst_energy": max(energies),
        # Error statistics
        "mean_error": np.mean(errors),
        "std_error": np.std(errors),
        "best_error": min(errors),
        "worst_error": max(errors),
        # Relative error statistics
        "mean_relative_error": np.mean(relative_errors),
        "std_relative_error": np.std(relative_errors),
        # Convergence
        "convergence_rate": sum(r.converged for r in results) / len(results),
        # Iterations
        "mean_iterations": np.mean(
            [r.vqe_iterations for r in results if r.vqe_iterations is not None]
        )
        if any(r.vqe_iterations is not None for r in results)
        else None,
    }


def significance_test(
    results_a: list[EvaluationMetrics],
    results_b: list[EvaluationMetrics],
    alpha: float = 0.05,
) -> dict:
    """
    Perform statistical significance test between two sets of results.

    Uses paired t-test if same number of runs, otherwise Welch's t-test.

    Args:
        results_a: Results for circuit A
        results_b: Results for circuit B
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    from scipy import stats

    errors_a = [r.energy_error for r in results_a]
    errors_b = [r.energy_error for r in results_b]

    if len(errors_a) == len(errors_b):
        # Paired t-test (same random seeds assumed)
        t_stat, p_value = stats.ttest_rel(errors_b, errors_a)
        test_type = "paired"
    else:
        # Welch's t-test (independent samples)
        t_stat, p_value = stats.ttest_ind(errors_b, errors_a, equal_var=False)
        test_type = "welch"

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.std(errors_a) ** 2 + np.std(errors_b) ** 2) / 2
    )
    if pooled_std > 0:
        cohens_d = (np.mean(errors_b) - np.mean(errors_a)) / pooled_std
    else:
        cohens_d = 0.0

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"

    return {
        "test_type": test_type,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "cohens_d": float(cohens_d),
        "effect_size": effect_size,
        "alpha": alpha,
        "conclusion": (
            f"Circuit A is significantly better (p={p_value:.4f})"
            if p_value < alpha and np.mean(errors_a) < np.mean(errors_b)
            else f"Circuit B is significantly better (p={p_value:.4f})"
            if p_value < alpha
            else f"No significant difference (p={p_value:.4f})"
        ),
    }


def format_metrics_table(
    metrics: dict[str, EvaluationMetrics],
    exact_energy: float | None = None,
) -> str:
    """
    Format metrics as a comparison table.

    Args:
        metrics: Dictionary mapping circuit names to their metrics
        exact_energy: Optional exact energy for reference

    Returns:
        Formatted table string
    """
    lines = []

    # Header
    names = list(metrics.keys())
    header = "| Metric | " + " | ".join(names) + " |"
    separator = "|--------|" + "|".join(["-" * max(10, len(n) + 2) for n in names]) + "|"

    lines.append(header)
    lines.append(separator)

    if exact_energy is not None:
        lines.append(
            f"| Exact Energy | "
            + " | ".join([f"{exact_energy:.6f}" for _ in names])
            + " |"
        )

    # Data rows
    rows = [
        ("Energy", lambda m: f"{m.energy:.6f}"),
        ("Energy Error", lambda m: f"{m.energy_error:.6f}"),
        ("Relative Error", lambda m: f"{m.relative_error:.2%}"),
        ("Depth", lambda m: f"{m.depth}"),
        ("Gates", lambda m: f"{m.gate_count}"),
        ("Parameters", lambda m: f"{m.param_count}"),
        ("2Q Gates", lambda m: f"{m.two_qubit_count}"),
    ]

    for row_name, formatter in rows:
        values = [formatter(metrics[name]) for name in names]
        lines.append(f"| {row_name} | " + " | ".join(values) + " |")

    return "\n".join(lines)


def compute_improvement_vs_baseline(
    discovered_metrics: EvaluationMetrics,
    baseline_metrics: EvaluationMetrics,
) -> dict:
    """
    Compute improvement of discovered circuit vs baseline.

    Args:
        discovered_metrics: Metrics for discovered circuit
        baseline_metrics: Metrics for baseline

    Returns:
        Dictionary with improvement metrics
    """
    # Energy error reduction
    if baseline_metrics.energy_error > 0:
        error_reduction = (
            (baseline_metrics.energy_error - discovered_metrics.energy_error)
            / baseline_metrics.energy_error
            * 100
        )
    else:
        error_reduction = 0.0

    # Depth reduction
    if baseline_metrics.depth > 0:
        depth_reduction = (
            (baseline_metrics.depth - discovered_metrics.depth)
            / baseline_metrics.depth
            * 100
        )
    else:
        depth_reduction = 0.0

    # Parameter reduction
    if baseline_metrics.param_count > 0:
        param_reduction = (
            (baseline_metrics.param_count - discovered_metrics.param_count)
            / baseline_metrics.param_count
            * 100
        )
    else:
        param_reduction = 0.0

    # Gate reduction
    if baseline_metrics.gate_count > 0:
        gate_reduction = (
            (baseline_metrics.gate_count - discovered_metrics.gate_count)
            / baseline_metrics.gate_count
            * 100
        )
    else:
        gate_reduction = 0.0

    return {
        "error_reduction_percent": error_reduction,
        "depth_reduction_percent": depth_reduction,
        "param_reduction_percent": param_reduction,
        "gate_reduction_percent": gate_reduction,
        "is_better": discovered_metrics.energy_error < baseline_metrics.energy_error,
        "summary": (
            f"Discovered circuit: {error_reduction:.1f}% error reduction, "
            f"{depth_reduction:.1f}% depth reduction"
        ),
    }
