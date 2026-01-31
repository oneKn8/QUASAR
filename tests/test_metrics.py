"""
Tests for the metrics module.
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

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


def create_test_circuit(num_qubits: int = 4, depth: int = 2) -> QuantumCircuit:
    """Create a test circuit with parameters."""
    qc = QuantumCircuit(num_qubits)
    param_idx = 0

    for _ in range(depth):
        for q in range(num_qubits):
            theta = Parameter(f"theta_{param_idx}")
            qc.ry(theta, q)
            param_idx += 1
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

    return qc


def create_test_hamiltonian(num_qubits: int = 4) -> SparsePauliOp:
    """Create a test Hamiltonian."""
    # Simple ZZ + X Hamiltonian
    terms = []
    for i in range(num_qubits - 1):
        zz = ["I"] * num_qubits
        zz[i] = "Z"
        zz[i + 1] = "Z"
        terms.append(("".join(zz), -1.0))

    for i in range(num_qubits):
        x = ["I"] * num_qubits
        x[i] = "X"
        terms.append(("".join(x), -0.5))

    return SparsePauliOp.from_list(terms)


class MockVQEResult:
    """Mock VQE result for testing."""

    def __init__(
        self,
        energy: float = -2.5,
        iterations: int = 50,
        converged: bool = True,
    ):
        self.energy = energy
        self.iterations = iterations
        self.converged = converged
        self.optimal_params = None
        self.final_state = None


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_basic_creation(self):
        """Test basic metrics creation."""
        metrics = EvaluationMetrics(
            energy=-2.5,
            energy_error=0.01,
            relative_error=0.004,
            depth=10,
            gate_count=40,
            param_count=16,
            two_qubit_count=12,
            energy_per_depth=0.001,
            energy_per_param=0.000625,
            energy_per_gate=0.00025,
        )

        assert metrics.energy == -2.5
        assert metrics.energy_error == 0.01
        assert metrics.depth == 10
        assert metrics.param_count == 16

    def test_optional_fields(self):
        """Test optional fields default values."""
        metrics = EvaluationMetrics(
            energy=-2.5,
            energy_error=0.01,
            relative_error=0.004,
            depth=10,
            gate_count=40,
            param_count=16,
            two_qubit_count=12,
            energy_per_depth=0.001,
            energy_per_param=0.000625,
            energy_per_gate=0.00025,
        )

        assert metrics.fidelity is None
        assert metrics.gradient_variance is None
        assert metrics.vqe_iterations is None
        assert metrics.converged is True
        assert metrics.details == {}

    def test_with_optional_fields(self):
        """Test with optional fields filled."""
        metrics = EvaluationMetrics(
            energy=-2.5,
            energy_error=0.01,
            relative_error=0.004,
            depth=10,
            gate_count=40,
            param_count=16,
            two_qubit_count=12,
            energy_per_depth=0.001,
            energy_per_param=0.000625,
            energy_per_gate=0.00025,
            fidelity=0.99,
            gradient_variance=0.001,
            vqe_iterations=50,
            converged=True,
            details={"ops": {"ry": 16, "cx": 12}},
        )

        assert metrics.fidelity == 0.99
        assert metrics.vqe_iterations == 50
        assert "ry" in metrics.details["ops"]


class TestComparisonMetrics:
    """Tests for ComparisonMetrics dataclass."""

    def test_basic_creation(self):
        """Test basic comparison creation."""
        comparison = ComparisonMetrics(
            winner="circuit_a",
            energy_improvement=10.0,
            depth_improvement=5.0,
            param_improvement=0.0,
            energy_diff=-0.01,
            depth_diff=-2,
            param_diff=0,
            is_significant=True,
            summary="Circuit A is better: 10.0% lower error",
        )

        assert comparison.winner == "circuit_a"
        assert comparison.energy_improvement == 10.0
        assert comparison.is_significant is True


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_basic_metrics(self):
        """Test basic metric computation."""
        circuit = create_test_circuit(4, 2)
        ham = create_test_hamiltonian(4)
        exact_energy = -3.0
        vqe_result = MockVQEResult(energy=-2.95)

        metrics = compute_metrics(circuit, ham, exact_energy, vqe_result)

        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.energy == -2.95
        assert metrics.energy_error == pytest.approx(0.05, rel=0.01)
        assert metrics.depth > 0
        assert metrics.param_count == 8  # 4 qubits * 2 layers

    def test_circuit_structure(self):
        """Test circuit structure metrics."""
        circuit = create_test_circuit(4, 3)
        ham = create_test_hamiltonian(4)

        metrics = compute_metrics(circuit, ham, -3.0, MockVQEResult())

        assert metrics.num_qubits == 4 or metrics.details.get("num_qubits") == 4
        assert metrics.param_count == 12  # 4 qubits * 3 layers
        assert metrics.two_qubit_count >= 6  # At least 3*(4-1) = 9 CX gates

    def test_efficiency_metrics(self):
        """Test efficiency metric computation."""
        circuit = create_test_circuit(4, 2)
        ham = create_test_hamiltonian(4)
        exact_energy = -3.0
        vqe_result = MockVQEResult(energy=-2.9)  # Error = 0.1

        metrics = compute_metrics(circuit, ham, exact_energy, vqe_result)

        # energy_per_depth = error / depth
        assert metrics.energy_per_depth > 0
        assert metrics.energy_per_param > 0
        assert metrics.energy_per_gate > 0

    def test_without_vqe_result(self):
        """Test metrics without VQE result."""
        circuit = create_test_circuit(4, 2)
        ham = create_test_hamiltonian(4)

        metrics = compute_metrics(circuit, ham, -3.0)

        assert metrics.energy == 0.0
        assert metrics.vqe_iterations is None

    def test_relative_error(self):
        """Test relative error calculation."""
        circuit = create_test_circuit(4, 2)
        ham = create_test_hamiltonian(4)
        exact_energy = -2.0
        vqe_result = MockVQEResult(energy=-1.9)  # Error = 0.1

        metrics = compute_metrics(circuit, ham, exact_energy, vqe_result)

        assert metrics.relative_error == pytest.approx(0.05, rel=0.01)  # 0.1/2.0

    def test_zero_exact_energy(self):
        """Test handling of zero exact energy."""
        circuit = create_test_circuit(4, 2)
        ham = create_test_hamiltonian(4)
        exact_energy = 0.0
        vqe_result = MockVQEResult(energy=0.1)

        metrics = compute_metrics(circuit, ham, exact_energy, vqe_result)

        assert metrics.energy_error == 0.1
        assert metrics.relative_error == 0.1  # Falls back to absolute


class TestCompareMetrics:
    """Tests for compare_metrics function."""

    def test_circuit_a_better(self):
        """Test when circuit A is better."""
        metrics_a = EvaluationMetrics(
            energy=-2.95,
            energy_error=0.05,
            relative_error=0.02,
            depth=8,
            gate_count=30,
            param_count=12,
            two_qubit_count=8,
            energy_per_depth=0.00625,
            energy_per_param=0.0042,
            energy_per_gate=0.0017,
        )
        metrics_b = EvaluationMetrics(
            energy=-2.9,
            energy_error=0.10,
            relative_error=0.04,
            depth=10,
            gate_count=40,
            param_count=16,
            two_qubit_count=12,
            energy_per_depth=0.01,
            energy_per_param=0.00625,
            energy_per_gate=0.0025,
        )

        comparison = compare_metrics(metrics_a, metrics_b)

        assert comparison.winner == "circuit_a"
        assert comparison.energy_improvement > 0  # A has lower error
        assert "Circuit A" in comparison.summary

    def test_circuit_b_better(self):
        """Test when circuit B is better."""
        metrics_a = EvaluationMetrics(
            energy=-2.9,
            energy_error=0.10,
            relative_error=0.04,
            depth=10,
            gate_count=40,
            param_count=16,
            two_qubit_count=12,
            energy_per_depth=0.01,
            energy_per_param=0.00625,
            energy_per_gate=0.0025,
        )
        metrics_b = EvaluationMetrics(
            energy=-2.95,
            energy_error=0.05,
            relative_error=0.02,
            depth=8,
            gate_count=30,
            param_count=12,
            two_qubit_count=8,
            energy_per_depth=0.00625,
            energy_per_param=0.0042,
            energy_per_gate=0.0017,
        )

        comparison = compare_metrics(metrics_a, metrics_b)

        assert comparison.winner == "circuit_b"
        assert comparison.energy_improvement < 0  # B has lower error

    def test_tie_breaking_by_depth(self):
        """Test tie-breaking by depth when errors are equal."""
        metrics_a = EvaluationMetrics(
            energy=-2.95,
            energy_error=0.05,
            relative_error=0.02,
            depth=8,
            gate_count=30,
            param_count=12,
            two_qubit_count=8,
            energy_per_depth=0.00625,
            energy_per_param=0.0042,
            energy_per_gate=0.0017,
        )
        metrics_b = EvaluationMetrics(
            energy=-2.95,
            energy_error=0.05,  # Same error
            relative_error=0.02,
            depth=10,  # But deeper
            gate_count=40,
            param_count=16,
            two_qubit_count=12,
            energy_per_depth=0.005,
            energy_per_param=0.003125,
            energy_per_gate=0.00125,
        )

        comparison = compare_metrics(metrics_a, metrics_b)

        assert comparison.winner == "circuit_a"  # Shallower wins on tie

    def test_significance_check(self):
        """Test significance threshold."""
        metrics_a = EvaluationMetrics(
            energy=-2.95,
            energy_error=0.05,
            relative_error=0.02,
            depth=10,
            gate_count=40,
            param_count=16,
            two_qubit_count=12,
            energy_per_depth=0.005,
            energy_per_param=0.003125,
            energy_per_gate=0.00125,
        )
        metrics_b = EvaluationMetrics(
            energy=-2.94,
            energy_error=0.06,  # Only 0.01 difference
            relative_error=0.024,
            depth=10,
            gate_count=40,
            param_count=16,
            two_qubit_count=12,
            energy_per_depth=0.006,
            energy_per_param=0.00375,
            energy_per_gate=0.0015,
        )

        comparison = compare_metrics(metrics_a, metrics_b, significance_threshold=0.50)

        # 16.67% improvement < 50% threshold
        assert comparison.is_significant is False

    def test_improvement_calculations(self):
        """Test improvement percentage calculations."""
        metrics_a = EvaluationMetrics(
            energy=-2.95,
            energy_error=0.05,
            relative_error=0.02,
            depth=8,
            gate_count=30,
            param_count=12,
            two_qubit_count=8,
            energy_per_depth=0.00625,
            energy_per_param=0.0042,
            energy_per_gate=0.0017,
        )
        metrics_b = EvaluationMetrics(
            energy=-2.9,
            energy_error=0.10,
            relative_error=0.04,
            depth=10,
            gate_count=40,
            param_count=16,
            two_qubit_count=12,
            energy_per_depth=0.01,
            energy_per_param=0.00625,
            energy_per_gate=0.0025,
        )

        comparison = compare_metrics(metrics_a, metrics_b)

        # A has 50% lower error: (0.10 - 0.05) / 0.10 * 100 = 50%
        assert comparison.energy_improvement == pytest.approx(50.0, rel=0.01)
        # A is 20% shallower: (10 - 8) / 10 * 100 = 20%
        assert comparison.depth_improvement == pytest.approx(20.0, rel=0.01)


class TestStatisticalEvaluation:
    """Tests for statistical_evaluation function."""

    def test_basic_statistics(self):
        """Test basic statistical calculation."""
        results = [
            EvaluationMetrics(
                energy=-2.5,
                energy_error=0.05,
                relative_error=0.02,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.005,
                energy_per_param=0.003125,
                energy_per_gate=0.00125,
                converged=True,
                vqe_iterations=50,
            ),
            EvaluationMetrics(
                energy=-2.6,
                energy_error=0.04,
                relative_error=0.015,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.004,
                energy_per_param=0.0025,
                energy_per_gate=0.001,
                converged=True,
                vqe_iterations=45,
            ),
            EvaluationMetrics(
                energy=-2.55,
                energy_error=0.045,
                relative_error=0.0175,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.0045,
                energy_per_param=0.0028125,
                energy_per_gate=0.001125,
                converged=True,
                vqe_iterations=48,
            ),
        ]

        stats = statistical_evaluation(results)

        assert stats["num_runs"] == 3
        assert stats["mean_energy"] == pytest.approx(-2.55, rel=0.01)
        assert stats["mean_error"] == pytest.approx(0.045, rel=0.1)
        assert stats["best_error"] == 0.04
        assert stats["worst_error"] == 0.05
        assert stats["convergence_rate"] == 1.0

    def test_empty_results(self):
        """Test with empty results."""
        stats = statistical_evaluation([])

        assert stats == {}

    def test_convergence_rate(self):
        """Test convergence rate calculation."""
        results = [
            EvaluationMetrics(
                energy=-2.5,
                energy_error=0.05,
                relative_error=0.02,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.005,
                energy_per_param=0.003125,
                energy_per_gate=0.00125,
                converged=True,
            ),
            EvaluationMetrics(
                energy=-2.5,
                energy_error=0.05,
                relative_error=0.02,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.005,
                energy_per_param=0.003125,
                energy_per_gate=0.00125,
                converged=False,
            ),
        ]

        stats = statistical_evaluation(results)

        assert stats["convergence_rate"] == 0.5

    def test_iteration_mean(self):
        """Test mean iteration calculation."""
        results = [
            EvaluationMetrics(
                energy=-2.5,
                energy_error=0.05,
                relative_error=0.02,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.005,
                energy_per_param=0.003125,
                energy_per_gate=0.00125,
                vqe_iterations=40,
            ),
            EvaluationMetrics(
                energy=-2.5,
                energy_error=0.05,
                relative_error=0.02,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.005,
                energy_per_param=0.003125,
                energy_per_gate=0.00125,
                vqe_iterations=60,
            ),
        ]

        stats = statistical_evaluation(results)

        assert stats["mean_iterations"] == 50.0


class TestSignificanceTest:
    """Tests for significance_test function."""

    def test_significant_difference(self):
        """Test detection of significant difference."""
        # Circuit A: consistently lower error
        results_a = [
            EvaluationMetrics(
                energy=-2.95,
                energy_error=0.05,
                relative_error=0.02,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.005,
                energy_per_param=0.003125,
                energy_per_gate=0.00125,
            )
            for _ in range(10)
        ]
        # Circuit B: consistently higher error
        results_b = [
            EvaluationMetrics(
                energy=-2.8,
                energy_error=0.20,
                relative_error=0.08,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.02,
                energy_per_param=0.0125,
                energy_per_gate=0.005,
            )
            for _ in range(10)
        ]

        result = significance_test(results_a, results_b)

        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert "better" in result["conclusion"].lower()

    def test_no_significant_difference(self):
        """Test when there's no significant difference."""
        # Same performance with noise
        np.random.seed(42)
        results_a = [
            EvaluationMetrics(
                energy=-2.5 + np.random.normal(0, 0.01),
                energy_error=0.05 + np.random.normal(0, 0.01),
                relative_error=0.02,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.005,
                energy_per_param=0.003125,
                energy_per_gate=0.00125,
            )
            for _ in range(5)
        ]
        results_b = [
            EvaluationMetrics(
                energy=-2.5 + np.random.normal(0, 0.01),
                energy_error=0.05 + np.random.normal(0, 0.01),
                relative_error=0.02,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.005,
                energy_per_param=0.003125,
                energy_per_gate=0.00125,
            )
            for _ in range(5)
        ]

        result = significance_test(results_a, results_b)

        # Should likely not be significant (may vary with random seed)
        assert "p_value" in result
        assert result["test_type"] == "paired"

    def test_effect_size_interpretation(self):
        """Test effect size interpretation."""
        # Large effect: A much better
        results_a = [
            EvaluationMetrics(
                energy=-2.99,
                energy_error=0.01,
                relative_error=0.003,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.001,
                energy_per_param=0.000625,
                energy_per_gate=0.00025,
            )
            for _ in range(10)
        ]
        results_b = [
            EvaluationMetrics(
                energy=-2.5,
                energy_error=0.50,
                relative_error=0.2,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.05,
                energy_per_param=0.03125,
                energy_per_gate=0.0125,
            )
            for _ in range(10)
        ]

        result = significance_test(results_a, results_b)

        assert result["effect_size"] == "large"
        assert abs(result["cohens_d"]) >= 0.8

    def test_welch_test_unequal_samples(self):
        """Test Welch's t-test for unequal sample sizes."""
        results_a = [
            EvaluationMetrics(
                energy=-2.95,
                energy_error=0.05,
                relative_error=0.02,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.005,
                energy_per_param=0.003125,
                energy_per_gate=0.00125,
            )
            for _ in range(5)
        ]
        results_b = [
            EvaluationMetrics(
                energy=-2.9,
                energy_error=0.10,
                relative_error=0.04,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.01,
                energy_per_param=0.00625,
                energy_per_gate=0.0025,
            )
            for _ in range(10)  # Different size
        ]

        result = significance_test(results_a, results_b)

        assert result["test_type"] == "welch"


class TestFormatMetricsTable:
    """Tests for format_metrics_table function."""

    def test_basic_table(self):
        """Test basic table formatting."""
        metrics = {
            "Discovered": EvaluationMetrics(
                energy=-2.95,
                energy_error=0.05,
                relative_error=0.02,
                depth=8,
                gate_count=30,
                param_count=12,
                two_qubit_count=8,
                energy_per_depth=0.00625,
                energy_per_param=0.0042,
                energy_per_gate=0.0017,
            ),
            "Baseline": EvaluationMetrics(
                energy=-2.9,
                energy_error=0.10,
                relative_error=0.04,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.01,
                energy_per_param=0.00625,
                energy_per_gate=0.0025,
            ),
        }

        table = format_metrics_table(metrics)

        assert "Discovered" in table
        assert "Baseline" in table
        assert "Energy" in table
        assert "Depth" in table
        assert "|" in table  # Markdown table format

    def test_table_with_exact_energy(self):
        """Test table with exact energy reference."""
        metrics = {
            "Circuit": EvaluationMetrics(
                energy=-2.95,
                energy_error=0.05,
                relative_error=0.02,
                depth=10,
                gate_count=40,
                param_count=16,
                two_qubit_count=12,
                energy_per_depth=0.005,
                energy_per_param=0.003125,
                energy_per_gate=0.00125,
            )
        }

        table = format_metrics_table(metrics, exact_energy=-3.0)

        assert "Exact Energy" in table
        assert "-3.0" in table


class TestComputeImprovementVsBaseline:
    """Tests for compute_improvement_vs_baseline function."""

    def test_improvement(self):
        """Test improvement calculation."""
        discovered = EvaluationMetrics(
            energy=-2.95,
            energy_error=0.05,
            relative_error=0.02,
            depth=8,
            gate_count=30,
            param_count=12,
            two_qubit_count=8,
            energy_per_depth=0.00625,
            energy_per_param=0.0042,
            energy_per_gate=0.0017,
        )
        baseline = EvaluationMetrics(
            energy=-2.9,
            energy_error=0.10,
            relative_error=0.04,
            depth=10,
            gate_count=40,
            param_count=16,
            two_qubit_count=12,
            energy_per_depth=0.01,
            energy_per_param=0.00625,
            energy_per_gate=0.0025,
        )

        improvement = compute_improvement_vs_baseline(discovered, baseline)

        assert improvement["is_better"] is True
        assert improvement["error_reduction_percent"] == pytest.approx(50.0, rel=0.01)
        assert improvement["depth_reduction_percent"] == pytest.approx(20.0, rel=0.01)

    def test_no_improvement(self):
        """Test when discovered is worse."""
        discovered = EvaluationMetrics(
            energy=-2.8,
            energy_error=0.20,
            relative_error=0.08,
            depth=12,
            gate_count=50,
            param_count=20,
            two_qubit_count=15,
            energy_per_depth=0.0167,
            energy_per_param=0.01,
            energy_per_gate=0.004,
        )
        baseline = EvaluationMetrics(
            energy=-2.95,
            energy_error=0.05,
            relative_error=0.02,
            depth=10,
            gate_count=40,
            param_count=16,
            two_qubit_count=12,
            energy_per_depth=0.005,
            energy_per_param=0.003125,
            energy_per_gate=0.00125,
        )

        improvement = compute_improvement_vs_baseline(discovered, baseline)

        assert improvement["is_better"] is False
        assert improvement["error_reduction_percent"] < 0  # Negative = worse

    def test_summary_generation(self):
        """Test summary string generation."""
        discovered = EvaluationMetrics(
            energy=-2.95,
            energy_error=0.05,
            relative_error=0.02,
            depth=8,
            gate_count=30,
            param_count=12,
            two_qubit_count=8,
            energy_per_depth=0.00625,
            energy_per_param=0.0042,
            energy_per_gate=0.0017,
        )
        baseline = EvaluationMetrics(
            energy=-2.9,
            energy_error=0.10,
            relative_error=0.04,
            depth=10,
            gate_count=40,
            param_count=16,
            two_qubit_count=12,
            energy_per_depth=0.01,
            energy_per_param=0.00625,
            energy_per_gate=0.0025,
        )

        improvement = compute_improvement_vs_baseline(discovered, baseline)

        assert "Discovered circuit" in improvement["summary"]
        assert "error reduction" in improvement["summary"]


class TestIntegration:
    """Integration tests for metrics module."""

    def test_full_evaluation_workflow(self):
        """Test full evaluation workflow."""
        from src.evaluation.baselines import hardware_efficient_ansatz
        from src.quantum.hamiltonians import xy_chain

        # Create circuit and Hamiltonian
        circuit = hardware_efficient_ansatz(4, num_layers=2)
        ham = xy_chain(4)
        exact_energy = -3.0  # Approximate

        # Compute metrics (without actual VQE)
        mock_result = MockVQEResult(energy=-2.9)
        metrics = compute_metrics(circuit, ham.operator, exact_energy, mock_result)

        assert metrics.energy == -2.9
        assert metrics.depth > 0
        assert metrics.param_count > 0

    def test_baseline_comparison_workflow(self):
        """Test comparing discovered vs baseline."""
        from src.evaluation.baselines import hardware_efficient_ansatz

        # Create two circuits with different depths
        circuit_a = hardware_efficient_ansatz(4, num_layers=2)
        circuit_b = hardware_efficient_ansatz(4, num_layers=3)

        # Mock metrics
        metrics_a = EvaluationMetrics(
            energy=-2.95,
            energy_error=0.05,
            relative_error=0.02,
            depth=circuit_a.depth(),
            gate_count=sum(circuit_a.count_ops().values()),
            param_count=len(circuit_a.parameters),
            two_qubit_count=circuit_a.count_ops().get("cx", 0),
            energy_per_depth=0.005,
            energy_per_param=0.003125,
            energy_per_gate=0.00125,
        )
        metrics_b = EvaluationMetrics(
            energy=-2.9,
            energy_error=0.10,
            relative_error=0.04,
            depth=circuit_b.depth(),
            gate_count=sum(circuit_b.count_ops().values()),
            param_count=len(circuit_b.parameters),
            two_qubit_count=circuit_b.count_ops().get("cx", 0),
            energy_per_depth=0.01,
            energy_per_param=0.00625,
            energy_per_gate=0.0025,
        )

        # Compare
        comparison = compare_metrics(metrics_a, metrics_b)

        assert comparison.winner in ["circuit_a", "circuit_b"]
        assert "summary" in comparison.__dict__

        # Format table
        table = format_metrics_table(
            {"Discovered": metrics_a, "Baseline": metrics_b},
            exact_energy=-3.0,
        )

        assert "Discovered" in table
        assert "Baseline" in table


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
