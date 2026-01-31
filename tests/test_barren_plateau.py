"""
Tests for the Barren Plateau detector module.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from src.quantum.barren_plateau import (
    BarrenPlateauResult,
    analyze_circuit_structure,
    compute_gradient_parameter_shift,
    detect_barren_plateau,
    estimate_trainability_score,
    quick_bp_heuristic,
)
from src.quantum.hamiltonians import xy_chain


class TestComputeGradient:
    """Tests for gradient computation."""

    def test_gradient_computation_basic(self):
        """Test basic gradient computation."""
        qc = QuantumCircuit(2)
        theta = Parameter("theta")
        qc.ry(theta, 0)
        qc.cx(0, 1)

        ham = xy_chain(2).operator
        param_dict = {theta: np.pi / 4}

        grad = compute_gradient_parameter_shift(qc, ham, param_dict, param_index=0)

        # Gradient should be finite and non-zero for this circuit
        assert np.isfinite(grad)

    def test_gradient_zero_at_extremum(self):
        """Gradient should be near zero at optimal point for simple cases."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.ry(theta, 0)

        # For measuring Z on |+> state created by RY(pi/2)
        from qiskit.quantum_info import SparsePauliOp

        ham = SparsePauliOp.from_list([("Z", 1.0)])

        # At theta=0, the gradient of <Z> should be non-zero
        # At theta=pi (|1> state), measuring Z gives -1, gradient should be ~0
        param_dict = {theta: np.pi}
        grad = compute_gradient_parameter_shift(qc, ham, param_dict, param_index=0)

        # At theta=pi, we're at an extremum
        assert abs(grad) < 0.1

    def test_invalid_param_index(self):
        """Should raise error for invalid parameter index."""
        qc = QuantumCircuit(2)
        theta = Parameter("theta")
        qc.ry(theta, 0)

        ham = xy_chain(2).operator
        param_dict = {theta: 0.5}

        with pytest.raises(ValueError):
            compute_gradient_parameter_shift(qc, ham, param_dict, param_index=5)


class TestDetectBarrenPlateau:
    """Tests for full barren plateau detection."""

    def test_shallow_circuit_is_trainable(self):
        """Shallow hardware-efficient circuit should be trainable."""
        qc = QuantumCircuit(4)
        params = [Parameter(f"p{i}") for i in range(4)]

        for i in range(4):
            qc.ry(params[i], i)
        for i in range(3):
            qc.cx(i, i + 1)

        ham = xy_chain(4).operator
        result = detect_barren_plateau(qc, ham, num_samples=30, seed=42)

        assert isinstance(result, BarrenPlateauResult)
        assert result.trainability in ["high", "medium"]
        assert not result.has_barren_plateau
        assert result.gradient_variance > 1e-4

    def test_no_parameters_detected(self):
        """Circuit without parameters should be flagged."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        ham = xy_chain(2).operator
        result = detect_barren_plateau(qc, ham)

        assert result.has_barren_plateau
        assert result.trainability == "unknown"
        assert "no trainable parameters" in result.recommendation.lower()

    def test_result_contains_details(self):
        """Result should contain detailed information."""
        qc = QuantumCircuit(2)
        params = [Parameter(f"p{i}") for i in range(2)]
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)

        ham = xy_chain(2).operator
        result = detect_barren_plateau(qc, ham, num_samples=20, seed=42)

        assert "num_samples" in result.details
        assert "num_params" in result.details
        assert "circuit_depth" in result.details

    def test_reproducibility_with_seed(self):
        """Same seed should give same result."""
        qc = QuantumCircuit(3)
        params = [Parameter(f"p{i}") for i in range(3)]
        for i in range(3):
            qc.ry(params[i], i)
        for i in range(2):
            qc.cx(i, i + 1)

        ham = xy_chain(3).operator

        result1 = detect_barren_plateau(qc, ham, num_samples=20, seed=123)
        result2 = detect_barren_plateau(qc, ham, num_samples=20, seed=123)

        assert result1.gradient_variance == result2.gradient_variance


class TestQuickHeuristic:
    """Tests for quick heuristic BP detection."""

    def test_deep_circuit_flagged(self):
        """Very deep circuit should be flagged."""
        qc = QuantumCircuit(4)
        params = [Parameter(f"p{i}") for i in range(40)]

        # Create deep circuit (depth >> 2*n)
        idx = 0
        for _ in range(10):
            for i in range(4):
                qc.ry(params[idx], i)
                idx += 1

        likely_bp, reason, _ = quick_bp_heuristic(qc)

        assert likely_bp
        assert "depth" in reason.lower()

    def test_no_parameters_flagged(self):
        """Circuit without parameters should be flagged."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        likely_bp, reason, _ = quick_bp_heuristic(qc)

        assert likely_bp
        assert "no trainable parameters" in reason.lower()

    def test_shallow_circuit_passes(self):
        """Shallow well-designed circuit should pass."""
        qc = QuantumCircuit(4)
        params = [Parameter(f"p{i}") for i in range(8)]

        # Single layer of rotations + entanglement
        for i in range(4):
            qc.ry(params[i], i)
        for i in range(3):
            qc.cx(i, i + 1)
        for i in range(4):
            qc.ry(params[i + 4], i)

        likely_bp, reason, details = quick_bp_heuristic(qc)

        assert not likely_bp
        assert "passed" in reason.lower()

    def test_no_entanglement_flagged(self):
        """Multi-qubit circuit without entanglement should be flagged."""
        qc = QuantumCircuit(4)
        params = [Parameter(f"p{i}") for i in range(4)]

        for i in range(4):
            qc.ry(params[i], i)

        likely_bp, reason, _ = quick_bp_heuristic(qc)

        assert likely_bp
        assert "entanglement" in reason.lower()


class TestAnalyzeCircuitStructure:
    """Tests for circuit structure analysis."""

    def test_basic_analysis(self):
        """Test basic structure analysis."""
        qc = QuantumCircuit(4)
        params = [Parameter(f"p{i}") for i in range(8)]

        for i in range(4):
            qc.ry(params[i], i)
        for i in range(3):
            qc.cx(i, i + 1)
        for i in range(4):
            qc.rz(params[i + 4], i)

        analysis = analyze_circuit_structure(qc)

        assert analysis["num_qubits"] == 4
        assert analysis["num_params"] == 8
        assert analysis["two_qubit_gates"] == 3
        assert analysis["single_qubit_gates"] == 8
        assert 0 <= analysis["risk_score"] <= 1

    def test_risk_factors_detected(self):
        """Deep circuit should have risk factors."""
        qc = QuantumCircuit(2)
        params = [Parameter(f"p{i}") for i in range(20)]

        # Create unnecessarily deep circuit
        for i, p in enumerate(params):
            qc.ry(p, i % 2)
            if i % 2 == 1:
                qc.cx(0, 1)

        analysis = analyze_circuit_structure(qc)

        assert "deep_circuit" in analysis["risk_factors"] or analysis["depth_per_qubit"] > 3
        assert analysis["risk_score"] > 0

    def test_empty_circuit(self):
        """Empty circuit should be handled."""
        qc = QuantumCircuit(2)

        analysis = analyze_circuit_structure(qc)

        assert analysis["num_params"] == 0
        assert analysis["total_gates"] == 0


class TestTrainabilityScore:
    """Tests for trainability score estimation."""

    def test_score_range(self):
        """Score should be between 0 and 1."""
        qc = QuantumCircuit(3)
        params = [Parameter(f"p{i}") for i in range(3)]
        for i in range(3):
            qc.ry(params[i], i)
        for i in range(2):
            qc.cx(i, i + 1)

        score = estimate_trainability_score(qc)

        assert 0 <= score <= 1

    def test_good_circuit_high_score(self):
        """Well-designed circuit should have high score."""
        qc = QuantumCircuit(4)
        params = [Parameter(f"p{i}") for i in range(8)]

        for i in range(4):
            qc.ry(params[i], i)
        for i in range(3):
            qc.cx(i, i + 1)
        for i in range(4):
            qc.ry(params[i + 4], i)

        score = estimate_trainability_score(qc)

        assert score > 0.5

    def test_score_with_hamiltonian(self):
        """Score computation with Hamiltonian for gradient analysis."""
        qc = QuantumCircuit(3)
        params = [Parameter(f"p{i}") for i in range(6)]

        for i in range(3):
            qc.ry(params[i], i)
        for i in range(2):
            qc.cx(i, i + 1)
        for i in range(3):
            qc.ry(params[i + 3], i)

        ham = xy_chain(3).operator
        score = estimate_trainability_score(qc, hamiltonian=ham, num_samples=15, seed=42)

        assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
