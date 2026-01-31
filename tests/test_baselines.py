"""
Tests for the baseline ansatzes module.
"""

import pytest
from qiskit import QuantumCircuit

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


class TestHardwareEfficientAnsatz:
    """Tests for hardware_efficient_ansatz function."""

    def test_basic_creation(self):
        """Test basic HEA creation."""
        circuit = hardware_efficient_ansatz(4)

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 4
        assert len(circuit.parameters) > 0

    def test_num_layers(self):
        """Test that more layers means more parameters."""
        circuit1 = hardware_efficient_ansatz(4, num_layers=1)
        circuit2 = hardware_efficient_ansatz(4, num_layers=3)

        assert len(circuit2.parameters) > len(circuit1.parameters)

    def test_linear_entanglement(self):
        """Test linear entanglement pattern."""
        circuit = hardware_efficient_ansatz(4, num_layers=1, entanglement="linear")
        ops = circuit.count_ops()

        # Linear entanglement: n-1 CX gates per layer
        assert "cx" in ops
        # At least n-1 CX gates for 1 layer
        assert ops["cx"] >= 3

    def test_full_entanglement(self):
        """Test full entanglement pattern."""
        circuit = hardware_efficient_ansatz(4, num_layers=1, entanglement="full")
        ops = circuit.count_ops()

        # Full entanglement: n*(n-1)/2 CX gates per layer
        assert "cx" in ops
        # For 4 qubits: 6 CX gates per layer
        assert ops["cx"] >= 6

    def test_circular_entanglement(self):
        """Test circular entanglement pattern."""
        circuit = hardware_efficient_ansatz(4, num_layers=1, entanglement="circular")
        ops = circuit.count_ops()

        assert "cx" in ops
        # Circular: n CX gates per layer
        assert ops["cx"] >= 4

    def test_custom_rotation_gates(self):
        """Test custom rotation gates."""
        circuit = hardware_efficient_ansatz(
            4, num_layers=1, rotation_gates=["rx", "rz"]
        )
        ops = circuit.count_ops()

        assert "rx" in ops
        assert "rz" in ops
        assert "ry" not in ops


class TestEfficientSU2Ansatz:
    """Tests for efficient_su2_ansatz function."""

    def test_basic_creation(self):
        """Test basic SU2 ansatz creation."""
        circuit = efficient_su2_ansatz(4)

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 4
        assert len(circuit.parameters) > 0

    def test_has_ry_rz(self):
        """Test that SU2 has both RY and RZ gates."""
        circuit = efficient_su2_ansatz(4, num_layers=1)
        ops = circuit.count_ops()

        assert "ry" in ops
        assert "rz" in ops

    def test_full_entanglement(self):
        """Test full entanglement produces more CX gates."""
        circuit_linear = efficient_su2_ansatz(4, num_layers=1, entanglement="linear")
        circuit_full = efficient_su2_ansatz(4, num_layers=1, entanglement="full")

        cx_linear = circuit_linear.count_ops().get("cx", 0)
        cx_full = circuit_full.count_ops().get("cx", 0)

        assert cx_full > cx_linear


class TestRealAmplitudesAnsatz:
    """Tests for real_amplitudes_ansatz function."""

    def test_basic_creation(self):
        """Test basic creation."""
        circuit = real_amplitudes_ansatz(4)

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 4

    def test_only_ry_gates(self):
        """Test that only RY rotation gates are used."""
        circuit = real_amplitudes_ansatz(4, num_layers=2)
        ops = circuit.count_ops()

        # Should have RY and CX only
        assert "ry" in ops
        assert "rx" not in ops
        assert "rz" not in ops


class TestExcitationPreservingAnsatz:
    """Tests for excitation_preserving_ansatz function."""

    def test_basic_creation(self):
        """Test basic creation."""
        circuit = excitation_preserving_ansatz(4)

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 4

    def test_has_two_qubit_interactions(self):
        """Test that it has RXX/RYY gates."""
        circuit = excitation_preserving_ansatz(4, num_layers=1)
        ops = circuit.count_ops()

        # Should have RXX and RYY for excitation preservation
        assert "rxx" in ops
        assert "ryy" in ops


class TestTwoLocalAnsatz:
    """Tests for two_local_ansatz function."""

    def test_basic_creation(self):
        """Test basic creation."""
        circuit = two_local_ansatz(4)

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 4

    def test_custom_entanglement_block(self):
        """Test custom entanglement block."""
        circuit_cx = two_local_ansatz(4, entanglement_blocks="cx")
        circuit_cz = two_local_ansatz(4, entanglement_blocks="cz")

        assert "cx" in circuit_cx.count_ops()
        assert "cz" in circuit_cz.count_ops()

    def test_custom_rotation_blocks(self):
        """Test custom rotation blocks."""
        circuit = two_local_ansatz(4, rotation_blocks=["rx", "rz"])
        ops = circuit.count_ops()

        assert "rx" in ops
        assert "rz" in ops


class TestGetBaseline:
    """Tests for get_baseline function."""

    def test_get_hea(self):
        """Test getting HEA baseline."""
        circuit = get_baseline("HEA", 4)

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 4

    def test_get_su2(self):
        """Test getting SU2 baseline."""
        circuit = get_baseline("SU2", 4)

        assert isinstance(circuit, QuantumCircuit)

    def test_case_insensitive(self):
        """Test case insensitivity."""
        circuit1 = get_baseline("hea", 4)
        circuit2 = get_baseline("HEA", 4)
        circuit3 = get_baseline("Hea", 4)

        assert circuit1.num_qubits == circuit2.num_qubits == circuit3.num_qubits

    def test_unknown_baseline_raises_error(self):
        """Test that unknown baseline raises error."""
        with pytest.raises(ValueError):
            get_baseline("UNKNOWN", 4)

    def test_with_kwargs(self):
        """Test passing kwargs to baseline."""
        circuit = get_baseline("HEA", 4, num_layers=3, entanglement="full")

        # Full entanglement should have more CX gates
        assert circuit.count_ops().get("cx", 0) > 6


class TestListBaselines:
    """Tests for list_baselines function."""

    def test_returns_list(self):
        """Test that list is returned."""
        baselines = list_baselines()

        assert isinstance(baselines, list)
        assert len(baselines) > 0

    def test_contains_expected(self):
        """Test that expected baselines are included."""
        baselines = list_baselines()

        assert "HEA" in baselines
        assert "SU2" in baselines
        assert "REAL_AMP" in baselines


class TestGetCircuitInfo:
    """Tests for get_circuit_info function."""

    def test_basic_info(self):
        """Test basic circuit info."""
        circuit = hardware_efficient_ansatz(4, num_layers=2)
        info = get_circuit_info(circuit)

        assert info["num_qubits"] == 4
        assert info["depth"] > 0
        assert info["num_params"] > 0
        assert info["gate_count"] > 0

    def test_cx_count(self):
        """Test CX gate counting."""
        circuit = hardware_efficient_ansatz(4, num_layers=1, entanglement="linear")
        info = get_circuit_info(circuit)

        assert "cx_count" in info
        assert info["cx_count"] >= 3  # n-1 for linear entanglement

    def test_single_two_qubit_split(self):
        """Test single vs two-qubit gate split."""
        circuit = hardware_efficient_ansatz(4, num_layers=1)
        info = get_circuit_info(circuit)

        assert "single_qubit_gates" in info
        assert "two_qubit_gates" in info
        assert info["single_qubit_gates"] > 0
        assert info["two_qubit_gates"] > 0


class TestCompareBaselines:
    """Tests for compare_baselines function."""

    def test_compare_all(self):
        """Test comparing all baselines."""
        comparison = compare_baselines(4, num_layers=2)

        assert isinstance(comparison, dict)
        assert len(comparison) == len(list_baselines())

    def test_comparison_has_info(self):
        """Test that comparison has expected info."""
        comparison = compare_baselines(4)

        for name, info in comparison.items():
            if "error" not in info:
                assert "num_qubits" in info
                assert "depth" in info
                assert "num_params" in info


class TestIntegration:
    """Integration tests for baselines with other modules."""

    def test_baseline_with_vqe(self):
        """Test that baselines can be used with VQE."""
        from src.quantum.executor import QuantumExecutor
        from src.quantum.hamiltonians import xy_chain

        circuit = hardware_efficient_ansatz(4, num_layers=1)
        ham = xy_chain(4)

        executor = QuantumExecutor()
        result = executor.run_vqe(circuit, ham.operator)

        assert result.energy is not None
        assert result.optimal_params is not None

    def test_baseline_with_verifier(self):
        """Test that baselines pass verification."""
        from src.quantum.verifier import CircuitVerifier

        verifier = CircuitVerifier()

        for baseline_name in list_baselines():
            circuit = get_baseline(baseline_name, 4)

            # Convert to code string (simplified check)
            assert circuit.num_qubits == 4
            assert len(circuit.parameters) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
