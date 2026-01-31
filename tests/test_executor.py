"""
Tests for the Quantum executor module.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from src.quantum.executor import (
    ExecutorConfig,
    MultiStartVQE,
    QuantumExecutor,
    VQEResult,
    estimate_gradient,
    parameter_sweep,
    run_vqe_simple,
)
from src.quantum.hamiltonians import xy_chain, transverse_ising


def create_simple_ansatz(num_qubits: int, num_layers: int = 1) -> QuantumCircuit:
    """Create a simple hardware-efficient ansatz for testing."""
    qc = QuantumCircuit(num_qubits)
    param_count = 0

    for _ in range(num_layers):
        for i in range(num_qubits):
            p = Parameter(f"theta_{param_count}")
            qc.ry(p, i)
            param_count += 1

        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

    # Final rotation layer
    for i in range(num_qubits):
        p = Parameter(f"theta_{param_count}")
        qc.ry(p, i)
        param_count += 1

    return qc


class TestQuantumExecutor:
    """Tests for QuantumExecutor class."""

    def test_basic_vqe(self):
        """Test basic VQE optimization."""
        circuit = create_simple_ansatz(2, num_layers=1)
        ham = xy_chain(2).operator

        executor = QuantumExecutor()
        result = executor.run_vqe(circuit, ham)

        assert isinstance(result, VQEResult)
        assert result.energy is not None
        assert result.optimal_params is not None
        assert len(result.optimal_params) == len(circuit.parameters)

    def test_vqe_improves_energy(self):
        """VQE should find lower energy than random initial."""
        circuit = create_simple_ansatz(2, num_layers=1)
        ham = xy_chain(2).operator

        # Random initial energy
        np.random.seed(42)
        random_params = np.random.uniform(0, 2 * np.pi, len(circuit.parameters))
        executor = QuantumExecutor()
        initial_energy = executor.compute_expectation(circuit, ham, random_params)

        # VQE result
        result = executor.run_vqe(circuit, ham)

        assert result.energy <= initial_energy

    def test_vqe_approaches_exact(self):
        """VQE should approach exact ground state for simple problems."""
        circuit = create_simple_ansatz(2, num_layers=2)
        ham_result = xy_chain(2)

        config = ExecutorConfig(max_iterations=300, seed=42)
        executor = QuantumExecutor(config)
        result = executor.run_vqe(circuit, ham_result.operator)

        # Should be within 10% of exact
        error = abs(result.energy - ham_result.exact_energy)
        relative_error = error / abs(ham_result.exact_energy)

        assert relative_error < 0.1, f"Error {relative_error:.2%} too large"

    def test_convergence_tracking(self):
        """Should track convergence history when enabled."""
        circuit = create_simple_ansatz(2)
        ham = xy_chain(2).operator

        config = ExecutorConfig(track_convergence=True, max_iterations=50)
        executor = QuantumExecutor(config)
        result = executor.run_vqe(circuit, ham)

        assert len(result.convergence_history) > 0
        # Energy should generally decrease
        assert result.convergence_history[-1] <= result.convergence_history[0] + 0.1

    def test_final_state_computation(self):
        """Should compute final state when enabled."""
        circuit = create_simple_ansatz(2)
        ham = xy_chain(2).operator

        config = ExecutorConfig(compute_final_state=True)
        executor = QuantumExecutor(config)
        result = executor.run_vqe(circuit, ham)

        assert result.final_state is not None
        # State should be normalized
        norm = np.linalg.norm(result.final_state.data)
        assert np.isclose(norm, 1.0)

    def test_no_parameters_raises_error(self):
        """Circuit without parameters should raise error."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        ham = xy_chain(2).operator
        executor = QuantumExecutor()

        with pytest.raises(ValueError):
            executor.run_vqe(circuit, ham)

    def test_different_optimizers(self):
        """Test different classical optimizers."""
        circuit = create_simple_ansatz(2)
        ham = xy_chain(2).operator

        for optimizer in ["COBYLA", "Nelder-Mead", "SLSQP"]:
            config = ExecutorConfig(optimizer=optimizer, max_iterations=100)
            executor = QuantumExecutor(config)
            result = executor.run_vqe(circuit, ham)

            assert result.optimizer == optimizer
            assert result.energy is not None

    def test_statistics_tracking(self):
        """Executor should track run statistics."""
        circuit = create_simple_ansatz(2)
        ham = xy_chain(2).operator

        executor = QuantumExecutor()
        executor.run_vqe(circuit, ham)
        executor.run_vqe(circuit, ham)

        stats = executor.get_stats()
        assert stats["total_runs"] == 2

    def test_compute_expectation(self):
        """Test direct expectation value computation."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        from qiskit.quantum_info import SparsePauliOp

        observable = SparsePauliOp.from_list([("ZZ", 1.0)])

        executor = QuantumExecutor()
        exp_val = executor.compute_expectation(circuit, observable)

        # Bell state should have <ZZ> = 1
        assert np.isclose(exp_val, 1.0, atol=1e-6)

    def test_state_fidelity(self):
        """Test state fidelity computation."""
        from qiskit.quantum_info import Statevector

        # Same state should have fidelity 1
        state1 = Statevector.from_label("00")
        state2 = Statevector.from_label("00")

        executor = QuantumExecutor()
        fidelity = executor.compute_state_fidelity(state1, state2)

        assert np.isclose(fidelity, 1.0)

        # Orthogonal states should have fidelity 0
        state3 = Statevector.from_label("11")
        fidelity_orth = executor.compute_state_fidelity(state1, state3)

        assert np.isclose(fidelity_orth, 0.0)


class TestRunVQESimple:
    """Tests for run_vqe_simple convenience function."""

    def test_basic_usage(self):
        """Test basic usage of simple VQE runner."""
        circuit = create_simple_ansatz(2)
        ham = xy_chain(2).operator

        result = run_vqe_simple(circuit, ham)

        assert isinstance(result, VQEResult)
        assert result.energy is not None

    def test_custom_parameters(self):
        """Test with custom parameters."""
        circuit = create_simple_ansatz(2)
        ham = xy_chain(2).operator

        result = run_vqe_simple(
            circuit,
            ham,
            optimizer="Nelder-Mead",
            max_iterations=50,
            seed=123,
        )

        assert result.optimizer == "Nelder-Mead"


class TestParameterSweep:
    """Tests for parameter_sweep function."""

    def test_basic_sweep(self):
        """Test basic parameter sweep."""
        circuit = QuantumCircuit(2)
        theta = Parameter("theta")
        circuit.ry(theta, 0)
        circuit.cx(0, 1)

        ham = xy_chain(2).operator
        values = np.linspace(0, 2 * np.pi, 10)

        results = parameter_sweep(circuit, ham, param_index=0, values=values)

        assert len(results) == 10
        assert all(isinstance(r, float) for r in results)

    def test_sweep_shows_variation(self):
        """Sweep should show energy variation."""
        circuit = QuantumCircuit(1)
        theta = Parameter("theta")
        circuit.ry(theta, 0)

        from qiskit.quantum_info import SparsePauliOp

        ham = SparsePauliOp.from_list([("Z", 1.0)])
        values = np.array([0, np.pi / 2, np.pi])

        results = parameter_sweep(circuit, ham, param_index=0, values=values)

        # At theta=0: |0>, <Z>=1
        # At theta=pi/2: |+>, <Z>=0
        # At theta=pi: |1>, <Z>=-1
        assert np.isclose(results[0], 1.0, atol=0.1)
        assert np.isclose(results[1], 0.0, atol=0.1)
        assert np.isclose(results[2], -1.0, atol=0.1)


class TestEstimateGradient:
    """Tests for gradient estimation."""

    def test_gradient_shape(self):
        """Gradient should match number of parameters."""
        circuit = create_simple_ansatz(2)
        ham = xy_chain(2).operator

        params = np.random.uniform(0, 2 * np.pi, len(circuit.parameters))
        grad = estimate_gradient(circuit, ham, params)

        assert len(grad) == len(circuit.parameters)

    def test_gradient_values_finite(self):
        """Gradient values should be finite."""
        circuit = create_simple_ansatz(2)
        ham = xy_chain(2).operator

        params = np.random.uniform(0, 2 * np.pi, len(circuit.parameters))
        grad = estimate_gradient(circuit, ham, params)

        assert all(np.isfinite(g) for g in grad)


class TestMultiStartVQE:
    """Tests for MultiStartVQE class."""

    def test_basic_multi_start(self):
        """Test basic multi-start VQE."""
        circuit = create_simple_ansatz(2)
        ham = xy_chain(2).operator

        multi_vqe = MultiStartVQE(num_starts=3)
        result = multi_vqe.run(circuit, ham)

        assert isinstance(result, VQEResult)
        assert "multi_start" in result.metadata
        assert len(result.metadata["multi_start"]["all_energies"]) == 3

    def test_multi_start_finds_better(self):
        """Multi-start should find at least as good as single start."""
        circuit = create_simple_ansatz(3, num_layers=2)
        ham = transverse_ising(3, J=1.0, h=0.5).operator

        # Single start
        config = ExecutorConfig(max_iterations=50, seed=42)
        single_executor = QuantumExecutor(config)
        single_result = single_executor.run_vqe(circuit, ham)

        # Multi-start
        multi_vqe = MultiStartVQE(num_starts=3, executor_config=config)
        multi_result = multi_vqe.run(circuit, ham, seeds=[42, 43, 44])

        # Multi-start should be at least as good
        assert multi_result.energy <= single_result.energy + 0.01

    def test_custom_seeds(self):
        """Test with custom seeds."""
        circuit = create_simple_ansatz(2)
        ham = xy_chain(2).operator

        multi_vqe = MultiStartVQE(num_starts=3)
        result = multi_vqe.run(circuit, ham, seeds=[100, 200, 300])

        assert len(result.metadata["multi_start"]["all_energies"]) == 3


class TestExecutorConfig:
    """Tests for ExecutorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExecutorConfig()

        assert config.optimizer == "COBYLA"
        assert config.max_iterations == 200
        assert config.seed == 42

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExecutorConfig(
            optimizer="SLSQP",
            max_iterations=100,
            tolerance=1e-8,
            seed=123,
        )

        assert config.optimizer == "SLSQP"
        assert config.max_iterations == 100
        assert config.tolerance == 1e-8
        assert config.seed == 123


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
