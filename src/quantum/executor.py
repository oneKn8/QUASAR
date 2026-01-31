"""
Quantum circuit execution module.

This module provides:
1. VQE (Variational Quantum Eigensolver) optimization
2. Local simulation with Aer
3. IBM Quantum hardware execution (when configured)
4. Result analysis and logging

The executor handles parameter optimization to find ground state energies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize


@dataclass
class VQEResult:
    """Result of VQE optimization."""

    energy: float
    optimal_params: np.ndarray
    num_iterations: int
    num_function_evals: int
    converged: bool
    convergence_history: list[float] = field(default_factory=list)
    final_state: Statevector | None = None
    circuit: QuantumCircuit | None = None
    optimizer: str = ""
    runtime_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ExecutorConfig:
    """Configuration for quantum executor."""

    optimizer: str = "COBYLA"
    max_iterations: int = 200
    tolerance: float = 1e-6
    seed: int = 42
    track_convergence: bool = True
    compute_final_state: bool = True
    verbose: bool = False


class QuantumExecutor:
    """
    Execute quantum circuits with VQE optimization.

    This executor uses classical simulation via Qiskit's StatevectorEstimator.
    For hardware execution, use IBMHardwareRunner.
    """

    def __init__(self, config: ExecutorConfig | None = None):
        """
        Initialize executor.

        Args:
            config: Executor configuration (uses defaults if None)
        """
        self.config = config or ExecutorConfig()
        self.estimator = StatevectorEstimator()

        # Statistics
        self.total_runs = 0
        self.total_converged = 0

    def run_vqe(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        initial_params: np.ndarray | None = None,
    ) -> VQEResult:
        """
        Run VQE optimization on the given circuit.

        Args:
            circuit: Parameterized quantum circuit (ansatz)
            hamiltonian: Target Hamiltonian to minimize
            initial_params: Initial parameter values (random if None)

        Returns:
            VQEResult with optimization outcome
        """
        start_time = datetime.now()
        self.total_runs += 1

        params = list(circuit.parameters)
        num_params = len(params)

        if num_params == 0:
            raise ValueError("Circuit has no parameters to optimize")

        # Initialize parameters
        if initial_params is None:
            np.random.seed(self.config.seed)
            initial_params = np.random.uniform(0, 2 * np.pi, num_params)

        # Track convergence
        convergence_history = []
        iteration_count = [0]

        def cost_function(param_values: np.ndarray) -> float:
            """Compute expectation value of Hamiltonian."""
            param_dict = dict(zip(params, param_values))
            bound_circuit = circuit.assign_parameters(param_dict)

            job = self.estimator.run([(bound_circuit, hamiltonian)])
            result = job.result()
            energy = float(result[0].data.evs)

            if self.config.track_convergence:
                convergence_history.append(energy)

            iteration_count[0] += 1

            if self.config.verbose and iteration_count[0] % 20 == 0:
                print(f"  Iteration {iteration_count[0]}: E = {energy:.6f}")

            return energy

        # Select optimizer
        optimizer_options = self._get_optimizer_options()

        # Run optimization
        result = minimize(
            cost_function,
            initial_params,
            method=self.config.optimizer,
            options=optimizer_options,
        )

        # Check convergence
        converged = result.success or iteration_count[0] < self.config.max_iterations

        if converged:
            self.total_converged += 1

        # Compute final state if requested
        final_state = None
        if self.config.compute_final_state:
            param_dict = dict(zip(params, result.x))
            bound_circuit = circuit.assign_parameters(param_dict)
            final_state = Statevector(bound_circuit)

        # Create optimized circuit
        param_dict = dict(zip(params, result.x))
        optimized_circuit = circuit.assign_parameters(param_dict)

        runtime = (datetime.now() - start_time).total_seconds()

        return VQEResult(
            energy=float(result.fun),
            optimal_params=result.x,
            num_iterations=iteration_count[0],
            num_function_evals=result.nfev if hasattr(result, "nfev") else iteration_count[0],
            converged=converged,
            convergence_history=convergence_history,
            final_state=final_state,
            circuit=optimized_circuit,
            optimizer=self.config.optimizer,
            runtime_seconds=runtime,
            metadata={
                "optimizer_message": result.message if hasattr(result, "message") else "",
                "seed": self.config.seed,
            },
        )

    def compute_expectation(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        param_values: np.ndarray | None = None,
    ) -> float:
        """
        Compute expectation value without optimization.

        Args:
            circuit: Quantum circuit
            hamiltonian: Observable to measure
            param_values: Parameter values (for parameterized circuits)

        Returns:
            Expectation value
        """
        if param_values is not None and len(circuit.parameters) > 0:
            params = list(circuit.parameters)
            param_dict = dict(zip(params, param_values))
            bound_circuit = circuit.assign_parameters(param_dict)
        else:
            bound_circuit = circuit

        job = self.estimator.run([(bound_circuit, hamiltonian)])
        result = job.result()
        return float(result[0].data.evs)

    def compute_state_fidelity(
        self,
        state1: Statevector,
        state2: Statevector,
    ) -> float:
        """
        Compute fidelity between two quantum states.

        F = |<psi1|psi2>|^2

        Args:
            state1: First state
            state2: Second state

        Returns:
            Fidelity value between 0 and 1
        """
        inner_product = np.abs(np.vdot(state1.data, state2.data)) ** 2
        return float(inner_product)

    def _get_optimizer_options(self) -> dict:
        """Get optimizer-specific options."""
        base_options = {
            "maxiter": self.config.max_iterations,
        }

        if self.config.optimizer == "COBYLA":
            return {
                **base_options,
                "rhobeg": 0.5,
                "tol": self.config.tolerance,
            }
        elif self.config.optimizer == "SLSQP":
            return {
                **base_options,
                "ftol": self.config.tolerance,
            }
        elif self.config.optimizer in ["L-BFGS-B", "BFGS"]:
            return {
                **base_options,
                "gtol": self.config.tolerance,
            }
        elif self.config.optimizer == "Nelder-Mead":
            return {
                **base_options,
                "xatol": self.config.tolerance,
                "fatol": self.config.tolerance,
            }
        else:
            return base_options

    def get_stats(self) -> dict:
        """Get execution statistics."""
        return {
            "total_runs": self.total_runs,
            "total_converged": self.total_converged,
            "convergence_rate": (
                self.total_converged / self.total_runs if self.total_runs > 0 else 0
            ),
        }

    def reset_stats(self):
        """Reset execution statistics."""
        self.total_runs = 0
        self.total_converged = 0


def run_vqe_simple(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    optimizer: str = "COBYLA",
    max_iterations: int = 200,
    seed: int = 42,
) -> VQEResult:
    """
    Simple VQE runner with minimal configuration.

    This is a convenience function for quick experiments.

    Args:
        circuit: Parameterized ansatz circuit
        hamiltonian: Target Hamiltonian
        optimizer: Optimizer name
        max_iterations: Maximum iterations
        seed: Random seed

    Returns:
        VQEResult
    """
    config = ExecutorConfig(
        optimizer=optimizer,
        max_iterations=max_iterations,
        seed=seed,
        verbose=False,
    )
    executor = QuantumExecutor(config)
    return executor.run_vqe(circuit, hamiltonian)


def parameter_sweep(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    param_index: int,
    values: np.ndarray,
    fixed_params: np.ndarray | None = None,
) -> list[float]:
    """
    Sweep a single parameter and compute expectation values.

    Useful for visualizing the energy landscape.

    Args:
        circuit: Parameterized circuit
        hamiltonian: Observable
        param_index: Index of parameter to sweep
        values: Values to sweep
        fixed_params: Fixed values for other parameters

    Returns:
        List of expectation values
    """
    executor = QuantumExecutor()
    params = list(circuit.parameters)
    num_params = len(params)

    if fixed_params is None:
        fixed_params = np.zeros(num_params)

    results = []
    for val in values:
        param_values = fixed_params.copy()
        param_values[param_index] = val
        energy = executor.compute_expectation(circuit, hamiltonian, param_values)
        results.append(energy)

    return results


def estimate_gradient(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    param_values: np.ndarray,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """
    Estimate gradient using parameter shift rule.

    Args:
        circuit: Parameterized circuit
        hamiltonian: Observable
        param_values: Current parameter values
        shift: Shift for parameter shift rule

    Returns:
        Gradient vector
    """
    executor = QuantumExecutor()
    params = list(circuit.parameters)
    num_params = len(params)

    gradients = np.zeros(num_params)

    for i in range(num_params):
        # Forward shift
        forward_params = param_values.copy()
        forward_params[i] += shift
        forward_exp = executor.compute_expectation(circuit, hamiltonian, forward_params)

        # Backward shift
        backward_params = param_values.copy()
        backward_params[i] -= shift
        backward_exp = executor.compute_expectation(circuit, hamiltonian, backward_params)

        # Parameter shift gradient
        gradients[i] = (forward_exp - backward_exp) / (2 * np.sin(shift))

    return gradients


class MultiStartVQE:
    """
    VQE with multiple random starting points.

    Helps avoid local minima by running optimization from different
    initial parameter values and returning the best result.
    """

    def __init__(
        self,
        num_starts: int = 5,
        executor_config: ExecutorConfig | None = None,
    ):
        """
        Initialize multi-start VQE.

        Args:
            num_starts: Number of random starting points
            executor_config: Configuration for each VQE run
        """
        self.num_starts = num_starts
        self.config = executor_config or ExecutorConfig()

    def run(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        seeds: list[int] | None = None,
    ) -> VQEResult:
        """
        Run VQE from multiple starting points.

        Args:
            circuit: Parameterized circuit
            hamiltonian: Target Hamiltonian
            seeds: Optional list of seeds for each start

        Returns:
            Best VQEResult among all runs
        """
        if seeds is None:
            seeds = list(range(self.num_starts))

        best_result = None
        all_results = []

        for i, seed in enumerate(seeds):
            config = ExecutorConfig(
                optimizer=self.config.optimizer,
                max_iterations=self.config.max_iterations,
                tolerance=self.config.tolerance,
                seed=seed,
                track_convergence=self.config.track_convergence,
                compute_final_state=self.config.compute_final_state,
                verbose=self.config.verbose,
            )

            executor = QuantumExecutor(config)
            result = executor.run_vqe(circuit, hamiltonian)
            all_results.append(result)

            if best_result is None or result.energy < best_result.energy:
                best_result = result

        # Add metadata about multi-start
        best_result.metadata["multi_start"] = {
            "num_starts": self.num_starts,
            "all_energies": [r.energy for r in all_results],
            "best_start_index": all_results.index(best_result),
        }

        return best_result
