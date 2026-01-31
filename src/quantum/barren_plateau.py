"""
Barren plateau detection for quantum circuits.

Barren plateaus occur when the gradient variance of a cost function
decays exponentially with system size, making optimization intractable.

This module implements:
1. Gradient variance analysis via parameter shift rule
2. Heuristic checks based on circuit structure
3. Trainability classification

Based on:
- McClean et al., "Barren plateaus in quantum neural network training landscapes"
  Nature Communications, 2018
- Cerezo et al., "Cost function dependent barren plateaus in shallow
  parametrized quantum circuits" Nature Communications, 2021
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp


@dataclass
class BarrenPlateauResult:
    """Result of barren plateau detection."""

    has_barren_plateau: bool
    gradient_variance: float
    gradient_mean: float
    trainability: Literal["high", "medium", "low", "very_low", "unknown"]
    recommendation: str
    details: dict


def compute_gradient_parameter_shift(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    param_values: dict,
    param_index: int,
    shift: float = np.pi / 2,
) -> float:
    """
    Compute gradient using parameter shift rule.

    The parameter shift rule states that for gates of the form exp(-i*theta*P/2):
    d<H>/d(theta) = (1/2) * [<H>|theta+pi/2 - <H>|theta-pi/2]

    Args:
        circuit: Parameterized quantum circuit
        hamiltonian: Observable to measure
        param_values: Dictionary mapping parameters to values
        param_index: Index of parameter to differentiate
        shift: Shift amount (default pi/2)

    Returns:
        Gradient value at the given point
    """
    estimator = StatevectorEstimator()

    params = list(circuit.parameters)
    if param_index >= len(params):
        raise ValueError(f"param_index {param_index} out of range (max {len(params) - 1})")

    target_param = params[param_index]

    # Forward shift
    forward_values = param_values.copy()
    forward_values[target_param] = param_values[target_param] + shift
    forward_circuit = circuit.assign_parameters(forward_values)

    # Backward shift
    backward_values = param_values.copy()
    backward_values[target_param] = param_values[target_param] - shift
    backward_circuit = circuit.assign_parameters(backward_values)

    # Compute expectation values
    job = estimator.run([(forward_circuit, hamiltonian), (backward_circuit, hamiltonian)])
    result = job.result()

    forward_exp = result[0].data.evs
    backward_exp = result[1].data.evs

    gradient = (forward_exp - backward_exp) / (2 * np.sin(shift))
    return float(gradient)


def detect_barren_plateau(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    num_samples: int = 50,
    seed: int = 42,
) -> BarrenPlateauResult:
    """
    Detect barren plateau by sampling gradient variance.

    This function samples random parameter values and computes
    the gradient variance. Low variance indicates a barren plateau.

    Args:
        circuit: Parameterized quantum circuit to analyze
        hamiltonian: Target Hamiltonian for optimization
        num_samples: Number of random parameter samples
        seed: Random seed for reproducibility

    Returns:
        BarrenPlateauResult with detection outcome and recommendations
    """
    np.random.seed(seed)

    params = list(circuit.parameters)
    num_params = len(params)

    if num_params == 0:
        return BarrenPlateauResult(
            has_barren_plateau=True,
            gradient_variance=0.0,
            gradient_mean=0.0,
            trainability="unknown",
            recommendation="Circuit has no trainable parameters. Cannot optimize.",
            details={"error": "no_parameters", "num_params": 0},
        )

    all_gradients = []
    failed_samples = 0

    for _ in range(num_samples):
        # Random parameter values in [0, 2*pi]
        random_values = np.random.uniform(0, 2 * np.pi, num_params)
        param_dict = dict(zip(params, random_values))

        # Compute gradient for first parameter (representative sample)
        # In practice, all parameters in deep random circuits show similar behavior
        try:
            grad = compute_gradient_parameter_shift(
                circuit, hamiltonian, param_dict, param_index=0
            )
            all_gradients.append(grad)
        except Exception:
            failed_samples += 1
            continue

    # Need minimum samples for reliable statistics
    if len(all_gradients) < 10:
        return BarrenPlateauResult(
            has_barren_plateau=True,
            gradient_variance=0.0,
            gradient_mean=0.0,
            trainability="unknown",
            recommendation="Could not compute enough gradient samples. Check circuit validity.",
            details={
                "error": "insufficient_samples",
                "successful_samples": len(all_gradients),
                "failed_samples": failed_samples,
            },
        )

    gradients = np.array(all_gradients)
    variance = float(np.var(gradients))
    mean_abs = float(np.mean(np.abs(gradients)))

    # Classification based on variance thresholds
    # These thresholds are from empirical studies in the literature
    #
    # variance > 1e-2: Circuit is well-trainable
    # variance > 1e-4: Circuit may have mild issues
    # variance > 1e-6: Moderate barren plateau
    # variance <= 1e-6: Severe barren plateau

    if variance > 1e-2:
        has_bp = False
        trainability = "high"
        recommendation = (
            "Circuit appears highly trainable. "
            "Gradient variance is sufficient for optimization."
        )
    elif variance > 1e-4:
        has_bp = False
        trainability = "medium"
        recommendation = (
            "Circuit may have mild trainability issues. "
            "Consider monitoring convergence carefully."
        )
    elif variance > 1e-6:
        has_bp = True
        trainability = "low"
        recommendation = (
            "Barren plateau detected. Consider using a shallower circuit, "
            "layer-wise training, or problem-inspired ansatz design."
        )
    else:
        has_bp = True
        trainability = "very_low"
        recommendation = (
            "Severe barren plateau detected. This circuit is unlikely to train. "
            "Reject and try a different architecture."
        )

    return BarrenPlateauResult(
        has_barren_plateau=has_bp,
        gradient_variance=variance,
        gradient_mean=mean_abs,
        trainability=trainability,
        recommendation=recommendation,
        details={
            "num_samples": len(all_gradients),
            "failed_samples": failed_samples,
            "num_params": num_params,
            "circuit_depth": circuit.depth(),
            "num_qubits": circuit.num_qubits,
        },
    )


def quick_bp_heuristic(circuit: QuantumCircuit) -> tuple[bool, str, dict]:
    """
    Quick heuristic check for barren plateau risk.

    This uses circuit structure analysis without gradient computation.
    Faster than full detection but less accurate.

    Args:
        circuit: Quantum circuit to analyze

    Returns:
        Tuple of (likely_has_bp, reason, details)
    """
    depth = circuit.depth()
    num_qubits = circuit.num_qubits
    num_params = len(circuit.parameters)
    ops = circuit.count_ops()

    details = {
        "depth": depth,
        "num_qubits": num_qubits,
        "num_params": num_params,
        "gate_counts": dict(ops),
    }

    # Heuristic 1: Very deep circuits are prone to BP
    # Literature suggests depth > 2*n is problematic for random circuits
    if depth > 2 * num_qubits:
        return True, f"Circuit depth ({depth}) > 2 * num_qubits ({2 * num_qubits})", details

    # Heuristic 2: No parameters means no gradient
    if num_params == 0:
        return True, "Circuit has no trainable parameters", details

    # Heuristic 3: Check for trainable rotation gates
    rotation_gates = {"rx", "ry", "rz", "u", "u1", "u2", "u3", "p"}
    has_rotations = any(g in ops for g in rotation_gates)
    if not has_rotations:
        return True, "No parameterized rotation gates found", details

    # Heuristic 4: Single-qubit circuits or no entanglement
    entangling_gates = {"cx", "cz", "cy", "swap", "iswap", "ecr", "rzx", "rxx", "ryy", "rzz"}
    has_entanglement = any(g in ops for g in entangling_gates)
    if num_qubits > 1 and not has_entanglement:
        return (
            True,
            "Multi-qubit circuit without entanglement - limited expressibility",
            details,
        )

    # Heuristic 5: Too many parameters for the circuit size
    # Over-parameterization can lead to redundancy but isn't necessarily BP
    if num_params > 4 * num_qubits * depth:
        details["warning"] = "Potentially over-parameterized"

    return False, "Passed heuristic checks", details


def analyze_circuit_structure(circuit: QuantumCircuit) -> dict:
    """
    Analyze circuit structure for BP risk factors.

    Returns detailed analysis of circuit properties relevant to trainability.

    Args:
        circuit: Circuit to analyze

    Returns:
        Dictionary with structural analysis
    """
    ops = circuit.count_ops()
    depth = circuit.depth()
    num_qubits = circuit.num_qubits
    num_params = len(circuit.parameters)

    # Count gate types
    single_qubit_gates = sum(
        count for gate, count in ops.items() if gate in {"rx", "ry", "rz", "h", "x", "y", "z", "s", "t", "u", "u1", "u2", "u3", "p"}
    )

    two_qubit_gates = sum(
        count for gate, count in ops.items() if gate in {"cx", "cz", "cy", "swap", "iswap", "ecr", "rzx", "rxx", "ryy", "rzz"}
    )

    # Calculate metrics
    total_gates = sum(ops.values())
    entanglement_ratio = two_qubit_gates / max(total_gates, 1)
    params_per_qubit = num_params / max(num_qubits, 1)
    depth_per_qubit = depth / max(num_qubits, 1)

    # Risk assessment
    risk_factors = []
    if depth > 2 * num_qubits:
        risk_factors.append("deep_circuit")
    if entanglement_ratio > 0.5:
        risk_factors.append("high_entanglement")
    if params_per_qubit > 10:
        risk_factors.append("over_parameterized")
    if depth_per_qubit > 3:
        risk_factors.append("high_depth_per_qubit")

    # Calculate overall risk score (0-1)
    risk_score = len(risk_factors) / 4.0

    return {
        "depth": depth,
        "num_qubits": num_qubits,
        "num_params": num_params,
        "total_gates": total_gates,
        "single_qubit_gates": single_qubit_gates,
        "two_qubit_gates": two_qubit_gates,
        "entanglement_ratio": entanglement_ratio,
        "params_per_qubit": params_per_qubit,
        "depth_per_qubit": depth_per_qubit,
        "risk_factors": risk_factors,
        "risk_score": risk_score,
        "gate_counts": dict(ops),
    }


def estimate_trainability_score(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp | None = None,
    num_samples: int = 20,
    seed: int = 42,
) -> float:
    """
    Estimate a trainability score from 0 (untrainable) to 1 (highly trainable).

    This combines heuristic analysis with optional gradient sampling
    to provide a quick assessment.

    Args:
        circuit: Circuit to evaluate
        hamiltonian: Optional Hamiltonian for gradient-based assessment
        num_samples: Number of gradient samples if hamiltonian provided
        seed: Random seed

    Returns:
        Trainability score between 0 and 1
    """
    # Start with heuristic analysis
    structure = analyze_circuit_structure(circuit)
    heuristic_score = 1.0 - structure["risk_score"]

    if hamiltonian is None:
        return heuristic_score

    # Refine with gradient variance if hamiltonian provided
    try:
        bp_result = detect_barren_plateau(circuit, hamiltonian, num_samples=num_samples, seed=seed)

        # Convert variance to score (log scale)
        variance = bp_result.gradient_variance
        if variance <= 1e-8:
            gradient_score = 0.0
        elif variance >= 1e-2:
            gradient_score = 1.0
        else:
            # Log scale between 1e-8 and 1e-2
            gradient_score = (np.log10(variance) + 8) / 6

        # Combine scores (gradient is more reliable)
        combined_score = 0.3 * heuristic_score + 0.7 * gradient_score
        return float(np.clip(combined_score, 0, 1))

    except Exception:
        return heuristic_score
