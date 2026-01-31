# Evaluation and Benchmarking

## Metrics Hierarchy

### Primary Metrics (Must Report)

| Metric | Symbol | Target | Unit |
|--------|--------|--------|------|
| Energy Error | E_err | < 0.01 | Hartree |
| Relative Error | E_rel | < 1% | % |
| Circuit Depth | D | Minimize | Gates |
| Gate Count | G | Minimize | Count |
| Parameter Count | P | Minimize | Count |

### Secondary Metrics (Should Report)

| Metric | Description | Target |
|--------|-------------|--------|
| State Fidelity | F = |<exact|prepared>|^2 | > 0.95 |
| Gradient Variance | Var(grad) | > 1e-4 |
| VQE Convergence | Iterations to converge | < 100 |
| Hardware Fidelity | Performance on real QPU | > 0.80 |

### Efficiency Metrics

| Metric | Formula | Better |
|--------|---------|--------|
| Depth Efficiency | E_err / D | Higher |
| Parameter Efficiency | E_err / P | Higher |
| Gate Efficiency | E_err / G | Higher |

---

## Baseline Comparisons

### Required Baselines

Every experiment must compare against:

1. **Hardware-Efficient Ansatz (HEA)**
   - Standard TwoLocal with RY-RZ rotations
   - Linear CX entanglement
   - Same number of layers as discovered circuit

2. **EfficientSU2**
   - Qiskit's built-in efficient ansatz
   - Full entanglement

3. **Random Initialization**
   - Same architecture, random parameters
   - Establishes optimization difficulty

4. **Exact Solution**
   - Diagonalize Hamiltonian
   - Ground truth for error calculation

### Optional Baselines

- UCCSD (for chemistry problems)
- QAOA (for optimization)
- Previous published results

---

## Evaluation Protocol

### Protocol 1: Single Circuit Evaluation

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class EvaluationResult:
    """Complete evaluation of a single circuit."""
    # Energy
    energy: float
    energy_error: float
    relative_error: float

    # Circuit metrics
    depth: int
    gate_count: int
    param_count: int

    # Quality metrics
    fidelity: Optional[float]
    gradient_variance: float

    # Optimization
    vqe_iterations: int
    converged: bool

    # Hardware (if applicable)
    hardware_energy: Optional[float]
    hardware_std: Optional[float]

def evaluate_circuit(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    exact_energy: float,
    run_hardware: bool = False
) -> EvaluationResult:
    """
    Complete evaluation of a quantum circuit.
    """
    # 1. Circuit metrics
    depth = circuit.depth()
    gate_count = sum(circuit.count_ops().values())
    param_count = len(circuit.parameters)

    # 2. Barren plateau check
    bp_result = detect_barren_plateau(circuit, hamiltonian)
    gradient_variance = bp_result.gradient_variance

    # 3. Run VQE
    vqe_result = run_vqe(circuit, hamiltonian)
    energy = vqe_result.energy
    vqe_iterations = vqe_result.iterations
    converged = vqe_result.converged

    # 4. Calculate errors
    energy_error = abs(energy - exact_energy)
    relative_error = energy_error / abs(exact_energy) * 100

    # 5. Fidelity (if exact state available)
    fidelity = compute_fidelity(vqe_result.state, exact_state) if exact_state else None

    # 6. Hardware validation (optional)
    if run_hardware:
        hw_energy, hw_std = run_on_hardware(circuit, hamiltonian, vqe_result.params)
    else:
        hw_energy, hw_std = None, None

    return EvaluationResult(
        energy=energy,
        energy_error=energy_error,
        relative_error=relative_error,
        depth=depth,
        gate_count=gate_count,
        param_count=param_count,
        fidelity=fidelity,
        gradient_variance=gradient_variance,
        vqe_iterations=vqe_iterations,
        converged=converged,
        hardware_energy=hw_energy,
        hardware_std=hw_std
    )
```

### Protocol 2: Statistical Evaluation

Run multiple trials for statistical significance:

```python
def statistical_evaluation(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    exact_energy: float,
    num_trials: int = 10,
    seeds: list = None
) -> dict:
    """
    Statistical evaluation with multiple random initializations.
    """
    if seeds is None:
        seeds = list(range(num_trials))

    results = []
    for seed in seeds:
        np.random.seed(seed)
        result = evaluate_circuit(circuit, hamiltonian, exact_energy)
        results.append(result)

    # Aggregate statistics
    energies = [r.energy for r in results]
    errors = [r.energy_error for r in results]

    return {
        "mean_energy": np.mean(energies),
        "std_energy": np.std(energies),
        "mean_error": np.mean(errors),
        "std_error": np.std(errors),
        "best_error": min(errors),
        "worst_error": max(errors),
        "convergence_rate": sum(r.converged for r in results) / num_trials,
        "num_trials": num_trials
    }
```

### Protocol 3: Comparative Evaluation

Compare discovered circuits against baselines:

```python
def comparative_evaluation(
    discovered_circuit: QuantumCircuit,
    goal: PhysicsGoal,
    num_trials: int = 10
) -> dict:
    """
    Compare discovered circuit against all baselines.
    """
    # Baselines
    hea = hardware_efficient_ansatz(goal.num_qubits, depth=2)
    su2 = efficient_su2_ansatz(goal.num_qubits, depth=2)

    # Evaluate all
    results = {
        "discovered": statistical_evaluation(discovered_circuit, goal.hamiltonian, goal.exact_energy, num_trials),
        "hea": statistical_evaluation(hea, goal.hamiltonian, goal.exact_energy, num_trials),
        "su2": statistical_evaluation(su2, goal.hamiltonian, goal.exact_energy, num_trials),
    }

    # Compute improvements
    results["improvement_vs_hea"] = {
        "error_reduction": (results["hea"]["mean_error"] - results["discovered"]["mean_error"]) / results["hea"]["mean_error"] * 100,
        "depth_reduction": (hea.depth() - discovered_circuit.depth()) / hea.depth() * 100,
    }

    return results
```

---

## Success Criteria

### For a Discovered Circuit to Be "Successful"

| Criterion | Threshold | Required |
|-----------|-----------|----------|
| Energy error | < 0.01 Ha | Yes |
| Better than HEA | > 10% improvement | Yes |
| Trainable | Gradient variance > 1e-4 | Yes |
| Reproducible | Std < 20% of mean | Yes |
| Hardware validated | Works on IBM | For publication |

### For the Overall Project to Be "Successful"

| Criterion | Target |
|-----------|--------|
| Circuits discovered | > 5 novel circuits |
| Problems solved | All 3 target Hamiltonians |
| Best improvement | > 20% over baseline |
| Hardware validation | At least 3 circuits |
| Paper potential | Clear novel contribution |

---

## Reporting Format

### Table Format

```
Table 1: Comparison of Discovered Circuits vs Baselines (4-qubit XY Chain)

| Circuit      | Energy    | Error     | Depth | Gates | Params |
|--------------|-----------|-----------|-------|-------|--------|
| Exact        | -2.8284   | -         | -     | -     | -      |
| HEA (2-layer)| -2.7102   | 0.1182    | 12    | 28    | 16     |
| EfficientSU2 | -2.7456   | 0.0828    | 10    | 24    | 12     |
| Discovered   | -2.8201   | 0.0083    | 6     | 14    | 8      |

Improvement: 93% error reduction, 50% depth reduction vs HEA
```

### Figure Format

Required figures:
1. **Energy convergence**: VQE iterations vs energy
2. **Error comparison**: Bar chart of energy errors
3. **Circuit diagram**: Discovered circuit architecture
4. **Scaling plot**: Performance vs qubit count

```python
import matplotlib.pyplot as plt

def plot_comparison(results: dict, save_path: str):
    """Generate comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Energy error
    methods = list(results.keys())
    errors = [results[m]["mean_error"] for m in methods]
    stds = [results[m]["std_error"] for m in methods]

    axes[0].bar(methods, errors, yerr=stds, capsize=5)
    axes[0].set_ylabel("Energy Error (Ha)")
    axes[0].set_title("Energy Error Comparison")

    # Depth
    depths = [results[m].get("depth", 0) for m in methods]
    axes[1].bar(methods, depths)
    axes[1].set_ylabel("Circuit Depth")
    axes[1].set_title("Circuit Depth Comparison")

    # Gate count
    gates = [results[m].get("gate_count", 0) for m in methods]
    axes[2].bar(methods, gates)
    axes[2].set_ylabel("Gate Count")
    axes[2].set_title("Gate Count Comparison")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## Hardware Evaluation

### IBM Quantum Validation

```python
def hardware_evaluation(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    optimal_params: list,
    shots: int = 4000,
    num_runs: int = 5
) -> dict:
    """
    Evaluate circuit on IBM quantum hardware.
    """
    runner = IBMHardwareRunner(backend_name="ibm_sherbrooke")

    results = []
    for i in range(num_runs):
        energy, std = runner.run_estimation(
            circuit, hamiltonian, optimal_params, shots
        )
        results.append({"energy": energy, "std": std})
        print(f"Run {i+1}: E = {energy:.6f} +/- {std:.6f}")

    # Aggregate
    energies = [r["energy"] for r in results]

    return {
        "mean_energy": np.mean(energies),
        "std_energy": np.std(energies),
        "best_energy": min(energies),
        "num_runs": num_runs,
        "shots_per_run": shots,
        "backend": "ibm_sherbrooke",
        "error_mitigation": "resilience_level=2"
    }
```

### Noise Analysis

```python
def noise_comparison(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    optimal_params: list
) -> dict:
    """
    Compare ideal simulation vs noisy simulation vs hardware.
    """
    # Ideal
    ideal_energy = run_ideal_simulation(circuit, hamiltonian, optimal_params)

    # Noisy simulation (with noise model)
    noisy_energy = run_noisy_simulation(circuit, hamiltonian, optimal_params)

    # Hardware
    hw_energy, hw_std = run_on_hardware(circuit, hamiltonian, optimal_params)

    return {
        "ideal": ideal_energy,
        "noisy_sim": noisy_energy,
        "hardware": hw_energy,
        "sim_to_ideal_gap": abs(noisy_energy - ideal_energy),
        "hw_to_ideal_gap": abs(hw_energy - ideal_energy),
        "hw_to_sim_gap": abs(hw_energy - noisy_energy)
    }
```

---

## Experiment Logging

### Log Format

```python
import json
from datetime import datetime

def log_experiment(
    experiment_name: str,
    goal: PhysicsGoal,
    circuit_code: str,
    results: dict,
    metadata: dict
):
    """Log experiment for reproducibility."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment": experiment_name,
        "goal": {
            "name": goal.name,
            "hamiltonian": goal.hamiltonian,
            "num_qubits": goal.num_qubits,
            "exact_energy": goal.exact_energy
        },
        "circuit_code": circuit_code,
        "results": results,
        "metadata": metadata
    }

    # Save to file
    filename = f"experiments/logs/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(log_entry, f, indent=2)

    return filename
```

### Required Metadata

```python
metadata = {
    "seed": 42,
    "vqe_optimizer": "COBYLA",
    "vqe_max_iterations": 200,
    "model_version": "quantum-mind-v1",
    "discovery_iteration": 15,
    "hardware_backend": "ibm_sherbrooke",
    "error_mitigation": True,
    "git_commit": "abc123..."
}
```

---

## Statistical Tests

### Significance Testing

```python
from scipy import stats

def test_improvement_significance(
    discovered_results: list,
    baseline_results: list,
    alpha: float = 0.05
) -> dict:
    """
    Test if discovered circuit significantly beats baseline.
    """
    # Paired t-test (same random seeds)
    t_stat, p_value = stats.ttest_rel(baseline_results, discovered_results)

    # Effect size (Cohen's d)
    diff = np.array(baseline_results) - np.array(discovered_results)
    cohens_d = np.mean(diff) / np.std(diff)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "cohens_d": cohens_d,
        "effect_size": "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"
    }
```

---

## Evaluation Checklist

Before claiming any result:

- [ ] Run statistical evaluation (10+ trials)
- [ ] Compare against all baselines
- [ ] Report mean +/- std
- [ ] Perform significance test
- [ ] Document all hyperparameters
- [ ] Save experiment logs
- [ ] Create visualizations
- [ ] Validate on hardware (for final results)

---

## Summary

| What to Measure | How | Target |
|-----------------|-----|--------|
| Energy error | VQE optimization | < 0.01 Ha |
| Improvement | vs HEA, EfficientSU2 | > 10% |
| Depth | circuit.depth() | Minimize |
| Trainability | Gradient variance | > 1e-4 |
| Reproducibility | Multiple trials | Std < 20% mean |
| Hardware | IBM Quantum | Confirms simulation |

Follow this evaluation protocol for all experiments. No exceptions.
