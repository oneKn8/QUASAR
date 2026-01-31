# QuantumMind Guideline: Phase 5 - Evaluation

> Steps 5.1-5.4: Baselines, Metrics, Comparison, Statistics

---

## Step 5.1: Baseline Ansatzes

**Status**: PENDING

**Prerequisites**: None (can run in parallel with Phase 3)

---

### Create Baselines Module

```python
# src/evaluation/baselines.py

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal, EfficientSU2
from typing import Callable
from dataclasses import dataclass


@dataclass
class BaselineAnsatz:
    """Describes a baseline ansatz for comparison."""
    name: str
    create_fn: Callable[[int], QuantumCircuit]
    description: str


def hardware_efficient_ansatz(num_qubits: int, depth: int = 2) -> QuantumCircuit:
    """
    Standard HEA with RY-RZ rotations and linear CX.

    Args:
        num_qubits: Number of qubits
        depth: Number of repetition layers

    Returns:
        Parameterized QuantumCircuit
    """
    return TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=['ry', 'rz'],
        entanglement_blocks='cx',
        entanglement='linear',
        reps=depth
    )


def efficient_su2_ansatz(num_qubits: int, depth: int = 2) -> QuantumCircuit:
    """
    Qiskit's EfficientSU2 ansatz.

    Args:
        num_qubits: Number of qubits
        depth: Number of repetition layers

    Returns:
        Parameterized QuantumCircuit
    """
    return EfficientSU2(
        num_qubits=num_qubits,
        reps=depth,
        entanglement='linear'
    )


def custom_hea(num_qubits: int, layers: int = 2) -> QuantumCircuit:
    """
    Custom HEA implementation for fair comparison.

    Structure per layer:
    - RY on each qubit
    - RZ on each qubit
    - Linear CX chain

    Final layer: RY on each qubit

    Args:
        num_qubits: Number of qubits
        layers: Number of entangling layers

    Returns:
        Parameterized QuantumCircuit
    """
    qc = QuantumCircuit(num_qubits)
    param_count = 0

    for layer in range(layers):
        # Rotation layer
        for i in range(num_qubits):
            qc.ry(Parameter(f'theta_{param_count}'), i)
            param_count += 1
            qc.rz(Parameter(f'theta_{param_count}'), i)
            param_count += 1

        # Entangling layer
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

    # Final rotation layer
    for i in range(num_qubits):
        qc.ry(Parameter(f'theta_{param_count}'), i)
        param_count += 1

    return qc


def random_ansatz(num_qubits: int, num_params: int = 16) -> QuantumCircuit:
    """
    Random circuit structure (control baseline).

    Args:
        num_qubits: Number of qubits
        num_params: Number of parameters

    Returns:
        Parameterized QuantumCircuit
    """
    import random
    random.seed(42)  # Deterministic for reproducibility

    qc = QuantumCircuit(num_qubits)
    params = [Parameter(f'p{i}') for i in range(num_params)]

    for p in params:
        qubit = random.randint(0, num_qubits - 1)
        gate = random.choice(['rx', 'ry', 'rz'])
        getattr(qc, gate)(p, qubit)

        # 50% chance to add entanglement
        if random.random() > 0.5 and num_qubits > 1:
            q1, q2 = random.sample(range(num_qubits), 2)
            qc.cx(q1, q2)

    return qc


# Registry of all baselines
BASELINES = {
    "hea_2layer": BaselineAnsatz(
        name="HEA (2-layer)",
        create_fn=lambda n: hardware_efficient_ansatz(n, 2),
        description="Standard hardware-efficient ansatz with 2 layers"
    ),
    "hea_3layer": BaselineAnsatz(
        name="HEA (3-layer)",
        create_fn=lambda n: hardware_efficient_ansatz(n, 3),
        description="Standard hardware-efficient ansatz with 3 layers"
    ),
    "su2_2layer": BaselineAnsatz(
        name="EfficientSU2 (2-layer)",
        create_fn=lambda n: efficient_su2_ansatz(n, 2),
        description="Qiskit EfficientSU2 with 2 layers"
    ),
    "custom_hea": BaselineAnsatz(
        name="Custom HEA",
        create_fn=custom_hea,
        description="Our custom HEA implementation for fair comparison"
    ),
    "random": BaselineAnsatz(
        name="Random",
        create_fn=random_ansatz,
        description="Random circuit structure (control baseline)"
    ),
}


def get_baseline(name: str, num_qubits: int) -> QuantumCircuit:
    """
    Get a baseline circuit by name.

    Args:
        name: Baseline identifier from BASELINES
        num_qubits: Number of qubits

    Returns:
        QuantumCircuit

    Raises:
        ValueError: If baseline name not found
    """
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINES.keys())}")
    return BASELINES[name].create_fn(num_qubits)


def get_all_baselines(num_qubits: int) -> dict:
    """
    Get all baseline circuits for given qubit count.

    Args:
        num_qubits: Number of qubits

    Returns:
        Dict mapping name to QuantumCircuit
    """
    return {name: b.create_fn(num_qubits) for name, b in BASELINES.items()}


def print_baseline_info(num_qubits: int = 4):
    """Print info about all baselines."""
    print(f"Baseline Ansatzes ({num_qubits} qubits):")
    print("-" * 60)

    for name, baseline in BASELINES.items():
        circuit = baseline.create_fn(num_qubits)
        ops = circuit.count_ops()
        print(f"\n{baseline.name}")
        print(f"  Description: {baseline.description}")
        print(f"  Parameters: {len(circuit.parameters)}")
        print(f"  Depth: {circuit.depth()}")
        print(f"  Gates: {ops}")
```

---

### Create Tests

```python
# tests/test_baselines.py

import pytest
from src.evaluation.baselines import (
    hardware_efficient_ansatz,
    efficient_su2_ansatz,
    custom_hea,
    random_ansatz,
    get_baseline,
    get_all_baselines,
    BASELINES
)


class TestHardwareEfficientAnsatz:

    @pytest.mark.parametrize("num_qubits", [4, 6, 8])
    def test_qubit_count(self, num_qubits):
        circuit = hardware_efficient_ansatz(num_qubits)
        assert circuit.num_qubits == num_qubits

    @pytest.mark.parametrize("num_qubits", [4, 6, 8])
    def test_has_parameters(self, num_qubits):
        circuit = hardware_efficient_ansatz(num_qubits)
        assert len(circuit.parameters) > 0

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_depth_parameter(self, depth):
        circuit = hardware_efficient_ansatz(4, depth=depth)
        assert circuit.depth() > 0

    def test_has_entanglement(self):
        circuit = hardware_efficient_ansatz(4)
        ops = circuit.count_ops()
        assert 'cx' in ops


class TestEfficientSU2Ansatz:

    @pytest.mark.parametrize("num_qubits", [4, 6, 8])
    def test_qubit_count(self, num_qubits):
        circuit = efficient_su2_ansatz(num_qubits)
        assert circuit.num_qubits == num_qubits

    @pytest.mark.parametrize("num_qubits", [4, 6, 8])
    def test_has_parameters(self, num_qubits):
        circuit = efficient_su2_ansatz(num_qubits)
        assert len(circuit.parameters) > 0


class TestCustomHEA:

    @pytest.mark.parametrize("num_qubits", [4, 6, 8])
    def test_qubit_count(self, num_qubits):
        circuit = custom_hea(num_qubits)
        assert circuit.num_qubits == num_qubits

    @pytest.mark.parametrize("num_qubits", [4, 6, 8])
    def test_has_parameters(self, num_qubits):
        circuit = custom_hea(num_qubits)
        assert len(circuit.parameters) > 0

    def test_parameter_count(self):
        # 2 layers: 2*(2*n) + n = 5n parameters for n qubits
        circuit = custom_hea(4, layers=2)
        expected = 2 * (2 * 4) + 4  # 20 parameters
        assert len(circuit.parameters) == expected


class TestRandomAnsatz:

    def test_deterministic(self):
        """Same seed should give same circuit."""
        c1 = random_ansatz(4)
        c2 = random_ansatz(4)
        assert str(c1) == str(c2)

    def test_has_parameters(self):
        circuit = random_ansatz(4)
        assert len(circuit.parameters) == 16


class TestBaselineRegistry:

    def test_all_baselines_registered(self):
        assert len(BASELINES) == 5

    @pytest.mark.parametrize("name", list(BASELINES.keys()))
    def test_get_baseline(self, name):
        circuit = get_baseline(name, 4)
        assert circuit.num_qubits == 4

    def test_invalid_baseline_raises(self):
        with pytest.raises(ValueError):
            get_baseline("nonexistent", 4)

    def test_get_all_baselines(self):
        baselines = get_all_baselines(4)
        assert len(baselines) == len(BASELINES)
        for circuit in baselines.values():
            assert circuit.num_qubits == 4

    @pytest.mark.parametrize("baseline_name", ["hea_2layer", "hea_3layer", "su2_2layer", "custom_hea"])
    def test_baselines_have_entanglement(self, baseline_name):
        circuit = get_baseline(baseline_name, 4)
        ops = circuit.count_ops()
        assert 'cx' in ops or 'cz' in ops
```

---

## Step 5.1 Verification Checklist

- [ ] `src/evaluation/baselines.py` created
- [ ] All 5 baseline functions implemented
- [ ] BASELINES registry complete with descriptions
- [ ] `get_baseline()` works for all names
- [ ] `get_all_baselines()` returns all circuits
- [ ] All baselines create valid circuits for 4, 6, 8 qubits
- [ ] All tests pass: `pytest tests/test_baselines.py -v` (minimum 15 tests)

**DO NOT PROCEED TO STEP 5.2 UNTIL ALL 7 CHECKS PASS**

---

## Step 5.2: Metrics Module

**Status**: PENDING

**Prerequisites**: Step 5.1 complete

---

### Create Metrics Module

```python
# src/evaluation/metrics.py

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from src.quantum.executor import run_vqe
from src.quantum.barren_plateau import detect_barren_plateau


@dataclass
class CircuitMetrics:
    """Complete metrics for a single circuit evaluation."""

    # Identity
    name: str
    num_qubits: int

    # Circuit structure
    depth: int
    gate_count: int
    param_count: int
    two_qubit_gates: int

    # Performance
    energy: float
    energy_error: float
    relative_error_percent: float

    # Quality
    gradient_variance: float
    trainability: str  # "high", "medium", "low", "very_low"

    # Optimization
    vqe_iterations: int
    converged: bool

    # Optional (hardware)
    fidelity: Optional[float] = None
    hardware_energy: Optional[float] = None
    hardware_std: Optional[float] = None


def compute_circuit_metrics(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    exact_energy: float,
    name: str = "unnamed",
    exact_state: Optional[np.ndarray] = None,
    run_hardware: bool = False
) -> CircuitMetrics:
    """
    Compute all metrics for a circuit.

    Args:
        circuit: Parameterized quantum circuit
        hamiltonian: Observable for energy measurement
        exact_energy: Known exact ground state energy
        name: Circuit identifier
        exact_state: Optional exact ground state for fidelity
        run_hardware: Whether to run on IBM hardware

    Returns:
        CircuitMetrics with all computed values
    """

    # Circuit structure metrics
    num_qubits = circuit.num_qubits
    depth = circuit.depth()
    ops = circuit.count_ops()
    gate_count = sum(ops.values())
    param_count = len(circuit.parameters)
    two_qubit_gates = ops.get('cx', 0) + ops.get('cz', 0) + ops.get('swap', 0)

    # Barren plateau check
    bp_result = detect_barren_plateau(circuit, hamiltonian, num_samples=50)

    # Run VQE optimization
    vqe_result = run_vqe(circuit, hamiltonian, max_iterations=200)

    # Calculate energy errors
    energy = vqe_result.energy
    energy_error = abs(energy - exact_energy)
    relative_error = (energy_error / abs(exact_energy)) * 100 if exact_energy != 0 else 0

    # Fidelity (if exact state available)
    fidelity = None
    if exact_state is not None and vqe_result.final_state is not None:
        try:
            fidelity = float(state_fidelity(vqe_result.final_state, exact_state))
        except Exception:
            pass

    # Hardware metrics placeholder
    hw_energy, hw_std = None, None
    if run_hardware:
        # Implemented in Phase 6
        pass

    return CircuitMetrics(
        name=name,
        num_qubits=num_qubits,
        depth=depth,
        gate_count=gate_count,
        param_count=param_count,
        two_qubit_gates=two_qubit_gates,
        energy=energy,
        energy_error=energy_error,
        relative_error_percent=relative_error,
        gradient_variance=bp_result.gradient_variance,
        trainability=bp_result.trainability,
        vqe_iterations=vqe_result.iterations,
        converged=vqe_result.converged,
        fidelity=fidelity,
        hardware_energy=hw_energy,
        hardware_std=hw_std
    )


@dataclass
class StatisticalMetrics:
    """Aggregated metrics from multiple trial runs."""

    name: str
    num_trials: int

    # Energy statistics
    mean_energy: float
    std_energy: float
    min_energy: float
    max_energy: float

    # Error statistics
    mean_error: float
    std_error: float
    min_error: float
    max_error: float

    # Convergence
    convergence_rate: float
    mean_iterations: float

    # Raw results for further analysis
    all_results: List[CircuitMetrics]


def compute_statistical_metrics(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    exact_energy: float,
    name: str = "unnamed",
    num_trials: int = 10,
    seeds: Optional[List[int]] = None
) -> StatisticalMetrics:
    """
    Run multiple trials and aggregate statistics.

    Args:
        circuit: Parameterized circuit
        hamiltonian: Observable
        exact_energy: Known exact energy
        name: Circuit identifier
        num_trials: Number of independent trials
        seeds: Random seeds for reproducibility

    Returns:
        StatisticalMetrics with aggregated results
    """

    if seeds is None:
        seeds = list(range(num_trials))

    results = []
    for seed in seeds:
        np.random.seed(seed)
        result = compute_circuit_metrics(
            circuit, hamiltonian, exact_energy, name
        )
        results.append(result)

    energies = [r.energy for r in results]
    errors = [r.energy_error for r in results]
    iterations = [r.vqe_iterations for r in results]

    return StatisticalMetrics(
        name=name,
        num_trials=num_trials,
        mean_energy=float(np.mean(energies)),
        std_energy=float(np.std(energies)),
        min_energy=float(min(energies)),
        max_energy=float(max(energies)),
        mean_error=float(np.mean(errors)),
        std_error=float(np.std(errors)),
        min_error=float(min(errors)),
        max_error=float(max(errors)),
        convergence_rate=sum(r.converged for r in results) / num_trials,
        mean_iterations=float(np.mean(iterations)),
        all_results=results
    )


def format_metrics_table(metrics_list: List[CircuitMetrics]) -> str:
    """
    Format metrics as markdown table.

    Args:
        metrics_list: List of CircuitMetrics to format

    Returns:
        Markdown table string
    """
    header = "| Circuit | Energy | Error | Rel.Err% | Depth | Gates | Params |"
    separator = "|---------|--------|-------|----------|-------|-------|--------|"

    rows = [header, separator]
    for m in metrics_list:
        row = (
            f"| {m.name} | {m.energy:.6f} | {m.energy_error:.6f} | "
            f"{m.relative_error_percent:.2f}% | {m.depth} | {m.gate_count} | {m.param_count} |"
        )
        rows.append(row)

    return "\n".join(rows)


def format_statistical_table(stats_list: List[StatisticalMetrics]) -> str:
    """
    Format statistical metrics as markdown table.

    Args:
        stats_list: List of StatisticalMetrics to format

    Returns:
        Markdown table string
    """
    header = "| Circuit | Mean Error | Std | Min | Max | Convergence |"
    separator = "|---------|------------|-----|-----|-----|-------------|"

    rows = [header, separator]
    for s in stats_list:
        row = (
            f"| {s.name} | {s.mean_error:.6f} | {s.std_error:.6f} | "
            f"{s.min_error:.6f} | {s.max_error:.6f} | {s.convergence_rate:.0%} |"
        )
        rows.append(row)

    return "\n".join(rows)
```

---

### Create Tests

```python
# tests/test_metrics.py

import pytest
import numpy as np
from src.evaluation.metrics import (
    compute_circuit_metrics,
    compute_statistical_metrics,
    format_metrics_table,
    format_statistical_table,
    CircuitMetrics,
    StatisticalMetrics
)
from src.evaluation.baselines import hardware_efficient_ansatz
from src.quantum.hamiltonians import xy_chain


class TestCircuitMetrics:

    @pytest.fixture
    def setup(self):
        ham_result = xy_chain(4)
        circuit = hardware_efficient_ansatz(4, depth=2)
        return circuit, ham_result.operator, ham_result.exact_energy

    def test_compute_circuit_metrics(self, setup):
        circuit, ham, exact = setup
        metrics = compute_circuit_metrics(circuit, ham, exact, name="test")

        assert metrics.name == "test"
        assert metrics.num_qubits == 4
        assert metrics.depth > 0
        assert metrics.gate_count > 0
        assert metrics.param_count > 0
        assert metrics.energy_error >= 0
        assert metrics.relative_error_percent >= 0
        assert metrics.trainability in ["high", "medium", "low", "very_low"]

    def test_metrics_dataclass_fields(self):
        required_fields = [
            'name', 'num_qubits', 'depth', 'gate_count', 'param_count',
            'two_qubit_gates', 'energy', 'energy_error', 'relative_error_percent',
            'gradient_variance', 'trainability', 'vqe_iterations', 'converged'
        ]
        actual_fields = [f.name for f in CircuitMetrics.__dataclass_fields__.values()]
        for field in required_fields:
            assert field in actual_fields


class TestStatisticalMetrics:

    @pytest.fixture
    def setup(self):
        ham_result = xy_chain(4)
        circuit = hardware_efficient_ansatz(4, depth=1)  # Simpler for speed
        return circuit, ham_result.operator, ham_result.exact_energy

    def test_statistical_metrics(self, setup):
        circuit, ham, exact = setup
        stats = compute_statistical_metrics(
            circuit, ham, exact, name="test", num_trials=3
        )

        assert stats.num_trials == 3
        assert len(stats.all_results) == 3
        assert stats.mean_error >= 0
        assert 0 <= stats.convergence_rate <= 1

    def test_stats_consistency(self, setup):
        circuit, ham, exact = setup
        stats = compute_statistical_metrics(
            circuit, ham, exact, name="test", num_trials=3
        )

        # Mean should be between min and max
        assert stats.min_error <= stats.mean_error <= stats.max_error


class TestFormatting:

    def test_format_metrics_table(self):
        metrics = CircuitMetrics(
            name="test", num_qubits=4, depth=5, gate_count=20,
            param_count=16, two_qubit_gates=8, energy=-1.5,
            energy_error=0.01, relative_error_percent=0.5,
            gradient_variance=0.001, trainability="high",
            vqe_iterations=100, converged=True
        )

        table = format_metrics_table([metrics])
        assert "test" in table
        assert "Energy" in table
        assert "|" in table

    def test_format_statistical_table(self):
        stats = StatisticalMetrics(
            name="test", num_trials=5,
            mean_energy=-1.5, std_energy=0.01, min_energy=-1.52, max_energy=-1.48,
            mean_error=0.02, std_error=0.01, min_error=0.01, max_error=0.03,
            convergence_rate=0.8, mean_iterations=100.0,
            all_results=[]
        )

        table = format_statistical_table([stats])
        assert "test" in table
        assert "Mean Error" in table
```

---

## Step 5.2 Verification Checklist

- [ ] `src/evaluation/metrics.py` created
- [ ] CircuitMetrics dataclass with 15+ fields
- [ ] StatisticalMetrics dataclass with aggregation
- [ ] `compute_circuit_metrics()` works with HEA on XY chain
- [ ] `compute_statistical_metrics()` runs multiple trials
- [ ] `format_metrics_table()` produces valid markdown
- [ ] `format_statistical_table()` produces valid markdown
- [ ] All tests pass: `pytest tests/test_metrics.py -v` (minimum 10 tests)

**DO NOT PROCEED TO STEP 5.3 UNTIL ALL 8 CHECKS PASS**

---

## Step 5.3: Comparative Evaluation

**Status**: PENDING

**Prerequisites**: Steps 5.1 and 5.2 complete

---

### Create Comparison Module

```python
# src/evaluation/compare.py

from dataclasses import dataclass
from typing import Dict, List
import json
from datetime import datetime
from pathlib import Path
from src.evaluation.metrics import compute_statistical_metrics, StatisticalMetrics
from src.evaluation.baselines import get_all_baselines, BASELINES
from src.quantum.hamiltonians import get_hamiltonian
from qiskit import QuantumCircuit


@dataclass
class ComparisonResult:
    """Result of comparing discovered circuit to baselines."""

    discovered_name: str
    hamiltonian_name: str
    num_qubits: int
    num_trials: int

    # Results
    discovered: StatisticalMetrics
    baselines: Dict[str, StatisticalMetrics]

    # Improvements (positive = discovered is better)
    improvements: Dict[str, Dict[str, float]]

    # Best baseline info
    best_baseline_name: str
    best_baseline_error: float

    # Summary
    beats_all_baselines: bool
    best_improvement_percent: float


def compare_to_baselines(
    discovered_circuit: QuantumCircuit,
    discovered_name: str,
    hamiltonian_name: str,
    num_qubits: int,
    num_trials: int = 10
) -> ComparisonResult:
    """
    Compare discovered circuit against all baselines.

    Args:
        discovered_circuit: The circuit to evaluate
        discovered_name: Name for the discovered circuit
        hamiltonian_name: Hamiltonian type (e.g., "XY_CHAIN")
        num_qubits: Number of qubits
        num_trials: Trials per circuit for statistics

    Returns:
        ComparisonResult with full comparison data
    """

    # Get Hamiltonian
    ham_result = get_hamiltonian(hamiltonian_name, num_qubits)
    hamiltonian = ham_result.operator
    exact_energy = ham_result.exact_energy

    print(f"Evaluating {discovered_name} on {hamiltonian_name} ({num_qubits} qubits)")
    print(f"Exact energy: {exact_energy:.6f}")

    # Evaluate discovered circuit
    print(f"\nEvaluating discovered circuit...")
    discovered = compute_statistical_metrics(
        discovered_circuit,
        hamiltonian,
        exact_energy,
        name=discovered_name,
        num_trials=num_trials
    )
    print(f"  Mean error: {discovered.mean_error:.6f}")

    # Evaluate all baselines
    baselines_circuits = get_all_baselines(num_qubits)
    baselines = {}

    for name, circuit in baselines_circuits.items():
        print(f"Evaluating baseline: {name}...")
        baselines[name] = compute_statistical_metrics(
            circuit,
            hamiltonian,
            exact_energy,
            name=name,
            num_trials=num_trials
        )
        print(f"  Mean error: {baselines[name].mean_error:.6f}")

    # Calculate improvements
    improvements = {}
    for name, baseline_stats in baselines.items():
        baseline_circuit = baselines_circuits[name]

        # Error reduction (positive = discovered better)
        error_reduction = (
            (baseline_stats.mean_error - discovered.mean_error)
            / baseline_stats.mean_error * 100
        ) if baseline_stats.mean_error > 0 else 0

        # Depth reduction
        baseline_depth = baseline_circuit.depth()
        discovered_depth = discovered_circuit.depth()
        depth_reduction = (
            (baseline_depth - discovered_depth)
            / baseline_depth * 100
        ) if baseline_depth > 0 else 0

        # Gate reduction
        baseline_gates = sum(baseline_circuit.count_ops().values())
        discovered_gates = sum(discovered_circuit.count_ops().values())
        gate_reduction = (
            (baseline_gates - discovered_gates)
            / baseline_gates * 100
        ) if baseline_gates > 0 else 0

        improvements[name] = {
            "error_reduction_percent": error_reduction,
            "depth_reduction_percent": depth_reduction,
            "gate_reduction_percent": gate_reduction
        }

    # Find best baseline
    best_baseline_name = min(baselines.keys(), key=lambda k: baselines[k].mean_error)
    best_baseline_error = baselines[best_baseline_name].mean_error

    # Check if discovered beats all baselines
    beats_all = all(
        discovered.mean_error < b.mean_error
        for b in baselines.values()
    )

    best_improvement = max(
        imp["error_reduction_percent"]
        for imp in improvements.values()
    )

    return ComparisonResult(
        discovered_name=discovered_name,
        hamiltonian_name=hamiltonian_name,
        num_qubits=num_qubits,
        num_trials=num_trials,
        discovered=discovered,
        baselines=baselines,
        improvements=improvements,
        best_baseline_name=best_baseline_name,
        best_baseline_error=best_baseline_error,
        beats_all_baselines=beats_all,
        best_improvement_percent=best_improvement
    )


def format_comparison_report(result: ComparisonResult) -> str:
    """Format comparison as markdown report."""

    lines = [
        f"# Comparison Report: {result.discovered_name}",
        "",
        f"**Hamiltonian**: {result.hamiltonian_name}",
        f"**Qubits**: {result.num_qubits}",
        f"**Trials per circuit**: {result.num_trials}",
        "",
        "## Energy Error Results",
        "",
        "| Circuit | Mean Error | Std Error | Convergence |",
        "|---------|------------|-----------|-------------|",
    ]

    # Discovered first (bold)
    d = result.discovered
    lines.append(
        f"| **{d.name}** | **{d.mean_error:.6f}** | {d.std_error:.6f} | {d.convergence_rate:.0%} |"
    )

    # Baselines
    for name, b in sorted(result.baselines.items(), key=lambda x: x[1].mean_error):
        lines.append(
            f"| {name} | {b.mean_error:.6f} | {b.std_error:.6f} | {b.convergence_rate:.0%} |"
        )

    lines.extend([
        "",
        "## Improvements vs Baselines",
        "",
        "| Baseline | Error Reduction | Depth Reduction | Gate Reduction |",
        "|----------|-----------------|-----------------|----------------|",
    ])

    for name, imp in result.improvements.items():
        sign_err = "+" if imp['error_reduction_percent'] > 0 else ""
        sign_dep = "+" if imp['depth_reduction_percent'] > 0 else ""
        sign_gat = "+" if imp['gate_reduction_percent'] > 0 else ""
        lines.append(
            f"| {name} | {sign_err}{imp['error_reduction_percent']:.1f}% | "
            f"{sign_dep}{imp['depth_reduction_percent']:.1f}% | "
            f"{sign_gat}{imp['gate_reduction_percent']:.1f}% |"
        )

    lines.extend([
        "",
        "## Summary",
        "",
        f"- **Beats all baselines**: {'YES' if result.beats_all_baselines else 'NO'}",
        f"- **Best baseline**: {result.best_baseline_name} (error: {result.best_baseline_error:.6f})",
        f"- **Best improvement**: {result.best_improvement_percent:.1f}%",
    ])

    return "\n".join(lines)


def save_comparison(result: ComparisonResult, path: str):
    """Save comparison result to JSON."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "discovered_name": result.discovered_name,
        "hamiltonian_name": result.hamiltonian_name,
        "num_qubits": result.num_qubits,
        "num_trials": result.num_trials,
        "discovered": {
            "mean_error": result.discovered.mean_error,
            "std_error": result.discovered.std_error,
            "convergence_rate": result.discovered.convergence_rate
        },
        "baselines": {
            name: {
                "mean_error": b.mean_error,
                "std_error": b.std_error,
                "convergence_rate": b.convergence_rate
            }
            for name, b in result.baselines.items()
        },
        "improvements": result.improvements,
        "beats_all_baselines": result.beats_all_baselines,
        "best_improvement_percent": result.best_improvement_percent,
        "best_baseline_name": result.best_baseline_name
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved comparison to {path}")
```

---

### Create Tests

```python
# tests/test_compare.py

import pytest
from src.evaluation.compare import (
    compare_to_baselines,
    format_comparison_report,
    save_comparison,
    ComparisonResult
)
from src.evaluation.baselines import hardware_efficient_ansatz


class TestComparison:

    def test_compare_to_baselines(self):
        """Test full comparison pipeline."""
        # Use HEA as "discovered" to verify pipeline
        discovered = hardware_efficient_ansatz(4, depth=2)

        result = compare_to_baselines(
            discovered_circuit=discovered,
            discovered_name="test_circuit",
            hamiltonian_name="XY_CHAIN",
            num_qubits=4,
            num_trials=2  # Low for speed
        )

        assert isinstance(result, ComparisonResult)
        assert result.discovered_name == "test_circuit"
        assert result.num_qubits == 4
        assert len(result.baselines) == 5
        assert len(result.improvements) == 5

    def test_format_comparison_report(self):
        # Create minimal result for formatting test
        from src.evaluation.metrics import StatisticalMetrics

        result = ComparisonResult(
            discovered_name="test",
            hamiltonian_name="XY_CHAIN",
            num_qubits=4,
            num_trials=3,
            discovered=StatisticalMetrics(
                name="test", num_trials=3,
                mean_energy=-1.5, std_energy=0.01, min_energy=-1.52, max_energy=-1.48,
                mean_error=0.02, std_error=0.01, min_error=0.01, max_error=0.03,
                convergence_rate=1.0, mean_iterations=100.0, all_results=[]
            ),
            baselines={
                "hea": StatisticalMetrics(
                    name="hea", num_trials=3,
                    mean_energy=-1.4, std_energy=0.02, min_energy=-1.45, max_energy=-1.35,
                    mean_error=0.12, std_error=0.02, min_error=0.1, max_error=0.14,
                    convergence_rate=0.8, mean_iterations=150.0, all_results=[]
                )
            },
            improvements={"hea": {"error_reduction_percent": 83.3, "depth_reduction_percent": 0, "gate_reduction_percent": 0}},
            best_baseline_name="hea",
            best_baseline_error=0.12,
            beats_all_baselines=True,
            best_improvement_percent=83.3
        )

        report = format_comparison_report(result)
        assert "test" in report
        assert "YES" in report  # beats_all_baselines
        assert "83.3%" in report

    def test_save_comparison(self, tmp_path):
        from src.evaluation.metrics import StatisticalMetrics

        result = ComparisonResult(
            discovered_name="test", hamiltonian_name="XY_CHAIN",
            num_qubits=4, num_trials=3,
            discovered=StatisticalMetrics(
                name="test", num_trials=3,
                mean_energy=-1.5, std_energy=0.01, min_energy=-1.52, max_energy=-1.48,
                mean_error=0.02, std_error=0.01, min_error=0.01, max_error=0.03,
                convergence_rate=1.0, mean_iterations=100.0, all_results=[]
            ),
            baselines={},
            improvements={},
            best_baseline_name="none",
            best_baseline_error=0,
            beats_all_baselines=False,
            best_improvement_percent=0
        )

        path = tmp_path / "comparison.json"
        save_comparison(result, str(path))

        assert path.exists()
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["discovered_name"] == "test"
```

---

## Step 5.3 Verification Checklist

- [ ] `src/evaluation/compare.py` created
- [ ] ComparisonResult dataclass complete
- [ ] `compare_to_baselines()` runs end-to-end
- [ ] `format_comparison_report()` produces valid markdown
- [ ] `save_comparison()` creates valid JSON
- [ ] Pipeline works with dummy "discovered" circuit
- [ ] All tests pass: `pytest tests/test_compare.py -v` (minimum 12 tests)

**DO NOT PROCEED TO STEP 5.4 UNTIL ALL 7 CHECKS PASS**

---

## Step 5.4: Statistical Tests

**Status**: PENDING

**Prerequisites**: Step 5.3 complete

---

### Create Statistics Module

```python
# src/evaluation/statistics.py

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy import stats


@dataclass
class SignificanceResult:
    """Result of statistical significance test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool  # at alpha=0.05
    effect_size: float
    effect_interpretation: str  # "small", "medium", "large"
    confidence_interval: Tuple[float, float]


def test_improvement_significance(
    discovered_errors: List[float],
    baseline_errors: List[float],
    alpha: float = 0.05
) -> SignificanceResult:
    """
    Test if discovered circuit significantly beats baseline.
    Uses paired t-test (same random seeds across trials).

    Args:
        discovered_errors: Error values from discovered circuit trials
        baseline_errors: Error values from baseline circuit trials
        alpha: Significance level

    Returns:
        SignificanceResult with test statistics
    """

    if len(discovered_errors) != len(baseline_errors):
        raise ValueError("Need same number of paired observations")

    if len(discovered_errors) < 2:
        raise ValueError("Need at least 2 observations")

    # Paired t-test (one-tailed: baseline > discovered)
    t_stat, p_value_two = stats.ttest_rel(baseline_errors, discovered_errors)
    p_value = p_value_two / 2 if t_stat > 0 else 1 - p_value_two / 2

    # Cohen's d effect size for paired samples
    diff = np.array(baseline_errors) - np.array(discovered_errors)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

    # Interpret effect size (Cohen's conventions)
    if abs(cohens_d) >= 0.8:
        effect = "large"
    elif abs(cohens_d) >= 0.5:
        effect = "medium"
    else:
        effect = "small"

    # Confidence interval for mean difference
    mean_diff = np.mean(diff)
    se = stats.sem(diff)
    df = len(diff) - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)

    return SignificanceResult(
        test_name="Paired t-test (one-tailed)",
        statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        effect_size=cohens_d,
        effect_interpretation=effect,
        confidence_interval=ci
    )


def wilcoxon_test(
    discovered_errors: List[float],
    baseline_errors: List[float],
    alpha: float = 0.05
) -> SignificanceResult:
    """
    Non-parametric alternative to paired t-test.
    Use when normality assumption is violated.

    Args:
        discovered_errors: Error values from discovered circuit
        baseline_errors: Error values from baseline circuit
        alpha: Significance level

    Returns:
        SignificanceResult with test statistics
    """

    diff = np.array(baseline_errors) - np.array(discovered_errors)

    # Remove zero differences (ties with zero)
    diff_nonzero = diff[diff != 0]

    if len(diff_nonzero) < 1:
        return SignificanceResult(
            test_name="Wilcoxon signed-rank",
            statistic=0,
            p_value=1.0,
            significant=False,
            effect_size=0,
            effect_interpretation="none",
            confidence_interval=(0, 0)
        )

    stat, p_value = stats.wilcoxon(
        diff_nonzero,
        alternative='greater'  # One-tailed
    )

    # Effect size: r = Z / sqrt(N)
    n = len(diff_nonzero)
    z = stats.norm.ppf(1 - p_value)
    r = abs(z) / np.sqrt(n) if n > 0 else 0

    if r >= 0.5:
        effect = "large"
    elif r >= 0.3:
        effect = "medium"
    else:
        effect = "small"

    return SignificanceResult(
        test_name="Wilcoxon signed-rank (one-tailed)",
        statistic=stat,
        p_value=p_value,
        significant=p_value < alpha,
        effect_size=r,
        effect_interpretation=effect,
        confidence_interval=(0, 0)  # Not directly applicable
    )


def bootstrap_confidence_interval(
    data: List[float],
    statistic_fn=np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for any statistic.

    Args:
        data: Sample data
        statistic_fn: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower, upper) bounds
    """

    np.random.seed(seed)
    data = np.array(data)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_fn(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return (float(lower), float(upper))


def format_significance_report(
    results: List[Tuple[str, SignificanceResult]]
) -> str:
    """
    Format significance tests as markdown.

    Args:
        results: List of (baseline_name, SignificanceResult) tuples

    Returns:
        Markdown formatted report
    """

    lines = [
        "## Statistical Significance Tests",
        "",
        "| Comparison | Test | Statistic | p-value | Significant | Effect Size |",
        "|------------|------|-----------|---------|-------------|-------------|",
    ]

    for baseline_name, result in results:
        sig_marker = "Yes" if result.significant else "No"
        lines.append(
            f"| vs {baseline_name} | {result.test_name} | "
            f"{result.statistic:.4f} | {result.p_value:.4f} | "
            f"{sig_marker} | {result.effect_interpretation} (d={result.effect_size:.2f}) |"
        )

    lines.extend([
        "",
        "**Interpretation**:",
        "- p < 0.05 indicates statistically significant improvement",
        "- Effect size: small (d<0.5), medium (0.5<=d<0.8), large (d>=0.8)"
    ])

    return "\n".join(lines)
```

---

### Create Tests

```python
# tests/test_statistics.py

import pytest
import numpy as np
from src.evaluation.statistics import (
    test_improvement_significance,
    wilcoxon_test,
    bootstrap_confidence_interval,
    format_significance_report,
    SignificanceResult
)


class TestPairedTTest:

    def test_significant_improvement(self):
        # Discovered clearly better
        discovered = [0.01, 0.02, 0.01, 0.02, 0.01]
        baseline = [0.10, 0.12, 0.11, 0.13, 0.10]

        result = test_improvement_significance(discovered, baseline)

        assert result.significant is True
        assert result.p_value < 0.05
        assert result.effect_interpretation in ["medium", "large"]

    def test_no_improvement(self):
        # Same values
        discovered = [0.10, 0.11, 0.10, 0.11, 0.10]
        baseline = [0.10, 0.11, 0.10, 0.11, 0.10]

        result = test_improvement_significance(discovered, baseline)

        assert result.significant is False
        assert result.effect_size == 0

    def test_confidence_interval(self):
        discovered = [0.01, 0.02, 0.01, 0.02, 0.01]
        baseline = [0.10, 0.12, 0.11, 0.13, 0.10]

        result = test_improvement_significance(discovered, baseline)

        ci = result.confidence_interval
        assert ci[0] < ci[1]  # Valid interval
        assert ci[0] > 0  # Improvement is positive

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError):
            test_improvement_significance([0.1, 0.2], [0.3])

    def test_single_observation_raises(self):
        with pytest.raises(ValueError):
            test_improvement_significance([0.1], [0.2])


class TestWilcoxonTest:

    def test_significant_improvement(self):
        discovered = [0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02]
        baseline = [0.10, 0.12, 0.11, 0.13, 0.10, 0.12, 0.11, 0.13]

        result = wilcoxon_test(discovered, baseline)

        assert result.significant is True
        assert result.p_value < 0.05

    def test_identical_values(self):
        values = [0.1] * 10
        result = wilcoxon_test(values, values)

        assert result.significant is False


class TestBootstrap:

    def test_bootstrap_ci(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower, upper = bootstrap_confidence_interval(data)

        assert lower < upper
        assert lower <= np.mean(data) <= upper

    def test_bootstrap_deterministic(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci1 = bootstrap_confidence_interval(data, seed=42)
        ci2 = bootstrap_confidence_interval(data, seed=42)

        assert ci1 == ci2


class TestFormatting:

    def test_format_significance_report(self):
        results = [
            ("hea", SignificanceResult(
                test_name="Paired t-test",
                statistic=5.0,
                p_value=0.001,
                significant=True,
                effect_size=1.5,
                effect_interpretation="large",
                confidence_interval=(0.05, 0.10)
            ))
        ]

        report = format_significance_report(results)

        assert "hea" in report
        assert "Yes" in report
        assert "large" in report
        assert "0.001" in report
```

---

## Step 5.4 Verification Checklist

- [ ] `src/evaluation/statistics.py` created
- [ ] `test_improvement_significance()` works with paired data
- [ ] `wilcoxon_test()` works as non-parametric alternative
- [ ] `bootstrap_confidence_interval()` produces valid intervals
- [ ] `format_significance_report()` produces valid markdown
- [ ] All tests pass: `pytest tests/test_statistics.py -v` (minimum 10 tests)
- [ ] Tests include edge cases (identical data, single observation)

**DO NOT PROCEED TO PHASE 6 UNTIL ALL 7 CHECKS PASS**

---

## After Completion

Update `00_OVERVIEW.md`:
- Change Phase 5 progress to 4/4
- Mark all Phase 5 steps as DONE

Next: Proceed to `06_HARDWARE.md` (Step 6.1)

---

*Phase 5 provides the evaluation framework for measuring discovery success.*
