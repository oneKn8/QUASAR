# QuantumMind Guideline: Phase 6 - Hardware Validation

> Steps 6.1-6.3: IBM Hardware Runner, Error Mitigation, Noise Analysis

---

## Step 6.1: IBM Hardware Runner

**Status**: PENDING

**Prerequisites**: Phase 1 complete (IBM token set), Phase 5 complete

---

### Create Hardware Module

```python
# src/quantum/hardware.py

from dataclasses import dataclass
from typing import List, Optional
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import time


@dataclass
class HardwareResult:
    """Result from IBM quantum hardware execution."""

    backend_name: str
    energy: float
    std_error: float
    shots: int
    job_id: str
    execution_time_seconds: float
    transpiled_depth: int
    transpiled_gates: int


class IBMHardwareRunner:
    """Interface to IBM Quantum hardware for circuit execution."""

    def __init__(
        self,
        backend_name: str = "ibm_sherbrooke",
        resilience_level: int = 2
    ):
        """
        Initialize hardware runner.

        Args:
            backend_name: IBM backend to use
            resilience_level: Error mitigation level (0-3)
        """
        self.service = QiskitRuntimeService()
        self.backend = self.service.backend(backend_name)
        self.backend_name = backend_name
        self.resilience_level = resilience_level

        # Create transpilation pass manager
        self.pm = generate_preset_pass_manager(
            backend=self.backend,
            optimization_level=3
        )

        print(f"Initialized hardware runner for {backend_name}")
        print(f"  Qubits: {self.backend.num_qubits}")
        print(f"  Resilience level: {resilience_level}")

    def get_backend_info(self) -> dict:
        """Get backend information."""
        return {
            "name": self.backend.name,
            "num_qubits": self.backend.num_qubits,
            "basis_gates": list(self.backend.operation_names),
            "max_circuits": self.backend.max_circuits,
            "status": self.backend.status().status_msg,
            "pending_jobs": self.backend.status().pending_jobs,
        }

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Transpile circuit for hardware.

        Args:
            circuit: Input circuit

        Returns:
            Transpiled circuit for target backend
        """
        return self.pm.run(circuit)

    def estimate_energy(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        parameter_values: List[float],
        shots: int = 4000
    ) -> HardwareResult:
        """
        Run energy estimation on hardware.

        Args:
            circuit: Parameterized circuit
            hamiltonian: Observable to measure
            parameter_values: Optimal parameter values from VQE
            shots: Number of shots per circuit

        Returns:
            HardwareResult with energy and metadata
        """

        # Bind parameters
        bound_circuit = circuit.assign_parameters(parameter_values)

        # Transpile
        transpiled = self.transpile(bound_circuit)
        transpiled_depth = transpiled.depth()
        transpiled_gates = sum(transpiled.count_ops().values())

        print(f"Transpiled circuit: depth={transpiled_depth}, gates={transpiled_gates}")

        start_time = time.time()

        with Session(backend=self.backend) as session:
            estimator = Estimator(session=session)
            estimator.options.resilience_level = self.resilience_level
            estimator.options.default_shots = shots

            print(f"Submitting job with {shots} shots...")
            job = estimator.run(
                circuits=[transpiled],
                observables=[hamiltonian]
            )

            print(f"Job ID: {job.job_id()}")
            print("Waiting for results...")

            result = job.result()

        execution_time = time.time() - start_time

        energy = float(result.values[0])
        std_error = float(result.metadata[0].get('std', 0.0))

        print(f"Energy: {energy:.6f} +/- {std_error:.6f}")
        print(f"Execution time: {execution_time:.1f}s")

        return HardwareResult(
            backend_name=self.backend_name,
            energy=energy,
            std_error=std_error,
            shots=shots,
            job_id=job.job_id(),
            execution_time_seconds=execution_time,
            transpiled_depth=transpiled_depth,
            transpiled_gates=transpiled_gates
        )

    def run_multiple(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        parameter_values: List[float],
        num_runs: int = 5,
        shots: int = 4000
    ) -> List[HardwareResult]:
        """
        Run multiple estimations for statistics.

        Args:
            circuit: Parameterized circuit
            hamiltonian: Observable
            parameter_values: Parameter values
            num_runs: Number of independent runs
            shots: Shots per run

        Returns:
            List of HardwareResult
        """
        results = []

        for i in range(num_runs):
            print(f"\n--- Hardware run {i+1}/{num_runs} ---")
            result = self.estimate_energy(
                circuit, hamiltonian, parameter_values, shots
            )
            results.append(result)

        # Print summary
        energies = [r.energy for r in results]
        import numpy as np
        print(f"\n=== Summary ===")
        print(f"Mean energy: {np.mean(energies):.6f} +/- {np.std(energies):.6f}")
        print(f"Min: {min(energies):.6f}, Max: {max(energies):.6f}")

        return results


def format_hardware_results(results: List[HardwareResult], exact_energy: float) -> str:
    """Format hardware results as markdown table."""

    import numpy as np

    lines = [
        "## Hardware Results",
        "",
        f"**Exact energy**: {exact_energy:.6f}",
        "",
        "| Run | Energy | Std Error | Error vs Exact | Job ID |",
        "|-----|--------|-----------|----------------|--------|",
    ]

    for i, r in enumerate(results, 1):
        error = abs(r.energy - exact_energy)
        lines.append(
            f"| {i} | {r.energy:.6f} | {r.std_error:.6f} | {error:.6f} | {r.job_id[:8]}... |"
        )

    energies = [r.energy for r in results]
    lines.extend([
        "",
        "**Statistics**:",
        f"- Mean energy: {np.mean(energies):.6f}",
        f"- Std deviation: {np.std(energies):.6f}",
        f"- Mean error vs exact: {np.mean([abs(e - exact_energy) for e in energies]):.6f}",
    ])

    return "\n".join(lines)
```

---

### Create Tests

```python
# tests/test_hardware.py

import pytest
import os
from src.quantum.hardware import IBMHardwareRunner, HardwareResult, format_hardware_results
from src.quantum.hamiltonians import xy_chain
from src.evaluation.baselines import hardware_efficient_ansatz

# Skip if no IBM token
HAS_IBM_TOKEN = os.environ.get('IBM_QUANTUM_TOKEN') is not None


class TestHardwareRunnerLocal:
    """Tests that don't require IBM connection."""

    def test_hardware_result_dataclass(self):
        result = HardwareResult(
            backend_name="test",
            energy=-1.5,
            std_error=0.01,
            shots=1000,
            job_id="abc123",
            execution_time_seconds=10.0,
            transpiled_depth=50,
            transpiled_gates=100
        )
        assert result.energy == -1.5
        assert result.backend_name == "test"

    def test_format_hardware_results(self):
        results = [
            HardwareResult(
                backend_name="test", energy=-1.5, std_error=0.01,
                shots=1000, job_id="abc123", execution_time_seconds=10.0,
                transpiled_depth=50, transpiled_gates=100
            )
        ]

        output = format_hardware_results(results, exact_energy=-1.52)
        assert "-1.5" in output
        assert "Run" in output


@pytest.mark.skipif(not HAS_IBM_TOKEN, reason="No IBM token")
class TestIBMHardware:
    """Tests requiring IBM connection."""

    def test_backend_connection(self):
        runner = IBMHardwareRunner()
        info = runner.get_backend_info()

        assert info["name"] == "ibm_sherbrooke"
        assert info["num_qubits"] >= 127

    def test_transpilation(self):
        runner = IBMHardwareRunner()
        circuit = hardware_efficient_ansatz(4)

        transpiled = runner.transpile(circuit)

        assert transpiled.num_qubits <= runner.backend.num_qubits

    @pytest.mark.slow
    def test_single_estimation(self):
        """Full hardware test - runs actual job."""
        runner = IBMHardwareRunner()
        circuit = hardware_efficient_ansatz(4)
        ham_result = xy_chain(4)

        import numpy as np
        params = np.random.uniform(0, 2*np.pi, len(circuit.parameters))

        result = runner.estimate_energy(
            circuit,
            ham_result.operator,
            list(params),
            shots=100  # Low for testing
        )

        assert isinstance(result, HardwareResult)
        assert result.job_id is not None
        assert result.energy != 0
```

---

## Step 6.1 Verification Checklist

- [ ] `src/quantum/hardware.py` created
- [ ] IBMHardwareRunner connects to ibm_sherbrooke
- [ ] `get_backend_info()` returns valid data
- [ ] `transpile()` produces hardware-compatible circuit
- [ ] `estimate_energy()` returns HardwareResult
- [ ] `run_multiple()` executes without error
- [ ] `format_hardware_results()` produces valid markdown
- [ ] All tests pass (skip if no token)

**DO NOT PROCEED TO STEP 6.2 UNTIL ALL 8 CHECKS PASS**

---

## Step 6.2: Error Mitigation Configuration

**Status**: PENDING

**Prerequisites**: Step 6.1 complete

---

### Add to Hardware Module

```python
# Add to src/quantum/hardware.py

from enum import Enum
from dataclasses import field


class ResilienceLevel(Enum):
    """IBM Quantum error mitigation levels."""
    NONE = 0        # No mitigation
    TWIRLING = 1    # Pauli twirling
    ZNE = 2         # Zero-noise extrapolation
    PEC = 3         # Probabilistic error cancellation


@dataclass
class ErrorMitigationConfig:
    """Configuration for error mitigation."""

    resilience_level: ResilienceLevel
    zne_noise_factors: List[float] = field(default_factory=lambda: [1, 3, 5])
    twirling_num_randomizations: int = 32

    def to_estimator_options(self) -> dict:
        """Convert to Estimator options dict."""
        options = {
            "resilience_level": self.resilience_level.value
        }

        if self.resilience_level == ResilienceLevel.ZNE:
            options["resilience"] = {
                "zne": {
                    "noise_factors": self.zne_noise_factors
                }
            }

        return options

    @property
    def description(self) -> str:
        """Human-readable description."""
        descs = {
            ResilienceLevel.NONE: "No error mitigation",
            ResilienceLevel.TWIRLING: "Pauli twirling for coherent error suppression",
            ResilienceLevel.ZNE: f"Zero-noise extrapolation with factors {self.zne_noise_factors}",
            ResilienceLevel.PEC: "Probabilistic error cancellation (highest accuracy, high overhead)"
        }
        return descs[self.resilience_level]


def compare_error_mitigation_levels(
    runner: IBMHardwareRunner,
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    parameter_values: List[float],
    exact_energy: float,
    shots: int = 4000
) -> dict:
    """
    Compare different error mitigation levels on same circuit.

    Args:
        runner: Hardware runner instance
        circuit: Bound circuit to evaluate
        hamiltonian: Observable
        parameter_values: Parameter values
        exact_energy: Known exact energy
        shots: Shots per run

    Returns:
        Dict with results per resilience level
    """

    import numpy as np

    results = {}
    original_level = runner.resilience_level

    for level in [0, 1, 2]:
        print(f"\n--- Testing Resilience Level {level} ---")
        runner.resilience_level = level

        result = runner.estimate_energy(
            circuit, hamiltonian, parameter_values, shots
        )

        error_vs_exact = abs(result.energy - exact_energy)

        results[f"level_{level}"] = {
            "energy": result.energy,
            "std_error": result.std_error,
            "error_vs_exact": error_vs_exact,
            "execution_time": result.execution_time_seconds,
            "job_id": result.job_id
        }

        print(f"Energy: {result.energy:.6f} (error: {error_vs_exact:.6f})")

    # Restore original level
    runner.resilience_level = original_level

    # Print comparison
    print("\n=== Error Mitigation Comparison ===")
    print(f"Exact energy: {exact_energy:.6f}")
    print("-" * 50)

    for level_name, data in results.items():
        print(f"{level_name}: {data['energy']:.6f} (error: {data['error_vs_exact']:.6f})")

    return results


def format_mitigation_comparison(results: dict, exact_energy: float) -> str:
    """Format mitigation comparison as markdown."""

    level_names = {
        "level_0": "None",
        "level_1": "Twirling",
        "level_2": "ZNE"
    }

    lines = [
        "## Error Mitigation Comparison",
        "",
        f"**Exact energy**: {exact_energy:.6f}",
        "",
        "| Level | Name | Energy | Error vs Exact | Time (s) |",
        "|-------|------|--------|----------------|----------|",
    ]

    for level, data in sorted(results.items()):
        name = level_names.get(level, level)
        lines.append(
            f"| {level[-1]} | {name} | {data['energy']:.6f} | "
            f"{data['error_vs_exact']:.6f} | {data['execution_time']:.1f} |"
        )

    # Find best level
    best = min(results.items(), key=lambda x: x[1]['error_vs_exact'])
    lines.extend([
        "",
        f"**Best level**: {level_names.get(best[0], best[0])} (error: {best[1]['error_vs_exact']:.6f})"
    ])

    return "\n".join(lines)
```

---

## Step 6.2 Verification Checklist

- [ ] ResilienceLevel enum defined with 4 levels
- [ ] ErrorMitigationConfig dataclass complete
- [ ] `to_estimator_options()` returns valid options dict
- [ ] `compare_error_mitigation_levels()` tests all levels
- [ ] `format_mitigation_comparison()` produces valid markdown
- [ ] Results documented for each level
- [ ] All tests pass

**DO NOT PROCEED TO STEP 6.3 UNTIL ALL 7 CHECKS PASS**

---

## Step 6.3: Noise Analysis

**Status**: PENDING

**Prerequisites**: Steps 6.1 and 6.2 complete

---

### Create Noise Analysis Module

```python
# src/evaluation/noise.py

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from src.quantum.hardware import IBMHardwareRunner


@dataclass
class NoiseAnalysisResult:
    """Comparison of ideal, noisy sim, and hardware results."""

    circuit_name: str
    num_qubits: int

    # Energies
    exact_energy: float
    ideal_sim_energy: float
    noisy_sim_energy: float
    hardware_energy: float

    # Errors (vs exact)
    ideal_sim_error: float
    noisy_sim_error: float
    hardware_error: float

    # Gaps between methods
    noisy_to_ideal_gap: float
    hw_to_ideal_gap: float
    hw_to_noisy_gap: float

    # Additional info
    hardware_std: float
    transpiled_depth: int
    hardware_job_id: str


def get_hardware_noise_model(backend_name: str = "ibm_sherbrooke") -> NoiseModel:
    """
    Get noise model from real hardware.

    Args:
        backend_name: IBM backend name

    Returns:
        NoiseModel calibrated to backend
    """
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    return NoiseModel.from_backend(backend)


def run_ideal_simulation(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    parameter_values: List[float]
) -> float:
    """
    Run ideal (noiseless) simulation.

    Args:
        circuit: Parameterized circuit
        hamiltonian: Observable
        parameter_values: Parameter values

    Returns:
        Energy expectation value
    """
    from qiskit.primitives import StatevectorEstimator

    bound = circuit.assign_parameters(parameter_values)

    estimator = StatevectorEstimator()
    job = estimator.run([(bound, hamiltonian)])
    result = job.result()

    return float(result[0].data.evs[0])


def run_noisy_simulation(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    parameter_values: List[float],
    noise_model: NoiseModel,
    shots: int = 10000
) -> float:
    """
    Run simulation with hardware noise model.

    Args:
        circuit: Parameterized circuit
        hamiltonian: Observable
        parameter_values: Parameter values
        noise_model: Noise model from hardware
        shots: Number of shots

    Returns:
        Energy expectation value
    """
    from qiskit_aer.primitives import Estimator as AerEstimator

    bound = circuit.assign_parameters(parameter_values)

    estimator = AerEstimator(
        backend_options={"noise_model": noise_model},
        run_options={"shots": shots}
    )

    job = estimator.run([(bound, hamiltonian)])
    result = job.result()

    return float(result[0].data.evs[0])


def analyze_noise(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    parameter_values: List[float],
    exact_energy: float,
    circuit_name: str = "circuit",
    hardware_shots: int = 4000
) -> NoiseAnalysisResult:
    """
    Complete noise analysis: ideal vs noisy sim vs hardware.

    Args:
        circuit: Parameterized circuit
        hamiltonian: Observable
        parameter_values: Optimized parameters
        exact_energy: Known exact ground state energy
        circuit_name: Identifier for the circuit
        hardware_shots: Shots for hardware run

    Returns:
        NoiseAnalysisResult with all comparisons
    """

    num_qubits = circuit.num_qubits
    print(f"\n=== Noise Analysis: {circuit_name} ({num_qubits} qubits) ===")
    print(f"Exact energy: {exact_energy:.6f}")

    # 1. Ideal simulation
    print("\n1. Running ideal simulation...")
    ideal_energy = run_ideal_simulation(circuit, hamiltonian, parameter_values)
    ideal_error = abs(ideal_energy - exact_energy)
    print(f"   Energy: {ideal_energy:.6f} (error: {ideal_error:.6f})")

    # 2. Noisy simulation
    print("\n2. Running noisy simulation...")
    noise_model = get_hardware_noise_model()
    noisy_energy = run_noisy_simulation(circuit, hamiltonian, parameter_values, noise_model)
    noisy_error = abs(noisy_energy - exact_energy)
    print(f"   Energy: {noisy_energy:.6f} (error: {noisy_error:.6f})")

    # 3. Hardware
    print("\n3. Running on hardware...")
    runner = IBMHardwareRunner(resilience_level=2)
    hw_result = runner.estimate_energy(circuit, hamiltonian, parameter_values, hardware_shots)
    hw_error = abs(hw_result.energy - exact_energy)
    print(f"   Energy: {hw_result.energy:.6f} (error: {hw_error:.6f})")

    # Calculate gaps
    noisy_to_ideal = abs(noisy_energy - ideal_energy)
    hw_to_ideal = abs(hw_result.energy - ideal_energy)
    hw_to_noisy = abs(hw_result.energy - noisy_energy)

    result = NoiseAnalysisResult(
        circuit_name=circuit_name,
        num_qubits=num_qubits,
        exact_energy=exact_energy,
        ideal_sim_energy=ideal_energy,
        noisy_sim_energy=noisy_energy,
        hardware_energy=hw_result.energy,
        ideal_sim_error=ideal_error,
        noisy_sim_error=noisy_error,
        hardware_error=hw_error,
        noisy_to_ideal_gap=noisy_to_ideal,
        hw_to_ideal_gap=hw_to_ideal,
        hw_to_noisy_gap=hw_to_noisy,
        hardware_std=hw_result.std_error,
        transpiled_depth=hw_result.transpiled_depth,
        hardware_job_id=hw_result.job_id
    )

    print("\n" + format_noise_report(result))

    return result


def format_noise_report(result: NoiseAnalysisResult) -> str:
    """Format noise analysis as markdown."""

    return f"""## Noise Analysis: {result.circuit_name}

**Circuit**: {result.num_qubits} qubits, transpiled depth: {result.transpiled_depth}

### Energy Comparison

| Condition | Energy | Error vs Exact |
|-----------|--------|----------------|
| **Exact** | {result.exact_energy:.6f} | - |
| Ideal Simulation | {result.ideal_sim_energy:.6f} | {result.ideal_sim_error:.6f} |
| Noisy Simulation | {result.noisy_sim_energy:.6f} | {result.noisy_sim_error:.6f} |
| Hardware | {result.hardware_energy:.6f} +/- {result.hardware_std:.6f} | {result.hardware_error:.6f} |

### Gap Analysis

| Gap | Value | Interpretation |
|-----|-------|----------------|
| Noisy Sim - Ideal | {result.noisy_to_ideal_gap:.6f} | Noise model accuracy |
| Hardware - Ideal | {result.hw_to_ideal_gap:.6f} | Total hardware degradation |
| Hardware - Noisy Sim | {result.hw_to_noisy_gap:.6f} | Simulation fidelity |

### Observations

- Hardware job ID: {result.hardware_job_id}
- Noisy simulation {"closely matches" if result.hw_to_noisy_gap < 0.01 else "differs from"} hardware
- Error mitigation {"effective" if result.hardware_error < result.noisy_sim_error else "limited"} for this circuit
"""
```

---

## Step 6.3 Verification Checklist

- [ ] `src/evaluation/noise.py` created
- [ ] NoiseAnalysisResult dataclass complete with all fields
- [ ] `get_hardware_noise_model()` returns valid NoiseModel
- [ ] `run_ideal_simulation()` produces energy values
- [ ] `run_noisy_simulation()` produces energy values
- [ ] `analyze_noise()` runs complete pipeline (ideal + noisy + hardware)
- [ ] `format_noise_report()` produces valid markdown
- [ ] All tests pass

**DO NOT PROCEED TO PHASE 7 UNTIL ALL 8 CHECKS PASS**

---

## After Completion

Update `00_OVERVIEW.md`:
- Change Phase 6 progress to 3/3
- Mark all Phase 6 steps as DONE

Next: Proceed to `07_DISCOVERY.md` (Step 7.1)

---

*Phase 6 enables validation on real quantum hardware.*
