# Build Plan

## Overview

This document provides step-by-step instructions for building QuantumMind. Follow sequentially. Do not skip steps. Verify each step before proceeding.

---

## Phase 1: Environment Setup

### Step 1.1: Create Project Structure

```bash
# Navigate to quantum-mind directory
cd /home/oneknight/zzzzzzzzzzzzz/quantum-mind

# Create directory structure
mkdir -p src/{agent,quantum,training,evaluation}
mkdir -p data/{raw,processed,results}
mkdir -p models/checkpoints
mkdir -p notebooks
mkdir -p experiments/logs
mkdir -p paper/figures
mkdir -p tests

# Create __init__.py files
touch src/__init__.py
touch src/agent/__init__.py
touch src/quantum/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py

# Verify
find . -type d | head -20
```

**Verification**: All directories exist

### Step 1.2: Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install quantum computing
pip install qiskit qiskit-ibm-runtime qiskit-aer qiskit-nature

# Install ML/training
pip install transformers datasets accelerate
pip install bitsandbytes  # For 4-bit quantization

# Install utilities
pip install numpy scipy matplotlib pandas
pip install pytest black isort

# Save requirements
pip freeze > requirements.txt
```

**Verification**: `python -c "import qiskit; print(qiskit.__version__)"`

### Step 1.3: IBM Quantum Setup

```python
# Test IBM connection
# Save as: test_ibm.py

from qiskit_ibm_runtime import QiskitRuntimeService
import os

# First time setup
token = os.environ.get('IBM_QUANTUM_TOKEN')
if token:
    QiskitRuntimeService.save_account(channel="ibm_quantum", token=token)

# Test connection
service = QiskitRuntimeService()
backends = service.backends()
print(f"Available backends: {len(backends)}")
for b in backends[:5]:
    print(f"  - {b.name}: {b.num_qubits} qubits")
```

**Verification**: Script runs without error, shows available backends

### Step 1.4: Verify Qiskit Installation

```python
# Save as: test_qiskit.py

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp

# Create simple circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Run on simulator
sim = AerSimulator()
job = sim.run(qc, shots=1000)
result = job.result()
counts = result.get_counts()
print(f"Counts: {counts}")

# Create Hamiltonian
H = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5), ("IX", 0.5)])
print(f"Hamiltonian: {H}")
print("Qiskit installation verified!")
```

**Verification**: Outputs counts and Hamiltonian without error

---

## Phase 2: Core Quantum Components

### Step 2.1: Hamiltonians

```python
# Save as: src/quantum/hamiltonians.py

"""
Hamiltonian definitions for target physics problems.
"""

from qiskit.quantum_info import SparsePauliOp
import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class HamiltonianResult:
    """Container for Hamiltonian and exact solution."""
    operator: SparsePauliOp
    exact_energy: float
    exact_state: np.ndarray | None = None
    name: str = ""
    num_qubits: int = 0


def xy_chain(num_qubits: int, J: float = 1.0) -> HamiltonianResult:
    """
    1D XY spin chain Hamiltonian.

    H = -J * sum_i (X_i X_{i+1} + Y_i Y_{i+1})

    Args:
        num_qubits: Number of qubits/spins
        J: Coupling strength

    Returns:
        HamiltonianResult with operator and exact energy
    """
    if num_qubits < 2:
        raise ValueError("XY chain requires at least 2 qubits")

    terms = []
    for i in range(num_qubits - 1):
        # XX term
        xx = ['I'] * num_qubits
        xx[i] = 'X'
        xx[i + 1] = 'X'
        terms.append((''.join(xx), -J))

        # YY term
        yy = ['I'] * num_qubits
        yy[i] = 'Y'
        yy[i + 1] = 'Y'
        terms.append((''.join(yy), -J))

    operator = SparsePauliOp.from_list(terms)

    # Exact energy for open chain
    exact_energy = -2 * J * (num_qubits - 1) * np.cos(np.pi / (num_qubits + 1))

    return HamiltonianResult(
        operator=operator,
        exact_energy=exact_energy,
        name=f"XY Chain ({num_qubits} qubits)",
        num_qubits=num_qubits
    )


def heisenberg_chain(num_qubits: int, J: float = 1.0) -> HamiltonianResult:
    """
    1D Heisenberg XXX model.

    H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    """
    if num_qubits < 2:
        raise ValueError("Heisenberg chain requires at least 2 qubits")

    terms = []
    for i in range(num_qubits - 1):
        for pauli in ['X', 'Y', 'Z']:
            term = ['I'] * num_qubits
            term[i] = pauli
            term[i + 1] = pauli
            terms.append((''.join(term), J))

    operator = SparsePauliOp.from_list(terms)

    # Exact energy from Bethe ansatz (for small systems, diagonalize)
    matrix = operator.to_matrix()
    eigenvalues = np.linalg.eigvalsh(matrix)
    exact_energy = eigenvalues[0]

    return HamiltonianResult(
        operator=operator,
        exact_energy=exact_energy,
        name=f"Heisenberg Chain ({num_qubits} qubits)",
        num_qubits=num_qubits
    )


def transverse_ising(num_qubits: int, J: float = 1.0, h: float = 1.0) -> HamiltonianResult:
    """
    Transverse-field Ising model.

    H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i
    """
    if num_qubits < 2:
        raise ValueError("TFIM requires at least 2 qubits")

    terms = []

    # ZZ interactions
    for i in range(num_qubits - 1):
        zz = ['I'] * num_qubits
        zz[i] = 'Z'
        zz[i + 1] = 'Z'
        terms.append((''.join(zz), -J))

    # Transverse field
    for i in range(num_qubits):
        x = ['I'] * num_qubits
        x[i] = 'X'
        terms.append((''.join(x), -h))

    operator = SparsePauliOp.from_list(terms)

    # Exact energy by diagonalization
    matrix = operator.to_matrix()
    eigenvalues = np.linalg.eigvalsh(matrix)
    exact_energy = eigenvalues[0]

    return HamiltonianResult(
        operator=operator,
        exact_energy=exact_energy,
        name=f"TFIM ({num_qubits} qubits)",
        num_qubits=num_qubits
    )


# Registry
HAMILTONIANS = {
    "XY_CHAIN": xy_chain,
    "HEISENBERG": heisenberg_chain,
    "TFIM": transverse_ising,
}


def get_hamiltonian(name: str, num_qubits: int, **kwargs) -> HamiltonianResult:
    """Get Hamiltonian by name."""
    if name not in HAMILTONIANS:
        raise ValueError(f"Unknown Hamiltonian: {name}. Available: {list(HAMILTONIANS.keys())}")
    return HAMILTONIANS[name](num_qubits, **kwargs)
```

**Verification**: Create test file and run

```python
# tests/test_hamiltonians.py
from src.quantum.hamiltonians import xy_chain, heisenberg_chain, transverse_ising

def test_xy_chain():
    result = xy_chain(4)
    assert result.num_qubits == 4
    assert result.exact_energy < 0  # Ground state is negative
    print(f"XY Chain: E = {result.exact_energy:.6f}")

def test_heisenberg():
    result = heisenberg_chain(4)
    assert result.num_qubits == 4
    print(f"Heisenberg: E = {result.exact_energy:.6f}")

def test_tfim():
    result = transverse_ising(4)
    assert result.num_qubits == 4
    print(f"TFIM: E = {result.exact_energy:.6f}")

if __name__ == "__main__":
    test_xy_chain()
    test_heisenberg()
    test_tfim()
    print("All Hamiltonian tests passed!")
```

### Step 2.2: Barren Plateau Detector

```python
# Save as: src/quantum/barren_plateau.py

"""
Barren plateau detection for quantum circuits.

Based on:
- McClean et al., "Barren plateaus in quantum neural network training landscapes"
- Cerezo et al., "Cost function dependent barren plateaus"
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.primitives import Estimator
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BarrenPlateauResult:
    """Result of barren plateau detection."""
    has_barren_plateau: bool
    gradient_variance: float
    gradient_mean: float
    trainability: str  # "high", "medium", "low"
    recommendation: str
    details: dict


def compute_parameter_shift_gradient(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    param_values: dict,
    param_index: int,
    shift: float = np.pi / 2
) -> float:
    """
    Compute gradient using parameter shift rule.

    grad = (f(theta + s) - f(theta - s)) / (2 * sin(s))
    """
    estimator = Estimator()

    params = list(circuit.parameters)
    target_param = params[param_index]

    # Forward shift
    forward_values = param_values.copy()
    forward_values[target_param] = param_values[target_param] + shift
    bound_circuit_forward = circuit.assign_parameters(forward_values)

    # Backward shift
    backward_values = param_values.copy()
    backward_values[target_param] = param_values[target_param] - shift
    bound_circuit_backward = circuit.assign_parameters(backward_values)

    # Compute expectation values
    job = estimator.run(
        [bound_circuit_forward, bound_circuit_backward],
        [hamiltonian, hamiltonian]
    )
    result = job.result()

    forward_exp = result.values[0]
    backward_exp = result.values[1]

    gradient = (forward_exp - backward_exp) / (2 * np.sin(shift))
    return gradient


def detect_barren_plateau(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    num_samples: int = 50,
    seed: int = 42
) -> BarrenPlateauResult:
    """
    Detect barren plateau by sampling gradient variance.

    Args:
        circuit: Parameterized quantum circuit
        hamiltonian: Target Hamiltonian
        num_samples: Number of random parameter samples
        seed: Random seed for reproducibility

    Returns:
        BarrenPlateauResult with detection outcome
    """
    np.random.seed(seed)

    params = list(circuit.parameters)
    num_params = len(params)

    if num_params == 0:
        return BarrenPlateauResult(
            has_barren_plateau=True,
            gradient_variance=0.0,
            gradient_mean=0.0,
            trainability="none",
            recommendation="Circuit has no trainable parameters",
            details={"error": "No parameters"}
        )

    all_gradients = []

    for _ in range(num_samples):
        # Random parameter values
        random_values = np.random.uniform(0, 2 * np.pi, num_params)
        param_dict = dict(zip(params, random_values))

        # Compute gradient for first parameter (representative)
        try:
            grad = compute_parameter_shift_gradient(
                circuit, hamiltonian, param_dict, param_index=0
            )
            all_gradients.append(grad)
        except Exception as e:
            continue

    if len(all_gradients) < 10:
        return BarrenPlateauResult(
            has_barren_plateau=True,
            gradient_variance=0.0,
            gradient_mean=0.0,
            trainability="unknown",
            recommendation="Could not compute enough gradients",
            details={"error": "Insufficient gradient samples"}
        )

    gradients = np.array(all_gradients)
    variance = np.var(gradients)
    mean = np.mean(np.abs(gradients))

    # Classification thresholds (based on literature)
    # Variance < 1e-6: Severe BP
    # Variance < 1e-4: Moderate BP
    # Variance > 1e-4: Likely trainable

    if variance > 1e-2:
        has_bp = False
        trainability = "high"
        recommendation = "Circuit appears trainable. Proceed with optimization."
    elif variance > 1e-4:
        has_bp = False
        trainability = "medium"
        recommendation = "Circuit may have mild trainability issues. Monitor carefully."
    elif variance > 1e-6:
        has_bp = True
        trainability = "low"
        recommendation = "Barren plateau detected. Consider shallower circuit or different architecture."
    else:
        has_bp = True
        trainability = "very_low"
        recommendation = "Severe barren plateau. Reject this circuit."

    return BarrenPlateauResult(
        has_barren_plateau=has_bp,
        gradient_variance=variance,
        gradient_mean=mean,
        trainability=trainability,
        recommendation=recommendation,
        details={
            "num_samples": len(all_gradients),
            "num_params": num_params,
            "circuit_depth": circuit.depth()
        }
    )


def quick_bp_check(circuit: QuantumCircuit) -> Tuple[bool, str]:
    """
    Quick heuristic check for barren plateau risk.
    Based on circuit structure, not gradient sampling.
    """
    depth = circuit.depth()
    num_qubits = circuit.num_qubits
    num_params = len(circuit.parameters)

    # Heuristics from literature
    # Deep random circuits (depth > 2*n) are prone to BP
    if depth > 2 * num_qubits:
        return True, f"Circuit too deep ({depth} > 2*{num_qubits})"

    # Check for hardware-efficient patterns (less prone)
    ops = circuit.count_ops()
    has_local_rotations = any(g in ops for g in ['rx', 'ry', 'rz'])
    has_entanglement = any(g in ops for g in ['cx', 'cz', 'swap'])

    if not has_local_rotations:
        return True, "No local rotations found"

    if not has_entanglement and num_qubits > 1:
        return True, "No entanglement in multi-qubit circuit"

    return False, "Passed heuristic check"
```

**Verification**:

```python
# tests/test_barren_plateau.py
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from src.quantum.hamiltonians import xy_chain
from src.quantum.barren_plateau import detect_barren_plateau, quick_bp_check

def test_shallow_circuit():
    """Shallow circuit should be trainable."""
    qc = QuantumCircuit(4)
    params = [Parameter(f'p{i}') for i in range(4)]
    for i in range(4):
        qc.ry(params[i], i)
    for i in range(3):
        qc.cx(i, i+1)

    ham = xy_chain(4).operator
    result = detect_barren_plateau(qc, ham, num_samples=30)
    print(f"Shallow circuit: variance={result.gradient_variance:.2e}, trainable={result.trainability}")
    assert result.trainability in ["high", "medium"]

def test_deep_circuit():
    """Deep random circuit should have BP issues."""
    qc = QuantumCircuit(4)
    params = [Parameter(f'p{i}') for i in range(40)]
    for layer in range(10):
        for i in range(4):
            qc.ry(params[layer*4 + i], i)
        for i in range(3):
            qc.cx(i, i+1)

    ham = xy_chain(4).operator
    result = detect_barren_plateau(qc, ham, num_samples=30)
    print(f"Deep circuit: variance={result.gradient_variance:.2e}, trainable={result.trainability}")

if __name__ == "__main__":
    test_shallow_circuit()
    test_deep_circuit()
    print("BP detection tests completed!")
```

### Step 2.3: Circuit Verifier

```python
# Save as: src/quantum/verifier.py

"""
Circuit verification and validation.
"""

from qiskit import QuantumCircuit
from dataclasses import dataclass
from typing import List, Set
import traceback


@dataclass
class VerificationResult:
    """Result of circuit verification."""
    is_valid: bool
    syntax_ok: bool
    depth_ok: bool
    gates_ok: bool
    params_ok: bool
    qubit_count_ok: bool
    errors: List[str]
    warnings: List[str]
    circuit: QuantumCircuit | None = None


def verify_circuit_code(
    code: str,
    expected_qubits: int,
    max_depth: int = 100,
    allowed_gates: Set[str] | None = None
) -> VerificationResult:
    """
    Verify LLM-generated circuit code.

    Args:
        code: Python code string that defines create_ansatz function
        expected_qubits: Expected number of qubits
        max_depth: Maximum allowed circuit depth
        allowed_gates: Set of allowed gate names

    Returns:
        VerificationResult with validation outcome
    """
    errors = []
    warnings = []

    if allowed_gates is None:
        allowed_gates = {'rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 'cx', 'cz', 'swap', 'barrier'}

    # Step 1: Syntax check
    try:
        # Create isolated namespace
        namespace = {}
        exec(code, namespace)
        syntax_ok = True
    except SyntaxError as e:
        errors.append(f"Syntax error: {e}")
        return VerificationResult(
            is_valid=False, syntax_ok=False, depth_ok=False,
            gates_ok=False, params_ok=False, qubit_count_ok=False,
            errors=errors, warnings=warnings
        )
    except Exception as e:
        errors.append(f"Execution error: {e}")
        return VerificationResult(
            is_valid=False, syntax_ok=False, depth_ok=False,
            gates_ok=False, params_ok=False, qubit_count_ok=False,
            errors=errors, warnings=warnings
        )

    # Step 2: Check for create_ansatz function
    if 'create_ansatz' not in namespace:
        errors.append("Code must define 'create_ansatz' function")
        return VerificationResult(
            is_valid=False, syntax_ok=True, depth_ok=False,
            gates_ok=False, params_ok=False, qubit_count_ok=False,
            errors=errors, warnings=warnings
        )

    # Step 3: Call the function
    try:
        circuit = namespace['create_ansatz'](expected_qubits)
        if not isinstance(circuit, QuantumCircuit):
            errors.append("create_ansatz must return a QuantumCircuit")
            return VerificationResult(
                is_valid=False, syntax_ok=True, depth_ok=False,
                gates_ok=False, params_ok=False, qubit_count_ok=False,
                errors=errors, warnings=warnings
            )
    except Exception as e:
        errors.append(f"Failed to create circuit: {e}")
        return VerificationResult(
            is_valid=False, syntax_ok=True, depth_ok=False,
            gates_ok=False, params_ok=False, qubit_count_ok=False,
            errors=errors, warnings=warnings
        )

    # Step 4: Qubit count check
    qubit_count_ok = circuit.num_qubits == expected_qubits
    if not qubit_count_ok:
        errors.append(f"Wrong qubit count: {circuit.num_qubits} != {expected_qubits}")

    # Step 5: Depth check
    depth_ok = circuit.depth() <= max_depth
    if not depth_ok:
        errors.append(f"Circuit too deep: {circuit.depth()} > {max_depth}")

    # Step 6: Gate check
    used_gates = set(circuit.count_ops().keys())
    invalid_gates = used_gates - allowed_gates - {'measure', 'barrier'}
    gates_ok = len(invalid_gates) == 0
    if not gates_ok:
        errors.append(f"Invalid gates used: {invalid_gates}")

    # Step 7: Parameter check
    params_ok = len(circuit.parameters) > 0
    if not params_ok:
        warnings.append("Circuit has no trainable parameters")

    # Final verdict
    is_valid = syntax_ok and depth_ok and gates_ok and qubit_count_ok
    if not params_ok:
        is_valid = False
        errors.append("Circuit must have trainable parameters")

    return VerificationResult(
        is_valid=is_valid,
        syntax_ok=syntax_ok,
        depth_ok=depth_ok,
        gates_ok=gates_ok,
        params_ok=params_ok,
        qubit_count_ok=qubit_count_ok,
        errors=errors,
        warnings=warnings,
        circuit=circuit if is_valid else None
    )
```

---

## Phase 3: LLM Fine-Tuning

See `MODELS.md` and `DATASETS.md` for detailed configuration.

### Step 3.1: Download Dataset

```python
# Save as: src/training/download_data.py

from datasets import load_dataset
import json
import os

def download_quantum_datasets():
    """Download and prepare quantum datasets."""
    os.makedirs("data/raw", exist_ok=True)

    # QuantumLLMInstruct
    print("Downloading QuantumLLMInstruct...")
    ds = load_dataset("BoltzmannEntropy/QuantumLLMInstruct")
    ds.save_to_disk("data/raw/quantum_llm_instruct")
    print(f"Downloaded {len(ds['train'])} examples")

    print("Done!")

if __name__ == "__main__":
    download_quantum_datasets()
```

### Step 3.2: Prepare Training Data

See `DATASETS.md` for format specification.

### Step 3.3: Fine-Tune Model

See `MODELS.md` for configuration.

---

## Phase 4: Agent Implementation

### Step 4.1: LLM Proposer

```python
# Save as: src/agent/proposer.py

"""
LLM-based quantum circuit proposer.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class ProposerConfig:
    model_path: str
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LLMProposer:
    """Generate quantum circuits using fine-tuned LLM."""

    def __init__(self, config: ProposerConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(
        self,
        goal_description: str,
        num_qubits: int,
        constraints: dict,
        memory_context: str = ""
    ) -> str:
        """
        Generate circuit code for the given goal.

        Args:
            goal_description: Natural language description of physics goal
            num_qubits: Number of qubits
            constraints: Dict with max_depth, allowed_gates, etc.
            memory_context: Previous successes/failures

        Returns:
            Python code string defining create_ansatz function
        """
        prompt = self._build_prompt(goal_description, num_qubits, constraints, memory_context)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract code from response
        code = self._extract_code(generated)
        return code

    def _build_prompt(
        self,
        goal: str,
        num_qubits: int,
        constraints: dict,
        memory: str
    ) -> str:
        """Build prompt for circuit generation."""
        prompt = f"""You are a quantum computing expert. Generate a parameterized quantum circuit in Qiskit.

GOAL: {goal}

SPECIFICATIONS:
- Number of qubits: {num_qubits}
- Maximum circuit depth: {constraints.get('max_depth', 50)}
- Allowed gates: {constraints.get('allowed_gates', ['rx', 'ry', 'rz', 'cx'])}

REQUIREMENTS:
1. Define a function called `create_ansatz(num_qubits: int) -> QuantumCircuit`
2. Use Parameters from qiskit.circuit for trainable parameters
3. Circuit must be hardware-efficient
4. Avoid patterns that cause barren plateaus

{f"CONTEXT FROM PREVIOUS ATTEMPTS:{chr(10)}{memory}" if memory else ""}

Generate ONLY the Python code. No explanations.

```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int) -> QuantumCircuit:
"""
        return prompt

    def _extract_code(self, generated: str) -> str:
        """Extract Python code from generated text."""
        # Find code block
        if "```python" in generated:
            start = generated.find("```python") + 9
            end = generated.find("```", start)
            if end > start:
                return generated[start:end].strip()

        # Find function definition
        if "def create_ansatz" in generated:
            start = generated.find("from qiskit")
            if start == -1:
                start = generated.find("def create_ansatz")
            return generated[start:].strip()

        return generated
```

### Step 4.2: Memory System

See `ARCHITECTURE.md` for specification.

### Step 4.3: Main Agent

See `ARCHITECTURE.md` for specification.

---

## Phase 5: Evaluation

### Step 5.1: Implement Baselines

```python
# Save as: src/evaluation/baselines.py

"""
Baseline ansatzes for comparison.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal, EfficientSU2
import numpy as np


def hardware_efficient_ansatz(num_qubits: int, depth: int = 3) -> QuantumCircuit:
    """
    Standard hardware-efficient ansatz (HEA).
    """
    return TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=['ry', 'rz'],
        entanglement_blocks='cx',
        entanglement='linear',
        reps=depth
    )


def efficient_su2_ansatz(num_qubits: int, depth: int = 3) -> QuantumCircuit:
    """
    Efficient SU(2) ansatz.
    """
    return EfficientSU2(
        num_qubits=num_qubits,
        reps=depth,
        entanglement='linear'
    )


def custom_hea(num_qubits: int, layers: int = 2) -> QuantumCircuit:
    """
    Custom hardware-efficient ansatz for benchmarking.
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

        # Entanglement layer
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

    # Final rotation layer
    for i in range(num_qubits):
        qc.ry(Parameter(f'theta_{param_count}'), i)
        param_count += 1

    return qc
```

### Step 5.2: Implement Metrics

See `EVALUATION.md` for specification.

---

## Phase 6: Hardware Validation

### Step 6.1: IBM Runtime Setup

```python
# Save as: src/quantum/hardware.py

"""
IBM Quantum hardware interface.
"""

from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from typing import List, Tuple
import time


class IBMHardwareRunner:
    def __init__(self, backend_name: str = "ibm_sherbrooke"):
        self.service = QiskitRuntimeService()
        self.backend = self.service.backend(backend_name)
        self.pm = generate_preset_pass_manager(
            backend=self.backend,
            optimization_level=3
        )

    def run_estimation(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        parameter_values: List[float],
        shots: int = 4000
    ) -> Tuple[float, float]:
        """
        Run expectation value estimation on hardware.

        Returns:
            Tuple of (expectation_value, standard_error)
        """
        # Transpile circuit
        bound_circuit = circuit.assign_parameters(parameter_values)
        transpiled = self.pm.run(bound_circuit)

        with Session(backend=self.backend) as session:
            estimator = Estimator(session=session)
            estimator.options.resilience_level = 2  # Enable error mitigation

            job = estimator.run(
                circuits=[transpiled],
                observables=[hamiltonian]
            )

            result = job.result()
            exp_val = result.values[0]
            std_err = result.metadata[0].get('std', 0.0)

        return exp_val, std_err
```

---

## Testing Strategy

### Unit Tests
- Every function has tests
- Run with `pytest tests/`

### Integration Tests
- Test component combinations
- Run with `pytest tests/integration/`

### End-to-End Tests
- Full agent loop on small problems
- Run with `pytest tests/e2e/`

### Hardware Tests
- Validate on IBM quantum
- Run manually when ready

---

## Build Sequence Summary

1. Environment setup
2. Hamiltonians implementation
3. Barren plateau detector
4. Circuit verifier
5. Download datasets
6. Prepare training data
7. Fine-tune LLM
8. Implement proposer
9. Implement memory
10. Implement main agent
11. Implement baselines
12. Implement evaluation
13. Run discovery campaigns
14. Hardware validation
15. Analysis and paper

**Follow this sequence. Do not skip steps.**
