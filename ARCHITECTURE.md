# QuantumMind Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         QUANTUMMIND AGENT                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│   │   PHYSICS    │    │     LLM      │    │   MEMORY     │              │
│   │    GOAL      │───▶│   PROPOSER   │◀───│   SYSTEM     │              │
│   │              │    │              │    │              │              │
│   └──────────────┘    └──────┬───────┘    └──────▲───────┘              │
│                              │                    │                      │
│                              ▼                    │                      │
│                       ┌──────────────┐            │                      │
│                       │   CIRCUIT    │            │                      │
│                       │  VERIFIER    │            │                      │
│                       │              │            │                      │
│                       └──────┬───────┘            │                      │
│                              │                    │                      │
│                              ▼                    │                      │
│                       ┌──────────────┐            │                      │
│                       │   BARREN     │            │                      │
│                       │  PLATEAU     │            │                      │
│                       │  DETECTOR    │            │                      │
│                       └──────┬───────┘            │                      │
│                              │                    │                      │
│                              ▼                    │                      │
│                       ┌──────────────┐            │                      │
│                       │  QUANTUM     │            │                      │
│                       │  EXECUTOR    │            │                      │
│                       │ (Sim + HW)   │            │                      │
│                       └──────┬───────┘            │                      │
│                              │                    │                      │
│                              ▼                    │                      │
│                       ┌──────────────┐            │                      │
│                       │   RESULT     │────────────┘                      │
│                       │  ANALYZER    │                                   │
│                       │              │                                   │
│                       └──────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Physics Goal (Input)

**Purpose**: Define what the agent should discover.

**Input Format**:
```python
@dataclass
class PhysicsGoal:
    name: str                    # "XY Spin Chain Ground State"
    hamiltonian: str             # Hamiltonian specification
    num_qubits: int              # System size
    target_metric: str           # "energy" | "fidelity"
    target_value: float          # Exact solution (for comparison)
    constraints: dict            # Hardware constraints

# Example
goal = PhysicsGoal(
    name="4-qubit XY Spin Chain",
    hamiltonian="XY_CHAIN",
    num_qubits=4,
    target_metric="energy",
    target_value=-2.0,  # Exact ground state energy
    constraints={
        "max_depth": 50,
        "max_gates": 100,
        "allowed_gates": ["rx", "ry", "rz", "cx", "cz"]
    }
)
```

**Supported Hamiltonians**:
| Hamiltonian | Description | Qubits | Exact Solution |
|-------------|-------------|--------|----------------|
| `XY_CHAIN` | 1D XY spin chain | 4-12 | Analytically solvable |
| `HEISENBERG` | Heisenberg XXX model | 4-8 | Analytically solvable |
| `TFIM` | Transverse-field Ising | 4-10 | Analytically solvable |
| `H2` | Hydrogen molecule | 4 | FCI available |
| `LIH` | Lithium hydride | 12 | FCI available |

---

### 2. LLM Proposer

**Purpose**: Generate quantum circuits from physics goals.

**Model**: Qwen2.5-Coder-7B fine-tuned on quantum datasets

**Input**:
```python
prompt = f"""
Goal: Find the ground state of a {goal.num_qubits}-qubit {goal.name}

Hamiltonian: {goal.hamiltonian_string}

Constraints:
- Maximum circuit depth: {goal.constraints['max_depth']}
- Allowed gates: {goal.constraints['allowed_gates']}
- Target energy: {goal.target_value}

Previous attempts that FAILED (avoid these patterns):
{memory.get_failures()}

Previous attempts that WORKED (build on these):
{memory.get_successes()}

Generate a parameterized quantum circuit in Qiskit that prepares a
variational ansatz for finding this ground state. The circuit should:
1. Be hardware-efficient
2. Avoid known barren plateau patterns
3. Respect qubit connectivity constraints

Output only valid Python code using Qiskit.
"""
```

**Output**: Valid Qiskit Python code

```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int = 4) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = [Parameter(f'theta_{i}') for i in range(num_qubits * 2)]

    # Layer 1: Single-qubit rotations
    for i in range(num_qubits):
        qc.ry(params[i], i)

    # Layer 2: Entanglement
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    # Layer 3: More rotations
    for i in range(num_qubits):
        qc.ry(params[num_qubits + i], i)

    return qc
```

---

### 3. Circuit Verifier

**Purpose**: Validate generated circuits before execution.

**Checks**:
```python
@dataclass
class VerificationResult:
    is_valid: bool
    syntax_ok: bool
    depth_ok: bool
    gates_ok: bool
    params_ok: bool
    errors: List[str]

def verify_circuit(code: str, goal: PhysicsGoal) -> VerificationResult:
    errors = []

    # 1. Syntax check - does the code execute?
    try:
        exec(code)
        circuit = create_ansatz(goal.num_qubits)
        syntax_ok = True
    except Exception as e:
        errors.append(f"Syntax error: {e}")
        syntax_ok = False
        return VerificationResult(False, False, False, False, False, errors)

    # 2. Depth check
    depth_ok = circuit.depth() <= goal.constraints['max_depth']
    if not depth_ok:
        errors.append(f"Depth {circuit.depth()} > max {goal.constraints['max_depth']}")

    # 3. Gate check
    used_gates = set(circuit.count_ops().keys())
    allowed = set(goal.constraints['allowed_gates'])
    gates_ok = used_gates.issubset(allowed)
    if not gates_ok:
        errors.append(f"Invalid gates: {used_gates - allowed}")

    # 4. Parameter check - must have trainable parameters
    params_ok = len(circuit.parameters) > 0
    if not params_ok:
        errors.append("No trainable parameters found")

    is_valid = syntax_ok and depth_ok and gates_ok and params_ok
    return VerificationResult(is_valid, syntax_ok, depth_ok, gates_ok, params_ok, errors)
```

---

### 4. Barren Plateau Detector

**Purpose**: Detect barren plateau risk before full optimization.

**Method**: Gradient variance analysis

```python
@dataclass
class BarrenPlateauResult:
    has_barren_plateau: bool
    gradient_variance: float
    estimated_trainability: str  # "high" | "medium" | "low"
    recommendation: str

def detect_barren_plateau(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    num_samples: int = 100
) -> BarrenPlateauResult:
    """
    Detect barren plateau by sampling gradient variance.

    Theory: If variance decays exponentially with qubit count,
    the circuit has a barren plateau. We sample random parameters
    and compute gradient variance.

    Based on: McClean et al., "Barren plateaus in quantum neural
    network training landscapes" (Nature Comms, 2018)
    """
    gradients = []

    for _ in range(num_samples):
        # Random parameter initialization
        params = np.random.uniform(0, 2*np.pi, len(circuit.parameters))
        param_dict = dict(zip(circuit.parameters, params))

        # Compute gradient using parameter shift rule
        grad = compute_gradient(circuit, hamiltonian, param_dict)
        gradients.append(grad)

    gradients = np.array(gradients)
    variance = np.var(gradients, axis=0).mean()

    # Threshold based on empirical studies
    # Variance < 1e-4 typically indicates barren plateau
    has_bp = variance < 1e-4

    if variance > 1e-2:
        trainability = "high"
        recommendation = "Circuit is trainable. Proceed with optimization."
    elif variance > 1e-4:
        trainability = "medium"
        recommendation = "Circuit may have trainability issues. Consider shallower depth."
    else:
        trainability = "low"
        recommendation = "Barren plateau detected. Reject this circuit."

    return BarrenPlateauResult(has_bp, variance, trainability, recommendation)
```

---

### 5. Quantum Executor

**Purpose**: Run circuits on simulator and real hardware.

**Backends**:
1. `qiskit_aer` - Fast local simulation
2. `ibm_sherbrooke` - IBM Eagle 127-qubit
3. `ibm_brisbane` - IBM Eagle 127-qubit (backup)

```python
class QuantumExecutor:
    def __init__(self, use_hardware: bool = False):
        self.use_hardware = use_hardware
        self.simulator = AerSimulator()

        if use_hardware:
            self.service = QiskitRuntimeService()
            self.backend = self.service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=127
            )

    def run_vqe(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        optimizer: str = "COBYLA",
        max_iterations: int = 200
    ) -> VQEResult:
        """
        Run VQE optimization on the given circuit.
        """
        if self.use_hardware:
            # Use Qiskit Runtime for hardware
            estimator = Estimator(backend=self.backend)
            # Enable error mitigation
            estimator.options.resilience_level = 2
        else:
            estimator = Estimator()

        # Set up VQE
        vqe = VQE(
            estimator=estimator,
            ansatz=circuit,
            optimizer=COBYLA(maxiter=max_iterations)
        )

        result = vqe.compute_minimum_eigenvalue(hamiltonian)

        return VQEResult(
            energy=result.eigenvalue.real,
            optimal_params=result.optimal_parameters,
            num_iterations=result.cost_function_evals,
            circuit=circuit.assign_parameters(result.optimal_parameters)
        )
```

---

### 6. Result Analyzer

**Purpose**: Evaluate results and generate feedback for learning.

```python
@dataclass
class AnalysisResult:
    success: bool
    energy_error: float
    fidelity: float
    circuit_depth: int
    gate_count: int
    feedback: str
    should_add_to_memory: bool

def analyze_result(
    result: VQEResult,
    goal: PhysicsGoal,
    circuit: QuantumCircuit
) -> AnalysisResult:
    """
    Analyze VQE result and generate feedback.
    """
    energy_error = abs(result.energy - goal.target_value)

    # Calculate fidelity if we have access to exact state
    if goal.exact_state is not None:
        fidelity = state_fidelity(result.final_state, goal.exact_state)
    else:
        fidelity = None

    circuit_depth = circuit.depth()
    gate_count = sum(circuit.count_ops().values())

    # Success criteria
    success = energy_error < 0.01  # Within 1% of exact

    # Generate feedback
    if success:
        feedback = f"""
        SUCCESS: Energy error {energy_error:.6f} < 0.01
        Circuit depth: {circuit_depth}
        Gate count: {gate_count}
        This circuit pattern works well. Consider variations.
        """
        should_add = True
    else:
        feedback = f"""
        FAILURE: Energy error {energy_error:.6f} >= 0.01
        Circuit depth: {circuit_depth}
        Gate count: {gate_count}
        Issues identified:
        - {"Circuit too shallow" if circuit_depth < 5 else ""}
        - {"Not enough entanglement" if "cx" not in circuit.count_ops() else ""}
        Avoid this pattern in future attempts.
        """
        should_add = True  # Still useful to remember failures

    return AnalysisResult(
        success=success,
        energy_error=energy_error,
        fidelity=fidelity,
        circuit_depth=circuit_depth,
        gate_count=gate_count,
        feedback=feedback,
        should_add_to_memory=should_add
    )
```

---

### 7. Memory System

**Purpose**: Learn from past experiments to improve future proposals.

```python
@dataclass
class MemoryEntry:
    circuit_code: str
    circuit_pattern: str  # Abstract pattern description
    result: AnalysisResult
    goal: PhysicsGoal
    timestamp: datetime

class AgentMemory:
    def __init__(self, max_entries: int = 1000):
        self.successes: List[MemoryEntry] = []
        self.failures: List[MemoryEntry] = []
        self.max_entries = max_entries

    def add(self, entry: MemoryEntry):
        if entry.result.success:
            self.successes.append(entry)
        else:
            self.failures.append(entry)

        # Prune old entries if needed
        self._prune()

    def get_successes(self, goal: PhysicsGoal, limit: int = 5) -> str:
        """Get relevant successful patterns for the given goal."""
        relevant = [e for e in self.successes if self._is_relevant(e, goal)]
        relevant.sort(key=lambda e: e.result.energy_error)

        summaries = []
        for entry in relevant[:limit]:
            summaries.append(f"""
            Pattern: {entry.circuit_pattern}
            Energy error: {entry.result.energy_error:.6f}
            Depth: {entry.result.circuit_depth}
            """)

        return "\n".join(summaries) if summaries else "No successful patterns yet."

    def get_failures(self, goal: PhysicsGoal, limit: int = 5) -> str:
        """Get patterns to avoid."""
        relevant = [e for e in self.failures if self._is_relevant(e, goal)]

        summaries = []
        for entry in relevant[-limit:]:
            summaries.append(f"""
            AVOID: {entry.circuit_pattern}
            Reason: Energy error {entry.result.energy_error:.6f}
            """)

        return "\n".join(summaries) if summaries else "No failed patterns recorded."

    def _is_relevant(self, entry: MemoryEntry, goal: PhysicsGoal) -> bool:
        """Check if a memory entry is relevant to the current goal."""
        return (
            entry.goal.hamiltonian == goal.hamiltonian and
            abs(entry.goal.num_qubits - goal.num_qubits) <= 2
        )

    def _prune(self):
        if len(self.successes) > self.max_entries:
            self.successes = self.successes[-self.max_entries:]
        if len(self.failures) > self.max_entries:
            self.failures = self.failures[-self.max_entries:]
```

---

## Main Agent Loop

```python
class QuantumMindAgent:
    def __init__(
        self,
        model_path: str,
        use_hardware: bool = False,
        max_iterations: int = 100
    ):
        self.proposer = LLMProposer(model_path)
        self.verifier = CircuitVerifier()
        self.bp_detector = BarrenPlateauDetector()
        self.executor = QuantumExecutor(use_hardware)
        self.analyzer = ResultAnalyzer()
        self.memory = AgentMemory()
        self.max_iterations = max_iterations

    def discover(self, goal: PhysicsGoal) -> DiscoveryResult:
        """
        Main discovery loop.
        """
        best_result = None
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n=== Iteration {iteration} ===")

            # 1. Generate circuit proposal
            code = self.proposer.generate(goal, self.memory)
            print(f"Generated circuit code")

            # 2. Verify circuit
            verification = self.verifier.verify(code, goal)
            if not verification.is_valid:
                print(f"Verification failed: {verification.errors}")
                self.memory.add(MemoryEntry(
                    circuit_code=code,
                    circuit_pattern="INVALID",
                    result=AnalysisResult(success=False, ...),
                    goal=goal,
                    timestamp=datetime.now()
                ))
                continue

            # 3. Build circuit
            circuit = self._build_circuit(code, goal.num_qubits)

            # 4. Check for barren plateaus
            bp_result = self.bp_detector.detect(circuit, goal.hamiltonian)
            if bp_result.has_barren_plateau:
                print(f"Barren plateau detected: {bp_result.recommendation}")
                self.memory.add(MemoryEntry(
                    circuit_code=code,
                    circuit_pattern="BARREN_PLATEAU",
                    result=AnalysisResult(success=False, ...),
                    goal=goal,
                    timestamp=datetime.now()
                ))
                continue

            # 5. Run VQE
            print(f"Running VQE...")
            vqe_result = self.executor.run_vqe(circuit, goal.hamiltonian)

            # 6. Analyze results
            analysis = self.analyzer.analyze(vqe_result, goal, circuit)
            print(f"Energy error: {analysis.energy_error:.6f}")

            # 7. Update memory
            pattern = self._extract_pattern(circuit)
            self.memory.add(MemoryEntry(
                circuit_code=code,
                circuit_pattern=pattern,
                result=analysis,
                goal=goal,
                timestamp=datetime.now()
            ))

            # 8. Track best
            if best_result is None or analysis.energy_error < best_result.energy_error:
                best_result = analysis
                best_circuit = circuit
                print(f"New best! Energy error: {analysis.energy_error:.6f}")

            # 9. Check convergence
            if analysis.success:
                print(f"\n=== SUCCESS after {iteration} iterations ===")
                break

        return DiscoveryResult(
            goal=goal,
            best_circuit=best_circuit,
            best_result=best_result,
            total_iterations=iteration,
            memory=self.memory
        )
```

---

## Data Flow

```
1. PhysicsGoal
      │
      ▼
2. LLM generates circuit code
      │
      ▼
3. Verifier checks syntax/constraints
      │ (if invalid → memory, retry)
      ▼
4. BP Detector checks trainability
      │ (if BP → memory, retry)
      ▼
5. Executor runs VQE (simulator first)
      │
      ▼
6. Analyzer evaluates result
      │
      ▼
7. Memory stores outcome
      │
      ▼
8. Loop until success or max iterations
      │
      ▼
9. Best circuit validated on IBM hardware
      │
      ▼
10. Results documented for paper
```

---

## Configuration

```python
# config.py

@dataclass
class AgentConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    model_path: str = "./models/quantum-mind-v1"  # After fine-tuning
    use_quantized: bool = True  # 4-bit for inference

    # Agent
    max_iterations: int = 100
    circuits_per_iteration: int = 3
    use_hardware: bool = False  # Enable for final validation

    # Barren Plateau Detection
    bp_num_samples: int = 100
    bp_threshold: float = 1e-4

    # VQE
    optimizer: str = "COBYLA"
    max_vqe_iterations: int = 200

    # Memory
    max_memory_entries: int = 1000

    # Hardware
    ibm_backend: str = "ibm_sherbrooke"
    error_mitigation: bool = True
    resilience_level: int = 2

    # Logging
    log_dir: str = "./experiments/logs"
    save_checkpoints: bool = True
```

---

## Error Handling

```python
class QuantumMindError(Exception):
    """Base exception for QuantumMind."""
    pass

class CircuitGenerationError(QuantumMindError):
    """LLM failed to generate valid circuit."""
    pass

class VerificationError(QuantumMindError):
    """Circuit failed verification."""
    pass

class ExecutionError(QuantumMindError):
    """Quantum execution failed."""
    pass

class HardwareError(QuantumMindError):
    """IBM hardware communication failed."""
    pass

# All errors are caught, logged, and the agent continues
# No error should crash the discovery loop
```

---

## Testing Requirements

Each component must have:

1. **Unit tests**: Test individual functions
2. **Integration tests**: Test component interactions
3. **End-to-end tests**: Test full discovery loop
4. **Hardware tests**: Test on IBM hardware (when ready)

See `BUILD_PLAN.md` for testing strategy.
