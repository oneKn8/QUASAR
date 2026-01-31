"""
Circuit verification and validation module.

This module validates LLM-generated quantum circuit code by checking:
1. Syntax validity (code compiles and executes)
2. Function signature (must define create_ansatz)
3. Circuit properties (depth, gates, parameters)
4. Hardware compatibility

Security Note: This module executes untrusted code in an isolated namespace.
For production use, consider additional sandboxing.
"""

from dataclasses import dataclass, field
from typing import Literal

from qiskit import QuantumCircuit


@dataclass
class VerificationResult:
    """Result of circuit code verification."""

    is_valid: bool
    syntax_ok: bool
    function_ok: bool
    depth_ok: bool
    gates_ok: bool
    params_ok: bool
    qubit_count_ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    circuit: QuantumCircuit | None = None
    details: dict = field(default_factory=dict)


# Default allowed gates for hardware-efficient circuits
DEFAULT_ALLOWED_GATES = frozenset(
    {
        # Single-qubit rotations
        "rx",
        "ry",
        "rz",
        "r",
        # Single-qubit Pauli gates
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        # Universal single-qubit
        "u",
        "u1",
        "u2",
        "u3",
        "p",
        # Two-qubit gates
        "cx",
        "cy",
        "cz",
        "swap",
        "iswap",
        # Controlled rotations
        "crx",
        "cry",
        "crz",
        "cp",
        # Other common gates
        "ecr",
        "rzx",
        "rxx",
        "ryy",
        "rzz",
        # Utility (not counted as gates)
        "barrier",
        "measure",
    }
)


def verify_circuit_code(
    code: str,
    expected_qubits: int,
    max_depth: int = 100,
    allowed_gates: frozenset[str] | None = None,
    require_params: bool = True,
    min_params: int = 1,
) -> VerificationResult:
    """
    Verify LLM-generated circuit code.

    This function executes the provided code in an isolated namespace
    and validates the resulting circuit against specified constraints.

    Args:
        code: Python code string defining create_ansatz function
        expected_qubits: Expected number of qubits in the circuit
        max_depth: Maximum allowed circuit depth
        allowed_gates: Set of allowed gate names (default: common gates)
        require_params: Whether trainable parameters are required
        min_params: Minimum number of parameters required

    Returns:
        VerificationResult with validation outcome

    Security:
        Code is executed in isolated namespace. For untrusted code,
        consider additional sandboxing (subprocess, container).
    """
    errors = []
    warnings = []

    if allowed_gates is None:
        allowed_gates = DEFAULT_ALLOWED_GATES

    # Step 1: Syntax check - compile the code
    try:
        compiled = compile(code, "<llm_generated>", "exec")
        syntax_ok = True
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        return VerificationResult(
            is_valid=False,
            syntax_ok=False,
            function_ok=False,
            depth_ok=False,
            gates_ok=False,
            params_ok=False,
            qubit_count_ok=False,
            errors=errors,
            warnings=warnings,
        )

    # Step 2: Execute in isolated namespace
    namespace = {"__builtins__": __builtins__}
    try:
        exec(compiled, namespace)
    except Exception as e:
        errors.append(f"Execution error: {type(e).__name__}: {e}")
        return VerificationResult(
            is_valid=False,
            syntax_ok=True,
            function_ok=False,
            depth_ok=False,
            gates_ok=False,
            params_ok=False,
            qubit_count_ok=False,
            errors=errors,
            warnings=warnings,
        )

    # Step 3: Check for create_ansatz function
    if "create_ansatz" not in namespace:
        errors.append("Code must define a function named 'create_ansatz'")
        return VerificationResult(
            is_valid=False,
            syntax_ok=True,
            function_ok=False,
            depth_ok=False,
            gates_ok=False,
            params_ok=False,
            qubit_count_ok=False,
            errors=errors,
            warnings=warnings,
        )

    create_ansatz = namespace["create_ansatz"]
    if not callable(create_ansatz):
        errors.append("'create_ansatz' must be a callable function")
        return VerificationResult(
            is_valid=False,
            syntax_ok=True,
            function_ok=False,
            depth_ok=False,
            gates_ok=False,
            params_ok=False,
            qubit_count_ok=False,
            errors=errors,
            warnings=warnings,
        )

    # Step 4: Call the function to get the circuit
    try:
        circuit = create_ansatz(expected_qubits)
    except TypeError:
        # Try without arguments
        try:
            circuit = create_ansatz()
        except Exception as e:
            errors.append(f"Failed to call create_ansatz: {type(e).__name__}: {e}")
            return VerificationResult(
                is_valid=False,
                syntax_ok=True,
                function_ok=False,
                depth_ok=False,
                gates_ok=False,
                params_ok=False,
                qubit_count_ok=False,
                errors=errors,
                warnings=warnings,
            )
    except Exception as e:
        errors.append(f"create_ansatz raised {type(e).__name__}: {e}")
        return VerificationResult(
            is_valid=False,
            syntax_ok=True,
            function_ok=False,
            depth_ok=False,
            gates_ok=False,
            params_ok=False,
            qubit_count_ok=False,
            errors=errors,
            warnings=warnings,
        )

    # Step 5: Verify return type
    if not isinstance(circuit, QuantumCircuit):
        errors.append(
            f"create_ansatz must return QuantumCircuit, got {type(circuit).__name__}"
        )
        return VerificationResult(
            is_valid=False,
            syntax_ok=True,
            function_ok=False,
            depth_ok=False,
            gates_ok=False,
            params_ok=False,
            qubit_count_ok=False,
            errors=errors,
            warnings=warnings,
        )

    function_ok = True

    # Step 6: Check qubit count
    qubit_count_ok = circuit.num_qubits == expected_qubits
    if not qubit_count_ok:
        errors.append(
            f"Wrong qubit count: got {circuit.num_qubits}, expected {expected_qubits}"
        )

    # Step 7: Check circuit depth
    depth = circuit.depth()
    depth_ok = depth <= max_depth
    if not depth_ok:
        errors.append(f"Circuit depth {depth} exceeds maximum {max_depth}")
    elif depth > max_depth * 0.8:
        warnings.append(f"Circuit depth {depth} is close to limit {max_depth}")

    # Step 8: Check gates
    ops = circuit.count_ops()
    used_gates = set(ops.keys())
    invalid_gates = used_gates - allowed_gates
    gates_ok = len(invalid_gates) == 0
    if not gates_ok:
        errors.append(f"Invalid gates used: {invalid_gates}")

    # Step 9: Check parameters
    num_params = len(circuit.parameters)
    if require_params:
        params_ok = num_params >= min_params
        if not params_ok:
            errors.append(
                f"Circuit has {num_params} parameters, minimum required is {min_params}"
            )
    else:
        params_ok = True
        if num_params == 0:
            warnings.append("Circuit has no trainable parameters")

    # Collect details
    details = {
        "num_qubits": circuit.num_qubits,
        "depth": depth,
        "num_params": num_params,
        "gate_counts": dict(ops),
        "total_gates": sum(ops.values()),
    }

    # Final verdict
    is_valid = (
        syntax_ok
        and function_ok
        and qubit_count_ok
        and depth_ok
        and gates_ok
        and params_ok
    )

    return VerificationResult(
        is_valid=is_valid,
        syntax_ok=syntax_ok,
        function_ok=function_ok,
        depth_ok=depth_ok,
        gates_ok=gates_ok,
        params_ok=params_ok,
        qubit_count_ok=qubit_count_ok,
        errors=errors,
        warnings=warnings,
        circuit=circuit if is_valid else None,
        details=details,
    )


def extract_circuit_from_code(code: str, num_qubits: int) -> QuantumCircuit | None:
    """
    Extract circuit from code without full validation.

    Useful for quick testing or when you need the circuit
    regardless of constraint violations.

    Args:
        code: Python code string
        num_qubits: Number of qubits to pass to create_ansatz

    Returns:
        QuantumCircuit if extraction succeeds, None otherwise
    """
    try:
        namespace = {}
        exec(code, namespace)

        if "create_ansatz" not in namespace:
            return None

        create_ansatz = namespace["create_ansatz"]

        try:
            circuit = create_ansatz(num_qubits)
        except TypeError:
            circuit = create_ansatz()

        if isinstance(circuit, QuantumCircuit):
            return circuit

        return None

    except Exception:
        return None


def validate_circuit_hardware_compatible(
    circuit: QuantumCircuit,
    backend_coupling_map: list[tuple[int, int]] | None = None,
    native_gates: frozenset[str] | None = None,
) -> tuple[bool, list[str]]:
    """
    Check if circuit is compatible with target hardware.

    Args:
        circuit: Circuit to validate
        backend_coupling_map: List of (qubit1, qubit2) pairs for allowed connections
        native_gates: Set of gates supported by backend

    Returns:
        Tuple of (is_compatible, list of issues)
    """
    issues = []

    if native_gates is not None:
        ops = set(circuit.count_ops().keys())
        unsupported = ops - native_gates - {"barrier", "measure"}
        if unsupported:
            issues.append(f"Gates not native to backend: {unsupported}")

    if backend_coupling_map is not None:
        coupling_set = set(backend_coupling_map)
        # Add reverse directions
        coupling_set.update((b, a) for a, b in backend_coupling_map)

        for instruction in circuit.data:
            if len(instruction.qubits) == 2:
                q1, q2 = instruction.qubits
                q1_idx = circuit.qubits.index(q1)
                q2_idx = circuit.qubits.index(q2)

                if (q1_idx, q2_idx) not in coupling_set:
                    issues.append(
                        f"Two-qubit gate on qubits ({q1_idx}, {q2_idx}) "
                        f"not in coupling map"
                    )

    return len(issues) == 0, issues


def analyze_code_safety(code: str) -> tuple[bool, list[str]]:
    """
    Basic safety analysis of generated code.

    Checks for potentially dangerous patterns. This is NOT a security
    guarantee - for production, use proper sandboxing.

    Args:
        code: Python code to analyze

    Returns:
        Tuple of (is_safe, list of concerns)
    """
    concerns = []

    # Patterns that might indicate problematic code
    dangerous_patterns = [
        ("import os", "OS operations"),
        ("import sys", "System operations"),
        ("import subprocess", "Process spawning"),
        ("open(", "File operations"),
        ("exec(", "Dynamic execution"),
        ("eval(", "Dynamic evaluation"),
        ("__import__", "Dynamic imports"),
        ("compile(", "Code compilation"),
        ("globals()", "Global access"),
        ("locals()", "Local access"),
        ("getattr(", "Attribute access"),
        ("setattr(", "Attribute modification"),
    ]

    code_lower = code.lower()
    for pattern, description in dangerous_patterns:
        if pattern.lower() in code_lower:
            concerns.append(f"Potentially dangerous: {description} ({pattern})")

    return len(concerns) == 0, concerns


class CircuitVerifier:
    """
    Stateful circuit verifier with configurable defaults.

    Use this class when verifying multiple circuits with the same constraints.
    """

    def __init__(
        self,
        max_depth: int = 100,
        allowed_gates: frozenset[str] | None = None,
        require_params: bool = True,
        min_params: int = 1,
        check_safety: bool = True,
    ):
        """
        Initialize verifier with default constraints.

        Args:
            max_depth: Maximum allowed circuit depth
            allowed_gates: Set of allowed gate names
            require_params: Whether trainable parameters are required
            min_params: Minimum number of parameters
            check_safety: Whether to run safety analysis
        """
        self.max_depth = max_depth
        self.allowed_gates = allowed_gates or DEFAULT_ALLOWED_GATES
        self.require_params = require_params
        self.min_params = min_params
        self.check_safety = check_safety

        # Statistics
        self.total_verified = 0
        self.total_valid = 0

    def verify(self, code: str, num_qubits: int) -> VerificationResult:
        """
        Verify circuit code.

        Args:
            code: Python code string
            num_qubits: Expected number of qubits

        Returns:
            VerificationResult
        """
        self.total_verified += 1

        # Optional safety check
        if self.check_safety:
            is_safe, concerns = analyze_code_safety(code)
            if not is_safe:
                return VerificationResult(
                    is_valid=False,
                    syntax_ok=False,
                    function_ok=False,
                    depth_ok=False,
                    gates_ok=False,
                    params_ok=False,
                    qubit_count_ok=False,
                    errors=[f"Safety concern: {c}" for c in concerns],
                    warnings=[],
                )

        result = verify_circuit_code(
            code=code,
            expected_qubits=num_qubits,
            max_depth=self.max_depth,
            allowed_gates=self.allowed_gates,
            require_params=self.require_params,
            min_params=self.min_params,
        )

        if result.is_valid:
            self.total_valid += 1

        return result

    def get_stats(self) -> dict:
        """Get verification statistics."""
        return {
            "total_verified": self.total_verified,
            "total_valid": self.total_valid,
            "validity_rate": (
                self.total_valid / self.total_verified if self.total_verified > 0 else 0
            ),
        }

    def reset_stats(self):
        """Reset verification statistics."""
        self.total_verified = 0
        self.total_valid = 0
