"""
Tests for the Circuit verifier module.
"""

import pytest

from src.quantum.verifier import (
    CircuitVerifier,
    VerificationResult,
    analyze_code_safety,
    extract_circuit_from_code,
    validate_circuit_hardware_compatible,
    verify_circuit_code,
)


# Test circuit code samples
VALID_CIRCUIT_CODE = '''
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = [Parameter(f"theta_{i}") for i in range(num_qubits)]

    for i in range(num_qubits):
        qc.ry(params[i], i)

    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    return qc
'''

SYNTAX_ERROR_CODE = '''
from qiskit import QuantumCircuit

def create_ansatz(num_qubits: int)  # Missing colon
    qc = QuantumCircuit(num_qubits)
    return qc
'''

NO_FUNCTION_CODE = '''
from qiskit import QuantumCircuit

qc = QuantumCircuit(4)
qc.h(0)
'''

WRONG_RETURN_TYPE_CODE = '''
def create_ansatz(num_qubits: int):
    return "not a circuit"
'''

NO_PARAMS_CODE = '''
from qiskit import QuantumCircuit

def create_ansatz(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc
'''

DEEP_CIRCUIT_CODE = '''
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = [Parameter(f"theta_{i}") for i in range(400)]

    # Creates very deep circuit - 100 layers on each qubit
    for i, p in enumerate(params):
        qc.ry(p, i % num_qubits)

    return qc
'''

FIXED_QUBITS_CODE = '''
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int) -> QuantumCircuit:
    # Ignores num_qubits and always creates 4 qubits
    qc = QuantumCircuit(4)
    params = [Parameter(f"theta_{i}") for i in range(4)]

    for i in range(4):
        qc.ry(params[i], i)

    for i in range(3):
        qc.cx(i, i + 1)

    return qc
'''

INVALID_GATES_CODE = '''
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = [Parameter(f"theta_{i}") for i in range(num_qubits)]

    for i in range(num_qubits):
        qc.ry(params[i], i)

    # CCX (Toffoli) might not be in allowed gates
    qc.ccx(0, 1, 2)

    return qc
'''


class TestVerifyCircuitCode:
    """Tests for verify_circuit_code function."""

    def test_valid_circuit(self):
        """Valid circuit code should pass verification."""
        result = verify_circuit_code(VALID_CIRCUIT_CODE, expected_qubits=4)

        assert result.is_valid
        assert result.syntax_ok
        assert result.function_ok
        assert result.depth_ok
        assert result.gates_ok
        assert result.params_ok
        assert result.qubit_count_ok
        assert result.circuit is not None
        assert len(result.errors) == 0

    def test_syntax_error(self):
        """Syntax errors should be caught."""
        result = verify_circuit_code(SYNTAX_ERROR_CODE, expected_qubits=4)

        assert not result.is_valid
        assert not result.syntax_ok
        assert len(result.errors) > 0
        assert "syntax" in result.errors[0].lower()

    def test_no_function(self):
        """Code without create_ansatz should fail."""
        result = verify_circuit_code(NO_FUNCTION_CODE, expected_qubits=4)

        assert not result.is_valid
        assert result.syntax_ok
        assert not result.function_ok
        assert any("create_ansatz" in e for e in result.errors)

    def test_wrong_return_type(self):
        """Non-QuantumCircuit return should fail."""
        result = verify_circuit_code(WRONG_RETURN_TYPE_CODE, expected_qubits=4)

        assert not result.is_valid
        assert not result.function_ok
        assert any("QuantumCircuit" in e for e in result.errors)

    def test_no_parameters(self):
        """Circuit without parameters should fail when required."""
        result = verify_circuit_code(
            NO_PARAMS_CODE, expected_qubits=4, require_params=True
        )

        assert not result.is_valid
        assert not result.params_ok
        assert any("parameter" in e.lower() for e in result.errors)

    def test_no_parameters_allowed(self):
        """Circuit without parameters should pass when not required."""
        result = verify_circuit_code(
            NO_PARAMS_CODE, expected_qubits=4, require_params=False
        )

        assert result.is_valid
        assert len(result.warnings) > 0  # Should have warning

    def test_depth_exceeded(self):
        """Deep circuit should fail depth check."""
        result = verify_circuit_code(DEEP_CIRCUIT_CODE, expected_qubits=4, max_depth=50)

        assert not result.is_valid
        assert not result.depth_ok
        assert any("depth" in e.lower() for e in result.errors)

    def test_invalid_gates(self):
        """Circuit with invalid gates should fail."""
        allowed = frozenset({"rx", "ry", "rz", "cx", "h"})
        result = verify_circuit_code(
            INVALID_GATES_CODE, expected_qubits=4, allowed_gates=allowed
        )

        assert not result.is_valid
        assert not result.gates_ok
        assert any("gate" in e.lower() for e in result.errors)

    def test_wrong_qubit_count(self):
        """Circuit with wrong qubit count should fail."""
        # FIXED_QUBITS_CODE always creates 4 qubits regardless of input
        result = verify_circuit_code(FIXED_QUBITS_CODE, expected_qubits=8)

        assert not result.is_valid
        assert not result.qubit_count_ok
        assert any("qubit" in e.lower() for e in result.errors)

    def test_details_populated(self):
        """Verification result should have details."""
        result = verify_circuit_code(VALID_CIRCUIT_CODE, expected_qubits=4)

        assert "num_qubits" in result.details
        assert "depth" in result.details
        assert "num_params" in result.details
        assert "gate_counts" in result.details


class TestExtractCircuit:
    """Tests for extract_circuit_from_code function."""

    def test_extract_valid(self):
        """Should extract circuit from valid code."""
        circuit = extract_circuit_from_code(VALID_CIRCUIT_CODE, num_qubits=4)

        assert circuit is not None
        assert circuit.num_qubits == 4

    def test_extract_invalid(self):
        """Should return None for invalid code."""
        circuit = extract_circuit_from_code(SYNTAX_ERROR_CODE, num_qubits=4)

        assert circuit is None

    def test_extract_no_function(self):
        """Should return None if no create_ansatz."""
        circuit = extract_circuit_from_code(NO_FUNCTION_CODE, num_qubits=4)

        assert circuit is None


class TestHardwareCompatibility:
    """Tests for validate_circuit_hardware_compatible function."""

    def test_compatible_circuit(self):
        """Circuit with linear connectivity should be compatible."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)

        coupling_map = [(0, 1), (1, 2)]
        is_compatible, issues = validate_circuit_hardware_compatible(
            qc, backend_coupling_map=coupling_map
        )

        assert is_compatible
        assert len(issues) == 0

    def test_incompatible_connectivity(self):
        """Circuit with non-adjacent CX should be flagged."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3)
        qc.cx(0, 2)  # Skips qubit 1

        coupling_map = [(0, 1), (1, 2)]
        is_compatible, issues = validate_circuit_hardware_compatible(
            qc, backend_coupling_map=coupling_map
        )

        assert not is_compatible
        assert len(issues) > 0

    def test_unsupported_gates(self):
        """Circuit with non-native gates should be flagged."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.ccx(0, 1, 2)  # Toffoli

        native_gates = frozenset({"rx", "ry", "rz", "cx"})
        is_compatible, issues = validate_circuit_hardware_compatible(
            qc, native_gates=native_gates
        )

        assert not is_compatible
        assert len(issues) > 0


class TestCodeSafety:
    """Tests for analyze_code_safety function."""

    def test_safe_code(self):
        """Normal circuit code should be safe."""
        is_safe, concerns = analyze_code_safety(VALID_CIRCUIT_CODE)

        assert is_safe
        assert len(concerns) == 0

    def test_os_import(self):
        """Code with os import should be flagged."""
        code = '''
import os
from qiskit import QuantumCircuit

def create_ansatz(num_qubits):
    os.system("rm -rf /")
    return QuantumCircuit(num_qubits)
'''
        is_safe, concerns = analyze_code_safety(code)

        assert not is_safe
        assert len(concerns) > 0

    def test_file_operations(self):
        """Code with file operations should be flagged."""
        code = '''
from qiskit import QuantumCircuit

def create_ansatz(num_qubits):
    with open("/etc/passwd") as f:
        pass
    return QuantumCircuit(num_qubits)
'''
        is_safe, concerns = analyze_code_safety(code)

        assert not is_safe
        assert len(concerns) > 0

    def test_eval(self):
        """Code with eval should be flagged."""
        code = '''
from qiskit import QuantumCircuit

def create_ansatz(num_qubits):
    eval("malicious code")
    return QuantumCircuit(num_qubits)
'''
        is_safe, concerns = analyze_code_safety(code)

        assert not is_safe
        assert len(concerns) > 0


class TestCircuitVerifier:
    """Tests for CircuitVerifier class."""

    def test_basic_verification(self):
        """Basic verification should work."""
        verifier = CircuitVerifier()
        result = verifier.verify(VALID_CIRCUIT_CODE, num_qubits=4)

        assert result.is_valid

    def test_stats_tracking(self):
        """Verifier should track statistics."""
        verifier = CircuitVerifier()

        verifier.verify(VALID_CIRCUIT_CODE, num_qubits=4)
        verifier.verify(SYNTAX_ERROR_CODE, num_qubits=4)
        verifier.verify(VALID_CIRCUIT_CODE, num_qubits=4)

        stats = verifier.get_stats()
        assert stats["total_verified"] == 3
        assert stats["total_valid"] == 2
        assert abs(stats["validity_rate"] - 2 / 3) < 0.01

    def test_reset_stats(self):
        """Stats should be resettable."""
        verifier = CircuitVerifier()
        verifier.verify(VALID_CIRCUIT_CODE, num_qubits=4)

        verifier.reset_stats()
        stats = verifier.get_stats()

        assert stats["total_verified"] == 0
        assert stats["total_valid"] == 0

    def test_safety_check_blocks_dangerous(self):
        """Safety check should block dangerous code."""
        verifier = CircuitVerifier(check_safety=True)

        dangerous_code = '''
import os
from qiskit import QuantumCircuit

def create_ansatz(num_qubits):
    return QuantumCircuit(num_qubits)
'''
        result = verifier.verify(dangerous_code, num_qubits=4)

        assert not result.is_valid
        assert any("safety" in e.lower() for e in result.errors)

    def test_custom_constraints(self):
        """Custom constraints should be respected."""
        verifier = CircuitVerifier(
            max_depth=10,
            min_params=5,
            allowed_gates=frozenset({"ry", "cx"}),
        )

        result = verifier.verify(VALID_CIRCUIT_CODE, num_qubits=4)

        # Valid code has 4 params, less than min_params=5
        # Should fail
        assert not result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
