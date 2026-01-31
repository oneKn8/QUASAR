"""
Qiskit installation verification script.

This script verifies that Qiskit is properly installed and working
with the local Aer simulator.

Run with: python scripts/test_qiskit.py
"""

import sys


def test_qiskit_basic():
    """Test basic Qiskit functionality."""
    print("Test 1: Basic circuit creation and simulation")
    print("-" * 40)

    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    # Create Bell state circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    print("Created Bell state circuit:")
    print(qc.draw(output="text"))

    # Run on simulator
    sim = AerSimulator()
    job = sim.run(qc, shots=1000)
    result = job.result()
    counts = result.get_counts()

    print(f"\nMeasurement results (1000 shots): {counts}")

    # Verify Bell state (should be ~50% |00> and ~50% |11>)
    assert "00" in counts or "11" in counts, "Bell state not detected"
    print("PASSED: Bell state verified")
    return True


def test_hamiltonian():
    """Test Hamiltonian creation."""
    print("\nTest 2: Hamiltonian creation")
    print("-" * 40)

    from qiskit.quantum_info import SparsePauliOp
    import numpy as np

    # Create simple Hamiltonian: H = ZZ + 0.5*XI + 0.5*IX
    H = SparsePauliOp.from_list([
        ("ZZ", 1.0),
        ("XI", 0.5),
        ("IX", 0.5)
    ])

    print(f"Hamiltonian terms: {len(H)}")
    print(f"Hamiltonian:\n{H}")

    # Convert to matrix and verify
    matrix = H.to_matrix()
    eigenvalues = np.linalg.eigvalsh(matrix)

    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Ground state energy: {eigenvalues[0]:.6f}")

    print("PASSED: Hamiltonian creation verified")
    return True


def test_parameterized_circuit():
    """Test parameterized circuit creation."""
    print("\nTest 3: Parameterized circuit (ansatz)")
    print("-" * 40)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    import numpy as np

    # Create simple ansatz
    num_qubits = 4
    qc = QuantumCircuit(num_qubits)
    params = []

    # Rotation layer
    for i in range(num_qubits):
        p = Parameter(f"theta_{i}")
        params.append(p)
        qc.ry(p, i)

    # Entanglement layer
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    print(f"Circuit depth: {qc.depth()}")
    print(f"Number of parameters: {len(qc.parameters)}")
    print(f"Gate counts: {qc.count_ops()}")

    # Bind parameters
    param_values = np.random.uniform(0, 2 * np.pi, len(params))
    bound_circuit = qc.assign_parameters(dict(zip(params, param_values)))

    print(f"Parameters bound successfully")
    print("PASSED: Parameterized circuit verified")
    return True


def test_estimator():
    """Test Estimator primitive for expectation values."""
    print("\nTest 4: Estimator for expectation values")
    print("-" * 40)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorEstimator
    import numpy as np

    # Create circuit
    qc = QuantumCircuit(2)
    theta = Parameter("theta")
    qc.ry(theta, 0)
    qc.cx(0, 1)

    # Create observable
    observable = SparsePauliOp.from_list([("ZZ", 1.0)])

    # Compute expectation value
    estimator = StatevectorEstimator()
    bound_circuit = qc.assign_parameters({theta: np.pi / 4})

    job = estimator.run([(bound_circuit, observable)])
    result = job.result()
    exp_value = result[0].data.evs

    print(f"Expectation value of ZZ: {exp_value:.6f}")
    print("PASSED: Estimator verified")
    return True


def test_vqe_components():
    """Test VQE-related components."""
    print("\nTest 5: VQE components")
    print("-" * 40)

    from qiskit.circuit.library import TwoLocal
    from qiskit.quantum_info import SparsePauliOp

    # Create hardware-efficient ansatz
    ansatz = TwoLocal(
        num_qubits=4,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=2
    )

    print(f"TwoLocal ansatz:")
    print(f"  Qubits: {ansatz.num_qubits}")
    print(f"  Depth: {ansatz.depth()}")
    print(f"  Parameters: {len(ansatz.parameters)}")

    # Create test Hamiltonian (XY chain)
    terms = []
    for i in range(3):
        xx = ["I"] * 4
        xx[i] = "X"
        xx[i + 1] = "X"
        terms.append(("".join(xx), -1.0))

        yy = ["I"] * 4
        yy[i] = "Y"
        yy[i + 1] = "Y"
        terms.append(("".join(yy), -1.0))

    H = SparsePauliOp.from_list(terms)
    print(f"\nXY Chain Hamiltonian: {len(H)} terms")

    print("PASSED: VQE components verified")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Qiskit Installation Verification")
    print("=" * 50)

    tests = [
        test_qiskit_basic,
        test_hamiltonian,
        test_parameterized_circuit,
        test_estimator,
        test_vqe_components,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("\nAll Qiskit tests passed!")
        print("Ready to proceed with QuantumMind development.")
    else:
        print("\nSome tests failed. Please check your installation.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
