"""
Baseline ansatzes for comparison in QuantumMind.

Provides standard quantum circuit templates (HEA, EfficientSU2, etc.)
for benchmarking discovered circuits.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def hardware_efficient_ansatz(
    num_qubits: int,
    num_layers: int = 2,
    rotation_gates: list[str] | None = None,
    entanglement: str = "linear",
) -> QuantumCircuit:
    """
    Create a hardware-efficient ansatz (HEA).

    Standard variational form with alternating rotation and entanglement layers.

    Args:
        num_qubits: Number of qubits
        num_layers: Number of layer repetitions
        rotation_gates: List of rotation gates (default: ["ry", "rz"])
        entanglement: Entanglement pattern ("linear", "full", "circular")

    Returns:
        Parameterized QuantumCircuit
    """
    if rotation_gates is None:
        rotation_gates = ["ry", "rz"]

    qc = QuantumCircuit(num_qubits)
    param_count = 0

    for layer in range(num_layers):
        # Rotation block
        for gate in rotation_gates:
            for qubit in range(num_qubits):
                p = Parameter(f"theta_{param_count}")
                param_count += 1

                if gate == "rx":
                    qc.rx(p, qubit)
                elif gate == "ry":
                    qc.ry(p, qubit)
                elif gate == "rz":
                    qc.rz(p, qubit)

        # Entanglement block
        if entanglement == "linear":
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
        elif entanglement == "full":
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    qc.cx(i, j)
        elif entanglement == "circular":
            for i in range(num_qubits):
                qc.cx(i, (i + 1) % num_qubits)

    # Final rotation layer
    for gate in rotation_gates:
        for qubit in range(num_qubits):
            p = Parameter(f"theta_{param_count}")
            param_count += 1

            if gate == "rx":
                qc.rx(p, qubit)
            elif gate == "ry":
                qc.ry(p, qubit)
            elif gate == "rz":
                qc.rz(p, qubit)

    return qc


def efficient_su2_ansatz(
    num_qubits: int,
    num_layers: int = 2,
    entanglement: str = "full",
) -> QuantumCircuit:
    """
    Create an EfficientSU2 ansatz.

    Uses RY-RZ rotations for SU(2) coverage with configurable entanglement.

    Args:
        num_qubits: Number of qubits
        num_layers: Number of layer repetitions
        entanglement: Entanglement pattern ("linear", "full", "circular")

    Returns:
        Parameterized QuantumCircuit
    """
    qc = QuantumCircuit(num_qubits)
    param_count = 0

    for layer in range(num_layers):
        # RY rotations
        for qubit in range(num_qubits):
            p = Parameter(f"theta_{param_count}")
            param_count += 1
            qc.ry(p, qubit)

        # RZ rotations
        for qubit in range(num_qubits):
            p = Parameter(f"theta_{param_count}")
            param_count += 1
            qc.rz(p, qubit)

        # Entanglement
        if entanglement == "linear":
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
        elif entanglement == "full":
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    qc.cx(i, j)
        elif entanglement == "circular":
            for i in range(num_qubits):
                qc.cx(i, (i + 1) % num_qubits)

    # Final rotation layer
    for qubit in range(num_qubits):
        p = Parameter(f"theta_{param_count}")
        param_count += 1
        qc.ry(p, qubit)

    for qubit in range(num_qubits):
        p = Parameter(f"theta_{param_count}")
        param_count += 1
        qc.rz(p, qubit)

    return qc


def real_amplitudes_ansatz(
    num_qubits: int,
    num_layers: int = 2,
    entanglement: str = "linear",
) -> QuantumCircuit:
    """
    Create a RealAmplitudes ansatz.

    Uses only RY rotations, producing real-valued amplitudes.
    Good for problems with real ground states.

    Args:
        num_qubits: Number of qubits
        num_layers: Number of layer repetitions
        entanglement: Entanglement pattern

    Returns:
        Parameterized QuantumCircuit
    """
    qc = QuantumCircuit(num_qubits)
    param_count = 0

    for layer in range(num_layers):
        # RY rotations
        for qubit in range(num_qubits):
            p = Parameter(f"theta_{param_count}")
            param_count += 1
            qc.ry(p, qubit)

        # Entanglement
        if entanglement == "linear":
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
        elif entanglement == "full":
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    qc.cx(i, j)
        elif entanglement == "circular":
            for i in range(num_qubits):
                qc.cx(i, (i + 1) % num_qubits)

    # Final RY layer
    for qubit in range(num_qubits):
        p = Parameter(f"theta_{param_count}")
        param_count += 1
        qc.ry(p, qubit)

    return qc


def excitation_preserving_ansatz(
    num_qubits: int,
    num_layers: int = 2,
) -> QuantumCircuit:
    """
    Create an excitation-preserving ansatz.

    Preserves the number of excitations (|1> states), useful for
    problems with particle number conservation.

    Args:
        num_qubits: Number of qubits
        num_layers: Number of layer repetitions

    Returns:
        Parameterized QuantumCircuit
    """
    qc = QuantumCircuit(num_qubits)
    param_count = 0

    for layer in range(num_layers):
        # Single-qubit Z rotations
        for qubit in range(num_qubits):
            p = Parameter(f"theta_{param_count}")
            param_count += 1
            qc.rz(p, qubit)

        # XX+YY interactions (excitation-preserving)
        for i in range(num_qubits - 1):
            p = Parameter(f"theta_{param_count}")
            param_count += 1

            # iSWAP-like gate: exp(-i*theta*(XX+YY)/2)
            qc.rxx(p, i, i + 1)
            qc.ryy(p, i, i + 1)

    # Final Z rotations
    for qubit in range(num_qubits):
        p = Parameter(f"theta_{param_count}")
        param_count += 1
        qc.rz(p, qubit)

    return qc


def two_local_ansatz(
    num_qubits: int,
    num_layers: int = 2,
    rotation_blocks: list[str] | None = None,
    entanglement_blocks: str = "cx",
    entanglement: str = "linear",
) -> QuantumCircuit:
    """
    Create a generic TwoLocal ansatz.

    Flexible ansatz with configurable rotation and entanglement gates.

    Args:
        num_qubits: Number of qubits
        num_layers: Number of layers
        rotation_blocks: Rotation gates (default: ["ry"])
        entanglement_blocks: Two-qubit gate ("cx", "cz", "swap")
        entanglement: Entanglement pattern

    Returns:
        Parameterized QuantumCircuit
    """
    if rotation_blocks is None:
        rotation_blocks = ["ry"]

    qc = QuantumCircuit(num_qubits)
    param_count = 0

    for layer in range(num_layers):
        # Rotation block
        for gate in rotation_blocks:
            for qubit in range(num_qubits):
                p = Parameter(f"theta_{param_count}")
                param_count += 1

                if gate == "rx":
                    qc.rx(p, qubit)
                elif gate == "ry":
                    qc.ry(p, qubit)
                elif gate == "rz":
                    qc.rz(p, qubit)
                elif gate == "h":
                    qc.h(qubit)

        # Entanglement block
        pairs = []
        if entanglement == "linear":
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
        elif entanglement == "full":
            pairs = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        elif entanglement == "circular":
            pairs = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]

        for i, j in pairs:
            if entanglement_blocks == "cx":
                qc.cx(i, j)
            elif entanglement_blocks == "cz":
                qc.cz(i, j)
            elif entanglement_blocks == "swap":
                qc.swap(i, j)

    # Final rotation
    for gate in rotation_blocks:
        for qubit in range(num_qubits):
            p = Parameter(f"theta_{param_count}")
            param_count += 1

            if gate == "rx":
                qc.rx(p, qubit)
            elif gate == "ry":
                qc.ry(p, qubit)
            elif gate == "rz":
                qc.rz(p, qubit)

    return qc


def get_baseline(
    name: str,
    num_qubits: int,
    num_layers: int = 2,
    **kwargs,
) -> QuantumCircuit:
    """
    Get a baseline ansatz by name.

    Args:
        name: Baseline name (HEA, SU2, REAL_AMP, EXCITATION, TWO_LOCAL)
        num_qubits: Number of qubits
        num_layers: Number of layers
        **kwargs: Additional arguments for specific baselines

    Returns:
        Parameterized QuantumCircuit
    """
    name_upper = name.upper()

    baselines = {
        "HEA": hardware_efficient_ansatz,
        "HARDWARE_EFFICIENT": hardware_efficient_ansatz,
        "SU2": efficient_su2_ansatz,
        "EFFICIENT_SU2": efficient_su2_ansatz,
        "REAL_AMP": real_amplitudes_ansatz,
        "REAL_AMPLITUDES": real_amplitudes_ansatz,
        "EXCITATION": excitation_preserving_ansatz,
        "EXCITATION_PRESERVING": excitation_preserving_ansatz,
        "TWO_LOCAL": two_local_ansatz,
    }

    if name_upper not in baselines:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(baselines.keys())}")

    return baselines[name_upper](num_qubits, num_layers, **kwargs)


def list_baselines() -> list[str]:
    """Return list of available baseline names."""
    return ["HEA", "SU2", "REAL_AMP", "EXCITATION", "TWO_LOCAL"]


def get_circuit_info(circuit: QuantumCircuit) -> dict:
    """
    Get information about a circuit.

    Args:
        circuit: QuantumCircuit to analyze

    Returns:
        Dictionary with circuit information
    """
    ops = circuit.count_ops()

    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "num_params": len(circuit.parameters),
        "gate_count": sum(ops.values()),
        "cx_count": ops.get("cx", 0),
        "single_qubit_gates": sum(
            v for k, v in ops.items() if k not in ["cx", "cz", "swap", "rxx", "ryy"]
        ),
        "two_qubit_gates": sum(
            ops.get(g, 0) for g in ["cx", "cz", "swap", "rxx", "ryy"]
        ),
        "ops": dict(ops),
    }


def compare_baselines(
    num_qubits: int,
    num_layers: int = 2,
) -> dict[str, dict]:
    """
    Compare all baseline ansatzes.

    Args:
        num_qubits: Number of qubits
        num_layers: Number of layers

    Returns:
        Dictionary mapping baseline names to their info
    """
    results = {}

    for name in list_baselines():
        try:
            circuit = get_baseline(name, num_qubits, num_layers)
            results[name] = get_circuit_info(circuit)
        except Exception as e:
            results[name] = {"error": str(e)}

    return results
