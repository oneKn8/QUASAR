"""
Hamiltonian definitions for target physics problems.

This module provides Hamiltonians for:
- XY Spin Chain
- Heisenberg XXX Model
- Transverse-Field Ising Model (TFIM)

Each Hamiltonian returns a HamiltonianResult containing:
- The operator (SparsePauliOp)
- Exact ground state energy
- Metadata for logging
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from qiskit.quantum_info import SparsePauliOp


@dataclass
class HamiltonianResult:
    """Container for Hamiltonian and exact solution."""

    operator: SparsePauliOp
    exact_energy: float
    exact_state: np.ndarray | None = None
    name: str = ""
    num_qubits: int = 0
    description: str = ""


def xy_chain(num_qubits: int, J: float = 1.0, periodic: bool = False) -> HamiltonianResult:
    """
    1D XY spin chain Hamiltonian.

    H = -J * sum_i (X_i X_{i+1} + Y_i Y_{i+1})

    The XY model describes spin-1/2 particles with nearest-neighbor
    interactions in the XY plane. It has U(1) symmetry (conserves total
    Z magnetization).

    Args:
        num_qubits: Number of qubits/spins in the chain
        J: Coupling strength (positive = ferromagnetic)
        periodic: If True, use periodic boundary conditions

    Returns:
        HamiltonianResult with operator and exact energy

    Raises:
        ValueError: If num_qubits < 2
    """
    if num_qubits < 2:
        raise ValueError("XY chain requires at least 2 qubits")

    terms = []
    num_bonds = num_qubits if periodic else num_qubits - 1

    for i in range(num_bonds):
        j = (i + 1) % num_qubits  # Handles periodic boundary

        # XX term
        xx = ["I"] * num_qubits
        xx[i] = "X"
        xx[j] = "X"
        terms.append(("".join(xx), -J))

        # YY term
        yy = ["I"] * num_qubits
        yy[i] = "Y"
        yy[j] = "Y"
        terms.append(("".join(yy), -J))

    operator = SparsePauliOp.from_list(terms)

    # Compute exact energy by diagonalization
    matrix = operator.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    exact_energy = eigenvalues[0]
    exact_state = eigenvectors[:, 0]

    boundary = "periodic" if periodic else "open"
    return HamiltonianResult(
        operator=operator,
        exact_energy=exact_energy,
        exact_state=exact_state,
        name=f"XY Chain ({num_qubits} qubits, {boundary})",
        num_qubits=num_qubits,
        description=f"1D XY model with J={J}, {boundary} boundary conditions",
    )


def heisenberg_chain(
    num_qubits: int, J: float = 1.0, periodic: bool = False
) -> HamiltonianResult:
    """
    1D Heisenberg XXX model.

    H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})

    The Heisenberg model describes isotropic spin-spin interactions.
    It has SU(2) symmetry (total spin is conserved).

    Args:
        num_qubits: Number of qubits/spins in the chain
        J: Coupling strength (positive = antiferromagnetic)
        periodic: If True, use periodic boundary conditions

    Returns:
        HamiltonianResult with operator and exact energy

    Raises:
        ValueError: If num_qubits < 2
    """
    if num_qubits < 2:
        raise ValueError("Heisenberg chain requires at least 2 qubits")

    terms = []
    num_bonds = num_qubits if periodic else num_qubits - 1

    for i in range(num_bonds):
        j = (i + 1) % num_qubits

        for pauli in ["X", "Y", "Z"]:
            term = ["I"] * num_qubits
            term[i] = pauli
            term[j] = pauli
            terms.append(("".join(term), J))

    operator = SparsePauliOp.from_list(terms)

    # Compute exact energy by diagonalization
    matrix = operator.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    exact_energy = eigenvalues[0]
    exact_state = eigenvectors[:, 0]

    boundary = "periodic" if periodic else "open"
    return HamiltonianResult(
        operator=operator,
        exact_energy=exact_energy,
        exact_state=exact_state,
        name=f"Heisenberg Chain ({num_qubits} qubits, {boundary})",
        num_qubits=num_qubits,
        description=f"1D Heisenberg XXX model with J={J}, {boundary} boundary conditions",
    )


def transverse_ising(
    num_qubits: int, J: float = 1.0, h: float = 1.0, periodic: bool = False
) -> HamiltonianResult:
    """
    Transverse-field Ising model (TFIM).

    H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i

    The TFIM describes Ising interactions with a transverse magnetic field.
    It exhibits a quantum phase transition at h/J = 1 (in 1D infinite limit).

    Args:
        num_qubits: Number of qubits/spins in the chain
        J: Ising coupling strength
        h: Transverse field strength
        periodic: If True, use periodic boundary conditions

    Returns:
        HamiltonianResult with operator and exact energy

    Raises:
        ValueError: If num_qubits < 2
    """
    if num_qubits < 2:
        raise ValueError("TFIM requires at least 2 qubits")

    terms = []
    num_bonds = num_qubits if periodic else num_qubits - 1

    # ZZ interactions
    for i in range(num_bonds):
        j = (i + 1) % num_qubits
        zz = ["I"] * num_qubits
        zz[i] = "Z"
        zz[j] = "Z"
        terms.append(("".join(zz), -J))

    # Transverse field
    for i in range(num_qubits):
        x = ["I"] * num_qubits
        x[i] = "X"
        terms.append(("".join(x), -h))

    operator = SparsePauliOp.from_list(terms)

    # Compute exact energy by diagonalization
    matrix = operator.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    exact_energy = eigenvalues[0]
    exact_state = eigenvectors[:, 0]

    boundary = "periodic" if periodic else "open"
    return HamiltonianResult(
        operator=operator,
        exact_energy=exact_energy,
        exact_state=exact_state,
        name=f"TFIM ({num_qubits} qubits, {boundary})",
        num_qubits=num_qubits,
        description=f"Transverse-field Ising model with J={J}, h={h}, {boundary} boundary",
    )


def h2_molecule(bond_length: float = 0.735) -> HamiltonianResult:
    """
    Hydrogen molecule (H2) Hamiltonian in minimal basis (STO-3G).

    This is a 4-qubit Hamiltonian representing H2 in second quantization,
    mapped to qubits via Jordan-Wigner transformation.

    Args:
        bond_length: H-H bond length in Angstroms (equilibrium ~0.735)

    Returns:
        HamiltonianResult with operator and approximate exact energy

    Note:
        This is a simplified version. For production use, consider
        qiskit-nature for accurate molecular Hamiltonians.
    """
    # Simplified H2 Hamiltonian at equilibrium geometry
    # Coefficients from literature for STO-3G basis
    # This is an approximation - for exact coefficients use qiskit-nature

    terms = [
        ("IIII", -0.8105),
        ("IIIZ", 0.1721),
        ("IIZI", -0.2257),
        ("IZII", 0.1721),
        ("ZIII", -0.2257),
        ("IIZZ", 0.1209),
        ("IZIZ", 0.1689),
        ("IZZI", 0.0453),
        ("ZIIZ", 0.0453),
        ("ZIZI", 0.1689),
        ("ZZII", 0.1209),
        ("XXYY", -0.0453),
        ("XYYX", 0.0453),
        ("YXXY", 0.0453),
        ("YYXX", -0.0453),
    ]

    operator = SparsePauliOp.from_list(terms)

    # Exact ground state energy for H2 at equilibrium
    matrix = operator.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    exact_energy = eigenvalues[0]
    exact_state = eigenvectors[:, 0]

    return HamiltonianResult(
        operator=operator,
        exact_energy=exact_energy,
        exact_state=exact_state,
        name=f"H2 Molecule (bond={bond_length}A)",
        num_qubits=4,
        description=f"Hydrogen molecule in STO-3G basis, bond length {bond_length} Angstroms",
    )


# Registry of available Hamiltonians
HAMILTONIANS: dict[str, Callable] = {
    "XY_CHAIN": xy_chain,
    "HEISENBERG": heisenberg_chain,
    "TFIM": transverse_ising,
    "H2": h2_molecule,
}


def get_hamiltonian(name: str, num_qubits: int | None = None, **kwargs) -> HamiltonianResult:
    """
    Get Hamiltonian by name.

    Args:
        name: Hamiltonian name (XY_CHAIN, HEISENBERG, TFIM, H2) - case insensitive
        num_qubits: Number of qubits (not needed for H2)
        **kwargs: Additional arguments for specific Hamiltonians

    Returns:
        HamiltonianResult

    Raises:
        ValueError: If name is unknown or num_qubits not provided when needed
    """
    # Normalize name to uppercase
    name_upper = name.upper()

    if name_upper not in HAMILTONIANS:
        raise ValueError(f"Unknown Hamiltonian: {name}. Available: {list(HAMILTONIANS.keys())}")

    if name_upper == "H2":
        return HAMILTONIANS[name_upper](**kwargs)

    if num_qubits is None:
        raise ValueError(f"num_qubits required for {name}")

    return HAMILTONIANS[name_upper](num_qubits, **kwargs)


def list_hamiltonians() -> list[str]:
    """Return list of available Hamiltonian names."""
    return list(HAMILTONIANS.keys())
