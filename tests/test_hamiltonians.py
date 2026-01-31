"""
Tests for the Hamiltonians module.
"""

import numpy as np
import pytest

from src.quantum.hamiltonians import (
    get_hamiltonian,
    h2_molecule,
    heisenberg_chain,
    list_hamiltonians,
    transverse_ising,
    xy_chain,
)


class TestXYChain:
    """Tests for XY chain Hamiltonian."""

    def test_basic_creation(self):
        """Test basic XY chain creation."""
        result = xy_chain(4)
        assert result.num_qubits == 4
        assert result.operator is not None
        assert result.exact_energy is not None
        assert "XY" in result.name

    def test_ground_state_energy_is_negative(self):
        """Ground state of ferromagnetic XY should be negative."""
        for n in [2, 3, 4, 5, 6]:
            result = xy_chain(n, J=1.0)
            assert result.exact_energy < 0, f"Ground state energy should be negative for n={n}"

    def test_operator_is_hermitian(self):
        """Hamiltonian must be Hermitian."""
        result = xy_chain(4)
        matrix = result.operator.to_matrix()
        assert np.allclose(matrix, matrix.conj().T), "Hamiltonian is not Hermitian"

    def test_exact_state_is_normalized(self):
        """Exact state should be normalized."""
        result = xy_chain(4)
        assert np.isclose(np.linalg.norm(result.exact_state), 1.0)

    def test_exact_state_is_eigenstate(self):
        """Exact state should be an eigenstate with correct eigenvalue."""
        result = xy_chain(4)
        matrix = result.operator.to_matrix()
        Hpsi = matrix @ result.exact_state
        Epsi = result.exact_energy * result.exact_state
        assert np.allclose(Hpsi, Epsi), "Exact state is not an eigenstate"

    def test_periodic_boundary(self):
        """Test periodic boundary conditions."""
        open_result = xy_chain(4, periodic=False)
        periodic_result = xy_chain(4, periodic=True)

        # Periodic should have more terms (additional bond)
        assert len(periodic_result.operator) > len(open_result.operator)

    def test_minimum_qubits(self):
        """Should raise error for less than 2 qubits."""
        with pytest.raises(ValueError):
            xy_chain(1)

    def test_coupling_strength(self):
        """Test that coupling strength affects energy."""
        result_J1 = xy_chain(4, J=1.0)
        result_J2 = xy_chain(4, J=2.0)

        # Doubling J should double the energy
        assert np.isclose(result_J2.exact_energy, 2 * result_J1.exact_energy)


class TestHeisenbergChain:
    """Tests for Heisenberg chain Hamiltonian."""

    def test_basic_creation(self):
        """Test basic Heisenberg chain creation."""
        result = heisenberg_chain(4)
        assert result.num_qubits == 4
        assert result.operator is not None
        assert "Heisenberg" in result.name

    def test_operator_is_hermitian(self):
        """Hamiltonian must be Hermitian."""
        result = heisenberg_chain(4)
        matrix = result.operator.to_matrix()
        assert np.allclose(matrix, matrix.conj().T)

    def test_antiferromagnetic_ground_state(self):
        """AFM Heisenberg should have negative ground state energy."""
        result = heisenberg_chain(4, J=1.0)
        assert result.exact_energy < 0

    def test_different_qubit_counts(self):
        """Test various qubit counts."""
        for n in [2, 3, 4, 5, 6]:
            result = heisenberg_chain(n)
            assert result.num_qubits == n
            assert result.exact_state is not None

    def test_minimum_qubits(self):
        """Should raise error for less than 2 qubits."""
        with pytest.raises(ValueError):
            heisenberg_chain(1)


class TestTransverseIsing:
    """Tests for Transverse-Field Ising Model."""

    def test_basic_creation(self):
        """Test basic TFIM creation."""
        result = transverse_ising(4)
        assert result.num_qubits == 4
        assert result.operator is not None
        assert "TFIM" in result.name

    def test_operator_is_hermitian(self):
        """Hamiltonian must be Hermitian."""
        result = transverse_ising(4)
        matrix = result.operator.to_matrix()
        assert np.allclose(matrix, matrix.conj().T)

    def test_field_strength_affects_energy(self):
        """Different field strengths should give different energies."""
        result_h1 = transverse_ising(4, J=1.0, h=0.5)
        result_h2 = transverse_ising(4, J=1.0, h=2.0)

        assert result_h1.exact_energy != result_h2.exact_energy

    def test_limiting_cases(self):
        """Test limiting cases of TFIM."""
        # Strong field limit (h >> J): ground state should be all spins aligned with field
        result_strong_field = transverse_ising(4, J=0.01, h=10.0)
        # Energy should be approximately -h * n
        expected_approx = -10.0 * 4
        assert np.isclose(result_strong_field.exact_energy, expected_approx, rtol=0.1)

    def test_minimum_qubits(self):
        """Should raise error for less than 2 qubits."""
        with pytest.raises(ValueError):
            transverse_ising(1)


class TestH2Molecule:
    """Tests for H2 molecule Hamiltonian."""

    def test_basic_creation(self):
        """Test basic H2 creation."""
        result = h2_molecule()
        assert result.num_qubits == 4
        assert result.operator is not None
        assert "H2" in result.name

    def test_operator_is_hermitian(self):
        """Hamiltonian must be Hermitian."""
        result = h2_molecule()
        matrix = result.operator.to_matrix()
        assert np.allclose(matrix, matrix.conj().T)

    def test_ground_state_energy_reasonable(self):
        """Ground state energy should be in expected range for H2."""
        result = h2_molecule()
        # H2 ground state energy at equilibrium is approximately -1.1 to -1.2 Hartree
        assert -2.0 < result.exact_energy < 0.0


class TestGetHamiltonian:
    """Tests for the get_hamiltonian factory function."""

    def test_get_xy_chain(self):
        """Test getting XY chain by name."""
        result = get_hamiltonian("XY_CHAIN", num_qubits=4)
        assert result.num_qubits == 4
        assert "XY" in result.name

    def test_get_heisenberg(self):
        """Test getting Heisenberg by name."""
        result = get_hamiltonian("HEISENBERG", num_qubits=4)
        assert result.num_qubits == 4

    def test_get_tfim(self):
        """Test getting TFIM by name."""
        result = get_hamiltonian("TFIM", num_qubits=4, J=1.0, h=0.5)
        assert result.num_qubits == 4

    def test_get_h2(self):
        """Test getting H2 (no num_qubits needed)."""
        result = get_hamiltonian("H2")
        assert result.num_qubits == 4

    def test_unknown_hamiltonian(self):
        """Should raise error for unknown Hamiltonian."""
        with pytest.raises(ValueError):
            get_hamiltonian("UNKNOWN", num_qubits=4)

    def test_missing_num_qubits(self):
        """Should raise error when num_qubits not provided."""
        with pytest.raises(ValueError):
            get_hamiltonian("XY_CHAIN")


class TestListHamiltonians:
    """Tests for list_hamiltonians function."""

    def test_returns_list(self):
        """Should return a list."""
        result = list_hamiltonians()
        assert isinstance(result, list)

    def test_contains_expected_hamiltonians(self):
        """Should contain expected Hamiltonian names."""
        result = list_hamiltonians()
        assert "XY_CHAIN" in result
        assert "HEISENBERG" in result
        assert "TFIM" in result
        assert "H2" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
