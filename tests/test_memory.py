"""
Tests for the memory module.
"""

import os
import tempfile

import pytest

from src.agent.memory import (
    AgentMemory,
    CircuitRecord,
    MemoryConfig,
    create_feedback,
)


class TestCircuitRecord:
    """Tests for CircuitRecord dataclass."""

    def test_basic_creation(self):
        """Test basic record creation."""
        record = CircuitRecord(
            circuit_id="test_001",
            code="def create_ansatz(): pass",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            energy=-2.5,
            energy_error=0.01,
            is_valid=True,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
            reasoning="Test reasoning",
            feedback="Good circuit",
            timestamp="2026-01-30T12:00:00",
        )

        assert record.circuit_id == "test_001"
        assert record.num_qubits == 4
        assert record.energy == -2.5
        assert record.is_valid is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = CircuitRecord(
            circuit_id="test_001",
            code="code",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            energy=None,
            energy_error=None,
            is_valid=True,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
            reasoning="",
            feedback="",
            timestamp="2026-01-30",
        )

        d = record.to_dict()
        assert isinstance(d, dict)
        assert d["circuit_id"] == "test_001"
        assert d["num_qubits"] == 4

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "circuit_id": "test_002",
            "code": "def create_ansatz(): pass",
            "hamiltonian_type": "heisenberg",
            "num_qubits": 6,
            "energy": -3.0,
            "energy_error": 0.02,
            "is_valid": True,
            "is_trainable": False,
            "depth": 10,
            "param_count": 12,
            "gate_count": 20,
            "reasoning": "Test",
            "feedback": "BP issue",
            "timestamp": "2026-01-30",
            "metadata": {"test": True},
        }

        record = CircuitRecord.from_dict(data)
        assert record.circuit_id == "test_002"
        assert record.num_qubits == 6
        assert record.metadata["test"] is True


class TestMemoryConfig:
    """Tests for MemoryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryConfig()

        assert config.max_records == 1000
        assert config.max_successful == 100
        assert config.max_failed == 200
        assert config.auto_save is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = MemoryConfig(
            max_records=500,
            max_successful=50,
            persistence_path="/tmp/test.json",
        )

        assert config.max_records == 500
        assert config.max_successful == 50
        assert config.persistence_path == "/tmp/test.json"


class TestAgentMemory:
    """Tests for AgentMemory class."""

    def test_add_record(self):
        """Test adding a record."""
        memory = AgentMemory()

        record = memory.add_record(
            code="def create_ansatz(): pass",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=True,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
            energy=-2.5,
            energy_error=0.01,
        )

        assert record is not None
        assert record.circuit_id is not None
        assert len(memory) == 1

    def test_successful_vs_failed(self):
        """Test separation of successful and failed records."""
        memory = AgentMemory()

        # Add successful
        memory.add_record(
            code="success",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=True,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
        )

        # Add failed (invalid)
        memory.add_record(
            code="failed",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=False,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
        )

        # Add failed (not trainable)
        memory.add_record(
            code="failed2",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=True,
            is_trainable=False,
            depth=6,
            param_count=8,
            gate_count=14,
        )

        successful = memory.get_successful()
        failed = memory.get_failed()

        assert len(successful) == 1
        assert len(failed) == 2

    def test_filter_by_hamiltonian(self):
        """Test filtering by Hamiltonian type."""
        memory = AgentMemory()

        memory.add_record(
            code="xy1",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=True,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
        )

        memory.add_record(
            code="heisenberg1",
            hamiltonian_type="heisenberg",
            num_qubits=4,
            is_valid=True,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
        )

        xy_records = memory.get_successful(hamiltonian_type="xy_chain")
        heisenberg_records = memory.get_successful(hamiltonian_type="heisenberg")

        assert len(xy_records) == 1
        assert len(heisenberg_records) == 1

    def test_filter_by_qubits(self):
        """Test filtering by qubit count."""
        memory = AgentMemory()

        memory.add_record(
            code="4qubit",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=True,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
        )

        memory.add_record(
            code="6qubit",
            hamiltonian_type="xy_chain",
            num_qubits=6,
            is_valid=True,
            is_trainable=True,
            depth=8,
            param_count=12,
            gate_count=20,
        )

        records_4 = memory.get_successful(num_qubits=4)
        records_6 = memory.get_successful(num_qubits=6)

        assert len(records_4) == 1
        assert len(records_6) == 1

    def test_get_best_circuit(self):
        """Test getting the best circuit."""
        memory = AgentMemory()

        # Add circuits with different energy errors
        memory.add_record(
            code="worse",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=True,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
            energy_error=0.1,
        )

        memory.add_record(
            code="better",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=True,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
            energy_error=0.01,
        )

        best = memory.get_best_circuit("xy_chain", 4)

        assert best is not None
        assert best.code == "better"
        assert best.energy_error == 0.01

    def test_get_best_circuit_none(self):
        """Test getting best when none exist."""
        memory = AgentMemory()

        best = memory.get_best_circuit("xy_chain", 4)
        assert best is None

    def test_get_context_for_proposal(self):
        """Test context generation for new proposals."""
        memory = AgentMemory()

        # Add successful
        memory.add_record(
            code="def create_ansatz(): pass",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=True,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
            energy_error=0.01,
        )

        # Add failed
        memory.add_record(
            code="failed",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=False,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
            feedback="Syntax error",
        )

        context = memory.get_context_for_proposal("xy_chain", 4)

        assert "SUCCESSFUL" in context
        assert "FAILED" in context
        assert "def create_ansatz" in context
        assert "Syntax error" in context

    def test_statistics(self):
        """Test statistics computation."""
        memory = AgentMemory()

        for i in range(5):
            memory.add_record(
                code=f"success_{i}",
                hamiltonian_type="xy_chain",
                num_qubits=4,
                is_valid=True,
                is_trainable=True,
                depth=6 + i,
                param_count=8,
                gate_count=14,
                energy_error=0.01 * (i + 1),
            )

        for i in range(3):
            memory.add_record(
                code=f"failed_{i}",
                hamiltonian_type="xy_chain",
                num_qubits=4,
                is_valid=False,
                is_trainable=True,
                depth=6,
                param_count=8,
                gate_count=14,
            )

        stats = memory.get_statistics()

        assert stats["total_records"] == 8
        assert stats["successful_count"] == 5
        assert stats["failed_count"] == 3
        assert stats["success_rate"] == 5 / 8
        assert stats["best_energy_error"] == 0.01
        assert stats["avg_depth"] == 8  # (6+7+8+9+10) / 5

    def test_failure_patterns(self):
        """Test failure pattern analysis."""
        memory = AgentMemory()

        # Add failures with different feedback
        memory.add_record(
            code="f1",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=False,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
            feedback="Barren plateau detected",
        )

        memory.add_record(
            code="f2",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=False,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
            feedback="Gradient variance too low",
        )

        memory.add_record(
            code="f3",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=False,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
            feedback="Syntax error in code",
        )

        patterns = memory.get_failure_patterns()

        assert len(patterns) > 0
        # Barren plateau should be counted twice
        bp_pattern = next((p for p in patterns if p["pattern"] == "barren_plateau"), None)
        assert bp_pattern is not None
        assert bp_pattern["count"] == 2

    def test_persistence(self):
        """Test saving and loading memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.json")
            config = MemoryConfig(persistence_path=path, auto_save=False)

            # Create and populate memory
            memory = AgentMemory(config)
            memory.add_record(
                code="test_code",
                hamiltonian_type="xy_chain",
                num_qubits=4,
                is_valid=True,
                is_trainable=True,
                depth=6,
                param_count=8,
                gate_count=14,
                energy_error=0.01,
            )
            memory.save()

            # Create new memory from saved file
            memory2 = AgentMemory(config)

            assert len(memory2) > 0
            records = memory2.get_successful()
            assert len(records) == 1
            assert records[0].code == "test_code"

    def test_clear(self):
        """Test clearing memory."""
        memory = AgentMemory()

        memory.add_record(
            code="test",
            hamiltonian_type="xy_chain",
            num_qubits=4,
            is_valid=True,
            is_trainable=True,
            depth=6,
            param_count=8,
            gate_count=14,
        )

        assert len(memory) == 1

        memory.clear()

        assert len(memory) == 0
        assert len(memory.get_successful()) == 0
        assert len(memory.get_failed()) == 0

    def test_max_records_trimming(self):
        """Test that records are trimmed when max is exceeded."""
        config = MemoryConfig(max_successful=3, max_failed=2)
        memory = AgentMemory(config)

        # Add more successful than max
        for i in range(5):
            memory.add_record(
                code=f"success_{i}",
                hamiltonian_type="xy_chain",
                num_qubits=4,
                is_valid=True,
                is_trainable=True,
                depth=6,
                param_count=8,
                gate_count=14,
                energy_error=0.1 - i * 0.01,  # Better as i increases
            )

        # Should keep only best 3
        successful = memory.get_successful()
        assert len(successful) <= 3

    def test_iterator(self):
        """Test iterating over records."""
        memory = AgentMemory()

        for i in range(3):
            memory.add_record(
                code=f"code_{i}",
                hamiltonian_type="xy_chain",
                num_qubits=4,
                is_valid=True,
                is_trainable=True,
                depth=6,
                param_count=8,
                gate_count=14,
            )

        codes = [r.code for r in memory]
        assert len(codes) == 3
        assert "code_0" in codes


class TestCreateFeedback:
    """Tests for create_feedback function."""

    def test_invalid_circuit_feedback(self):
        """Test feedback for invalid circuit."""
        feedback = create_feedback(
            is_valid=False,
            is_trainable=True,
            verification_errors=["Syntax error on line 5", "Missing import"],
        )

        assert "VERIFICATION FAILED" in feedback
        assert "Syntax error" in feedback
        assert "Missing import" in feedback

    def test_not_trainable_feedback(self):
        """Test feedback for non-trainable circuit."""
        feedback = create_feedback(
            is_valid=True,
            is_trainable=False,
            bp_details={
                "gradient_variance": 1e-8,
                "risk_factors": ["Deep circuit", "High entanglement"],
            },
        )

        assert "TRAINABILITY ISSUE" in feedback
        assert "1e-08" in feedback or "1.00e-08" in feedback
        assert "reducing circuit depth" in feedback

    def test_successful_meeting_target(self):
        """Test feedback for successful circuit meeting target."""
        feedback = create_feedback(
            is_valid=True,
            is_trainable=True,
            energy_error=0.005,
            target_error=0.01,
        )

        assert "VALID AND TRAINABLE" in feedback
        assert "meets target" in feedback

    def test_successful_not_meeting_target(self):
        """Test feedback for successful circuit not meeting target."""
        feedback = create_feedback(
            is_valid=True,
            is_trainable=True,
            energy_error=0.05,
            target_error=0.01,
        )

        assert "VALID AND TRAINABLE" in feedback
        assert "> target" in feedback
        assert "expressive" in feedback

    def test_high_depth_warning(self):
        """Test feedback warns about high depth."""
        feedback = create_feedback(
            is_valid=True,
            is_trainable=True,
            depth=30,
        )

        assert "high" in feedback.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
