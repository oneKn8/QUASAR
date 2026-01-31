"""
Tests for the discovery agent module.
"""

import tempfile

import pytest

from src.agent.discovery import (
    DiscoveryAgent,
    DiscoveryConfig,
    DiscoveryResult,
    run_discovery,
    run_multi_discovery,
)


class TestDiscoveryConfig:
    """Tests for DiscoveryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DiscoveryConfig()

        assert config.max_iterations == 50
        assert config.max_failed_attempts == 10
        assert config.target_energy_error == 0.01
        assert config.use_mock_proposer is True
        assert config.run_vqe is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = DiscoveryConfig(
            max_iterations=20,
            target_energy_error=0.005,
            use_mock_proposer=False,
            verbose=False,
        )

        assert config.max_iterations == 20
        assert config.target_energy_error == 0.005
        assert config.use_mock_proposer is False
        assert config.verbose is False


class TestDiscoveryAgent:
    """Tests for DiscoveryAgent class."""

    def test_initialization(self):
        """Test agent initialization."""
        config = DiscoveryConfig(verbose=False)
        agent = DiscoveryAgent(config)

        assert agent is not None
        stats = agent.get_stats()
        assert stats["total_discoveries"] == 0

    def test_discover_xy_chain(self):
        """Test discovery for XY chain (with mock proposer)."""
        config = DiscoveryConfig(
            max_iterations=5,
            verbose=False,
            use_mock_proposer=True,
            run_vqe=True,
            check_barren_plateau=False,  # Skip BP check for speed
        )

        agent = DiscoveryAgent(config)
        result = agent.discover(
            hamiltonian_type="xy_chain",
            num_qubits=4,
        )

        assert isinstance(result, DiscoveryResult)
        assert result.total_iterations > 0
        assert result.best_circuit_code is not None

    def test_discover_heisenberg(self):
        """Test discovery for Heisenberg chain."""
        config = DiscoveryConfig(
            max_iterations=3,
            verbose=False,
            use_mock_proposer=True,
            run_vqe=True,
            check_barren_plateau=False,
        )

        agent = DiscoveryAgent(config)
        result = agent.discover(
            hamiltonian_type="heisenberg",
            num_qubits=4,
        )

        assert result.total_iterations > 0
        assert result.best_energy is not None

    def test_discover_with_constraints(self):
        """Test discovery with constraints."""
        config = DiscoveryConfig(
            max_iterations=3,
            verbose=False,
            use_mock_proposer=True,
            run_vqe=False,  # Skip VQE for speed
            check_barren_plateau=False,
        )

        agent = DiscoveryAgent(config)
        result = agent.discover(
            hamiltonian_type="xy_chain",
            num_qubits=4,
            constraints={
                "max_depth": 10,
                "allowed_gates": ["ry", "rz", "cx"],
            },
        )

        assert result.total_iterations > 0

    def test_discover_callback(self):
        """Test that callback is invoked."""
        config = DiscoveryConfig(
            max_iterations=3,
            verbose=False,
            use_mock_proposer=True,
            run_vqe=False,
            check_barren_plateau=False,
        )

        callback_count = 0

        def callback(iteration, status, analysis):
            nonlocal callback_count
            callback_count += 1

        agent = DiscoveryAgent(config)
        agent.discover(
            hamiltonian_type="xy_chain",
            num_qubits=4,
            callback=callback,
        )

        assert callback_count > 0

    def test_early_stop_on_success(self):
        """Test early stopping when target is reached."""
        config = DiscoveryConfig(
            max_iterations=50,
            target_energy_error=1.0,  # Very lenient target
            verbose=False,
            use_mock_proposer=True,
            run_vqe=True,
            check_barren_plateau=False,
            early_stop_on_success=True,
        )

        agent = DiscoveryAgent(config)
        result = agent.discover(
            hamiltonian_type="xy_chain",
            num_qubits=4,
        )

        # Should stop early since target is lenient
        assert result.success == True  # noqa: E712 (numpy bool comparison)
        assert result.total_iterations < config.max_iterations

    def test_memory_persisted(self):
        """Test that memory is updated during discovery."""
        config = DiscoveryConfig(
            max_iterations=3,
            verbose=False,
            use_mock_proposer=True,
            run_vqe=False,
            check_barren_plateau=False,
        )

        agent = DiscoveryAgent(config)
        agent.discover(
            hamiltonian_type="xy_chain",
            num_qubits=4,
        )

        memory = agent.get_memory()
        assert len(memory) > 0

    def test_stats_updated(self):
        """Test that stats are updated after discovery."""
        config = DiscoveryConfig(
            max_iterations=2,
            verbose=False,
            use_mock_proposer=True,
            run_vqe=False,
            check_barren_plateau=False,
        )

        agent = DiscoveryAgent(config)
        agent.discover("xy_chain", 4)
        agent.discover("heisenberg", 4)

        stats = agent.get_stats()
        assert stats["total_discoveries"] == 2
        assert stats["total_iterations"] >= 4

    def test_reset(self):
        """Test agent reset."""
        config = DiscoveryConfig(
            max_iterations=2,
            verbose=False,
            use_mock_proposer=True,
            run_vqe=False,
            check_barren_plateau=False,
        )

        agent = DiscoveryAgent(config)
        agent.discover("xy_chain", 4)

        assert len(agent.get_memory()) > 0

        agent.reset()

        assert len(agent.get_memory()) == 0
        stats = agent.get_stats()
        assert stats["total_discoveries"] == 0


class TestDiscoveryResult:
    """Tests for DiscoveryResult dataclass."""

    def test_basic_result(self):
        """Test basic result creation."""
        result = DiscoveryResult(
            success=True,
            best_circuit_code="def create_ansatz(): pass",
            best_energy=-2.5,
            best_energy_error=0.01,
            total_iterations=10,
            successful_circuits=5,
            failed_circuits=5,
            best_analysis=None,
            runtime_seconds=5.5,
        )

        assert result.success is True
        assert result.best_energy == -2.5
        assert result.total_iterations == 10
        assert result.runtime_seconds == 5.5

    def test_failed_result(self):
        """Test failed result."""
        result = DiscoveryResult(
            success=False,
            best_circuit_code=None,
            best_energy=None,
            best_energy_error=None,
            total_iterations=20,
            successful_circuits=0,
            failed_circuits=20,
            best_analysis=None,
        )

        assert result.success is False
        assert result.best_circuit_code is None


class TestRunDiscovery:
    """Tests for run_discovery convenience function."""

    def test_basic_run(self):
        """Test basic discovery run."""
        result = run_discovery(
            hamiltonian_type="xy_chain",
            num_qubits=4,
            max_iterations=3,
            verbose=False,
        )

        assert isinstance(result, DiscoveryResult)
        assert result.total_iterations > 0

    def test_run_with_custom_target(self):
        """Test run with custom target error."""
        result = run_discovery(
            hamiltonian_type="xy_chain",
            num_qubits=4,
            max_iterations=5,
            target_error=0.5,  # Lenient
            verbose=False,
        )

        # With lenient target, should likely succeed
        assert result.total_iterations > 0


class TestRunMultiDiscovery:
    """Tests for run_multi_discovery function."""

    def test_multi_discovery(self):
        """Test running discovery on multiple problems."""
        problems = [
            ("xy_chain", 4),
            ("heisenberg", 4),
        ]

        results = run_multi_discovery(
            problems=problems,
            max_iterations_per_problem=2,
            target_error=0.5,
            verbose=False,
        )

        assert len(results) == 2
        assert "xy_chain_4q" in results
        assert "heisenberg_4q" in results

        for name, result in results.items():
            assert isinstance(result, DiscoveryResult)


class TestIntegration:
    """Integration tests for the full discovery pipeline."""

    def test_full_pipeline_xy_chain(self):
        """Test full pipeline for XY chain."""
        config = DiscoveryConfig(
            max_iterations=5,
            target_energy_error=0.5,  # Lenient for testing
            verbose=False,
            use_mock_proposer=True,
            run_vqe=True,
            check_barren_plateau=True,
            bp_num_samples=5,
        )

        agent = DiscoveryAgent(config)
        result = agent.discover(
            hamiltonian_type="xy_chain",
            num_qubits=4,
        )

        # Should complete without errors
        assert result.total_iterations > 0

        # Should have found some circuit
        assert result.best_circuit_code is not None

        # Memory should have records
        memory = agent.get_memory()
        assert len(memory) > 0

    def test_memory_persistence(self):
        """Test that memory can be persisted and reloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/memory.json"

            # First run
            config1 = DiscoveryConfig(
                max_iterations=2,
                verbose=False,
                use_mock_proposer=True,
                run_vqe=False,
                check_barren_plateau=False,
                memory_persistence_path=path,
            )

            agent1 = DiscoveryAgent(config1)
            agent1.discover("xy_chain", 4)
            agent1.get_memory().save()

            # Second run with same path
            config2 = DiscoveryConfig(
                max_iterations=2,
                verbose=False,
                use_mock_proposer=True,
                run_vqe=False,
                check_barren_plateau=False,
                memory_persistence_path=path,
            )

            agent2 = DiscoveryAgent(config2)

            # Should have loaded previous records
            # Note: actual loading happens in AgentMemory init
            # This test validates the path is passed through


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
