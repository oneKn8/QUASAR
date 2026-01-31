"""
Tests for the LLM proposer module.
"""

import pytest

from src.agent.proposer import (
    CircuitProposal,
    MockProposer,
    ProposerConfig,
    build_circuit_prompt,
    extract_code_from_response,
    extract_reasoning_from_response,
)
from src.quantum.verifier import verify_circuit_code


class TestProposerConfig:
    """Tests for ProposerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProposerConfig()

        assert config.model_name == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert config.max_new_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.use_local is True
        assert config.load_in_4bit is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProposerConfig(
            model_name="custom/model",
            max_new_tokens=512,
            temperature=0.5,
            use_local=False,
            api_key="test-key",
        )

        assert config.model_name == "custom/model"
        assert config.max_new_tokens == 512
        assert config.temperature == 0.5
        assert config.use_local is False
        assert config.api_key == "test-key"


class TestBuildCircuitPrompt:
    """Tests for build_circuit_prompt function."""

    def test_basic_prompt(self):
        """Test basic prompt generation."""
        prompt = build_circuit_prompt(
            goal_description="Find ground state energy",
            num_qubits=4,
            hamiltonian_type="xy_chain",
        )

        assert "Find ground state energy" in prompt
        assert "4" in prompt
        assert "xy_chain" in prompt

    def test_prompt_with_constraints(self):
        """Test prompt with constraints."""
        prompt = build_circuit_prompt(
            goal_description="Find ground state",
            num_qubits=4,
            hamiltonian_type="xy_chain",
            constraints={
                "max_depth": 10,
                "max_params": 20,
                "allowed_gates": ["ry", "rz", "cx"],
            },
        )

        assert "CONSTRAINTS" in prompt
        assert "10" in prompt
        assert "20" in prompt
        assert "ry" in prompt

    def test_prompt_with_feedback(self):
        """Test prompt with feedback from previous attempt."""
        prompt = build_circuit_prompt(
            goal_description="Find ground state",
            num_qubits=4,
            hamiltonian_type="xy_chain",
            feedback="Circuit was too deep, causing barren plateaus",
        )

        assert "FEEDBACK" in prompt
        assert "barren plateaus" in prompt

    def test_prompt_includes_instructions(self):
        """Test that prompt includes key instructions."""
        prompt = build_circuit_prompt(
            goal_description="Test",
            num_qubits=2,
            hamiltonian_type="test",
        )

        assert "REASONING" in prompt
        assert "CODE" in prompt


class TestExtractCodeFromResponse:
    """Tests for extract_code_from_response function."""

    def test_extract_from_code_block(self):
        """Extract code from markdown code block."""
        response = """Here's the circuit:

```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    return qc
```

This should work well.
"""
        code = extract_code_from_response(response)

        assert code is not None
        assert "def create_ansatz" in code
        assert "QuantumCircuit" in code

    def test_extract_from_generic_block(self):
        """Extract code from generic code block."""
        response = """
```
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    return qc
```
"""
        code = extract_code_from_response(response)

        assert code is not None
        assert "def create_ansatz" in code

    def test_extract_without_code_block(self):
        """Extract code when no code block present."""
        response = """
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    p = Parameter('theta_0')
    qc.ry(p, 0)
    return qc
"""
        code = extract_code_from_response(response)

        assert code is not None
        assert "def create_ansatz" in code

    def test_no_code_returns_none(self):
        """Return None when no valid code found."""
        response = "This is just text without any code."
        code = extract_code_from_response(response)

        assert code is None

    def test_adds_missing_imports(self):
        """Should add imports if missing."""
        response = """
def create_ansatz(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    p = Parameter('theta_0')
    qc.ry(p, 0)
    return qc
"""
        code = extract_code_from_response(response)

        # Should have added imports
        assert code is not None
        assert "from qiskit import QuantumCircuit" in code


class TestExtractReasoningFromResponse:
    """Tests for extract_reasoning_from_response function."""

    def test_extract_reasoning_section(self):
        """Extract reasoning from labeled section."""
        response = """REASONING: The XY model has U(1) symmetry, so we use RY rotations
which preserve real amplitudes. Linear entanglement matches the chain geometry.

CODE:
```python
def create_ansatz(num_qubits):
    pass
```
"""
        reasoning = extract_reasoning_from_response(response)

        assert "XY model" in reasoning
        assert "U(1) symmetry" in reasoning

    def test_extract_explanation(self):
        """Extract from 'Explanation' label."""
        response = """Explanation: This circuit uses a hardware-efficient approach.

```python
code here
```
"""
        reasoning = extract_reasoning_from_response(response)

        assert "hardware-efficient" in reasoning

    def test_fallback_to_first_paragraph(self):
        """Fall back to first paragraph if no label."""
        response = """This is my explanation of the design.

Some more text here.

```python
code
```
"""
        reasoning = extract_reasoning_from_response(response)

        assert "explanation" in reasoning.lower()


class TestMockProposer:
    """Tests for MockProposer class."""

    def test_propose_xy_chain(self):
        """Test proposal for XY chain."""
        proposer = MockProposer()
        proposal = proposer.propose(
            goal_description="Find XY chain ground state",
            num_qubits=4,
            hamiltonian_type="xy_chain",
        )

        assert isinstance(proposal, CircuitProposal)
        assert proposal.code is not None
        assert "def create_ansatz" in proposal.code
        assert proposal.metadata["is_mock"] is True

    def test_propose_heisenberg(self):
        """Test proposal for Heisenberg model."""
        proposer = MockProposer()
        proposal = proposer.propose(
            goal_description="Find Heisenberg ground state",
            num_qubits=4,
            hamiltonian_type="heisenberg",
        )

        assert "def create_ansatz" in proposal.code
        # Heisenberg template should have both RY and RZ
        assert "ry" in proposal.code.lower()
        assert "rz" in proposal.code.lower()

    def test_propose_tfim(self):
        """Test proposal for transverse-field Ising."""
        proposer = MockProposer()
        proposal = proposer.propose(
            goal_description="Find TFIM ground state",
            num_qubits=4,
            hamiltonian_type="transverse_ising",
        )

        assert "def create_ansatz" in proposal.code
        # TFIM template should have RX for transverse field
        assert "rx" in proposal.code.lower()

    def test_propose_default(self):
        """Test proposal for unknown hamiltonian uses default."""
        proposer = MockProposer()
        proposal = proposer.propose(
            goal_description="Find ground state",
            num_qubits=4,
            hamiltonian_type="unknown_type",
        )

        assert "def create_ansatz" in proposal.code
        assert proposal.code is not None

    def test_stats_tracking(self):
        """Test that stats are tracked."""
        proposer = MockProposer()
        proposer.propose("test", 4, "xy_chain")
        proposer.propose("test", 4, "heisenberg")

        stats = proposer.get_stats()
        assert stats["total_proposals"] == 2

    def test_generated_code_is_valid(self):
        """Test that generated code passes verification."""
        proposer = MockProposer()
        proposal = proposer.propose(
            goal_description="Test",
            num_qubits=4,
            hamiltonian_type="xy_chain",
        )

        # Verify the code is syntactically valid and creates a circuit
        result = verify_circuit_code(
            proposal.code,
            expected_qubits=4,
            require_params=True,
        )

        assert result.is_valid, f"Code validation failed: {result.errors}"
        assert result.circuit is not None
        assert len(result.circuit.parameters) > 0

    def test_all_templates_produce_valid_code(self):
        """Test all templates produce valid circuits."""
        proposer = MockProposer()
        hamiltonian_types = ["xy_chain", "heisenberg", "transverse_ising", "default"]

        for ham_type in hamiltonian_types:
            proposal = proposer.propose(
                goal_description=f"Test {ham_type}",
                num_qubits=4,
                hamiltonian_type=ham_type,
            )

            result = verify_circuit_code(
                proposal.code,
                expected_qubits=4,
                require_params=True,
            )

            assert result.is_valid, f"{ham_type} template invalid: {result.errors}"


class TestCircuitProposal:
    """Tests for CircuitProposal dataclass."""

    def test_basic_proposal(self):
        """Test basic proposal creation."""
        proposal = CircuitProposal(
            code="def create_ansatz(): pass",
            reasoning="Test reasoning",
            raw_response="Full response",
        )

        assert proposal.code == "def create_ansatz(): pass"
        assert proposal.reasoning == "Test reasoning"
        assert proposal.raw_response == "Full response"
        assert proposal.metadata == {}

    def test_proposal_with_metadata(self):
        """Test proposal with metadata."""
        proposal = CircuitProposal(
            code="code",
            reasoning="reasoning",
            raw_response="response",
            metadata={"num_qubits": 4, "custom_field": "value"},
        )

        assert proposal.metadata["num_qubits"] == 4
        assert proposal.metadata["custom_field"] == "value"


class TestIntegration:
    """Integration tests combining proposer with verifier."""

    def test_mock_proposer_to_verification(self):
        """Test full flow from mock proposal to verification."""
        proposer = MockProposer()

        # Generate proposal
        proposal = proposer.propose(
            goal_description="Find ground state of 4-qubit XY chain",
            num_qubits=4,
            hamiltonian_type="xy_chain",
        )

        # Verify the proposal
        result = verify_circuit_code(
            proposal.code,
            expected_qubits=4,
            require_params=True,
            max_depth=50,
        )

        assert result.is_valid
        assert result.circuit is not None
        assert result.details["num_qubits"] == 4

    def test_multiple_proposals_vary(self):
        """Test that different hamiltonian types produce different circuits."""
        proposer = MockProposer()

        xy_proposal = proposer.propose("test", 4, "xy_chain")
        heisenberg_proposal = proposer.propose("test", 4, "heisenberg")

        # Templates should be different
        assert xy_proposal.code != heisenberg_proposal.code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
