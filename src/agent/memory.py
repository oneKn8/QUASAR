"""
Memory system for QuantumMind agent.

Tracks successful and failed circuit proposals, enabling learning from
past attempts and providing context for future proposals.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Iterator


@dataclass
class CircuitRecord:
    """Record of a circuit proposal and its evaluation."""

    circuit_id: str
    code: str
    hamiltonian_type: str
    num_qubits: int
    energy: float | None
    energy_error: float | None
    is_valid: bool
    is_trainable: bool
    depth: int
    param_count: int
    gate_count: int
    reasoning: str
    feedback: str
    timestamp: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CircuitRecord":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MemoryConfig:
    """Configuration for the memory system."""

    max_records: int = 1000
    max_successful: int = 100
    max_failed: int = 200
    persistence_path: str | None = None
    auto_save: bool = True
    save_interval: int = 10  # Save every N records


class AgentMemory:
    """
    Memory system for the quantum circuit discovery agent.

    Tracks successful and failed proposals, computes statistics,
    and provides context for new proposals.
    """

    def __init__(self, config: MemoryConfig | None = None):
        """
        Initialize the memory system.

        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()

        self._successful: list[CircuitRecord] = []
        self._failed: list[CircuitRecord] = []
        self._all_records: list[CircuitRecord] = []

        self._record_counter = 0
        self._unsaved_count = 0

        # Load existing records if persistence path exists
        if self.config.persistence_path and os.path.exists(self.config.persistence_path):
            self._load()

    def _generate_id(self) -> str:
        """Generate a unique record ID."""
        self._record_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"circuit_{timestamp}_{self._record_counter:04d}"

    def add_record(
        self,
        code: str,
        hamiltonian_type: str,
        num_qubits: int,
        is_valid: bool,
        is_trainable: bool,
        depth: int,
        param_count: int,
        gate_count: int,
        reasoning: str = "",
        feedback: str = "",
        energy: float | None = None,
        energy_error: float | None = None,
        metadata: dict | None = None,
    ) -> CircuitRecord:
        """
        Add a new circuit record to memory.

        Args:
            code: Circuit code
            hamiltonian_type: Type of Hamiltonian
            num_qubits: Number of qubits
            is_valid: Whether circuit passed verification
            is_trainable: Whether circuit is trainable (no barren plateau)
            depth: Circuit depth
            param_count: Number of parameters
            gate_count: Total gate count
            reasoning: LLM's reasoning for the design
            feedback: Generated feedback about the circuit
            energy: Achieved energy (if VQE was run)
            energy_error: Energy error vs exact (if available)
            metadata: Additional metadata

        Returns:
            The created CircuitRecord
        """
        record = CircuitRecord(
            circuit_id=self._generate_id(),
            code=code,
            hamiltonian_type=hamiltonian_type,
            num_qubits=num_qubits,
            energy=energy,
            energy_error=energy_error,
            is_valid=is_valid,
            is_trainable=is_trainable,
            depth=depth,
            param_count=param_count,
            gate_count=gate_count,
            reasoning=reasoning,
            feedback=feedback,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )

        # Determine if successful
        is_successful = is_valid and is_trainable

        if is_successful:
            self._successful.append(record)
            # Trim if needed
            if len(self._successful) > self.config.max_successful:
                # Keep best by energy error (lowest first)
                self._successful.sort(
                    key=lambda r: r.energy_error if r.energy_error is not None else float("inf")
                )
                self._successful = self._successful[: self.config.max_successful]
        else:
            self._failed.append(record)
            # Trim if needed (keep most recent)
            if len(self._failed) > self.config.max_failed:
                self._failed = self._failed[-self.config.max_failed :]

        self._all_records.append(record)
        if len(self._all_records) > self.config.max_records:
            self._all_records = self._all_records[-self.config.max_records :]

        # Auto-save if enabled
        self._unsaved_count += 1
        if (
            self.config.auto_save
            and self.config.persistence_path
            and self._unsaved_count >= self.config.save_interval
        ):
            self._save()
            self._unsaved_count = 0

        return record

    def get_successful(
        self,
        hamiltonian_type: str | None = None,
        num_qubits: int | None = None,
        limit: int | None = None,
    ) -> list[CircuitRecord]:
        """
        Get successful circuit records.

        Args:
            hamiltonian_type: Filter by Hamiltonian type
            num_qubits: Filter by qubit count
            limit: Maximum number to return

        Returns:
            List of successful records
        """
        records = self._successful

        if hamiltonian_type:
            records = [r for r in records if r.hamiltonian_type == hamiltonian_type]

        if num_qubits:
            records = [r for r in records if r.num_qubits == num_qubits]

        # Sort by energy error (best first)
        records = sorted(
            records,
            key=lambda r: r.energy_error if r.energy_error is not None else float("inf"),
        )

        if limit:
            records = records[:limit]

        return records

    def get_failed(
        self,
        hamiltonian_type: str | None = None,
        num_qubits: int | None = None,
        limit: int | None = None,
    ) -> list[CircuitRecord]:
        """
        Get failed circuit records.

        Args:
            hamiltonian_type: Filter by Hamiltonian type
            num_qubits: Filter by qubit count
            limit: Maximum number to return (most recent)

        Returns:
            List of failed records
        """
        records = self._failed

        if hamiltonian_type:
            records = [r for r in records if r.hamiltonian_type == hamiltonian_type]

        if num_qubits:
            records = [r for r in records if r.num_qubits == num_qubits]

        # Most recent first
        records = list(reversed(records))

        if limit:
            records = records[:limit]

        return records

    def get_best_circuit(
        self,
        hamiltonian_type: str,
        num_qubits: int,
    ) -> CircuitRecord | None:
        """
        Get the best circuit for a specific problem.

        Args:
            hamiltonian_type: Type of Hamiltonian
            num_qubits: Number of qubits

        Returns:
            Best circuit record or None
        """
        records = self.get_successful(
            hamiltonian_type=hamiltonian_type,
            num_qubits=num_qubits,
            limit=1,
        )
        return records[0] if records else None

    def get_context_for_proposal(
        self,
        hamiltonian_type: str,
        num_qubits: int,
        include_successful: int = 2,
        include_failed: int = 2,
    ) -> str:
        """
        Generate context string for a new proposal.

        Includes examples of successful and failed attempts to help
        the LLM learn from past experience.

        Args:
            hamiltonian_type: Type of Hamiltonian
            num_qubits: Number of qubits
            include_successful: Number of successful examples
            include_failed: Number of failed examples

        Returns:
            Context string for the LLM
        """
        context_parts = []

        # Add successful examples
        successful = self.get_successful(
            hamiltonian_type=hamiltonian_type,
            num_qubits=num_qubits,
            limit=include_successful,
        )

        if successful:
            context_parts.append("SUCCESSFUL CIRCUITS FROM PREVIOUS ATTEMPTS:")
            for i, record in enumerate(successful, 1):
                error_str = (
                    f"error={record.energy_error:.6f}"
                    if record.energy_error is not None
                    else "error=N/A"
                )
                context_parts.append(
                    f"""
Example {i} ({error_str}, depth={record.depth}):
{record.code}
"""
                )

        # Add failed examples with feedback
        failed = self.get_failed(
            hamiltonian_type=hamiltonian_type,
            num_qubits=num_qubits,
            limit=include_failed,
        )

        if failed:
            context_parts.append("\nFAILED CIRCUITS TO AVOID:")
            for i, record in enumerate(failed, 1):
                context_parts.append(
                    f"""
Failed attempt {i}:
Issue: {record.feedback}
"""
                )

        return "\n".join(context_parts)

    def get_statistics(
        self,
        hamiltonian_type: str | None = None,
    ) -> dict:
        """
        Get memory statistics.

        Args:
            hamiltonian_type: Optional filter by Hamiltonian type

        Returns:
            Statistics dictionary
        """
        if hamiltonian_type:
            successful = [r for r in self._successful if r.hamiltonian_type == hamiltonian_type]
            failed = [r for r in self._failed if r.hamiltonian_type == hamiltonian_type]
            all_records = [r for r in self._all_records if r.hamiltonian_type == hamiltonian_type]
        else:
            successful = self._successful
            failed = self._failed
            all_records = self._all_records

        total = len(all_records)
        success_count = len(successful)
        fail_count = len(failed)

        # Energy statistics for successful circuits
        energies = [r.energy_error for r in successful if r.energy_error is not None]

        stats = {
            "total_records": total,
            "successful_count": success_count,
            "failed_count": fail_count,
            "success_rate": success_count / total if total > 0 else 0,
        }

        if energies:
            stats["best_energy_error"] = min(energies)
            stats["avg_energy_error"] = sum(energies) / len(energies)
            stats["worst_energy_error"] = max(energies)

        # Depth statistics for successful circuits
        depths = [r.depth for r in successful]
        if depths:
            stats["avg_depth"] = sum(depths) / len(depths)
            stats["min_depth"] = min(depths)
            stats["max_depth"] = max(depths)

        # Parameter statistics
        params = [r.param_count for r in successful]
        if params:
            stats["avg_params"] = sum(params) / len(params)

        return stats

    def get_failure_patterns(
        self,
        hamiltonian_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Analyze common failure patterns.

        Args:
            hamiltonian_type: Optional filter
            limit: Maximum patterns to return

        Returns:
            List of failure pattern summaries
        """
        failed = self._failed
        if hamiltonian_type:
            failed = [r for r in failed if r.hamiltonian_type == hamiltonian_type]

        # Count feedback categories
        feedback_counts: dict[str, int] = {}
        for record in failed:
            feedback = record.feedback.lower()
            if "barren" in feedback or "gradient" in feedback:
                key = "barren_plateau"
            elif "syntax" in feedback:
                key = "syntax_error"
            elif "depth" in feedback:
                key = "too_deep"
            elif "parameter" in feedback or "param" in feedback:
                key = "parameter_issue"
            elif "gate" in feedback:
                key = "invalid_gate"
            elif "qubit" in feedback:
                key = "qubit_mismatch"
            else:
                key = "other"

            feedback_counts[key] = feedback_counts.get(key, 0) + 1

        # Sort by count
        patterns = [
            {"pattern": k, "count": v, "percentage": v / len(failed) if failed else 0}
            for k, v in sorted(feedback_counts.items(), key=lambda x: -x[1])
        ]

        return patterns[:limit]

    def _save(self):
        """Save memory to disk."""
        if not self.config.persistence_path:
            return

        os.makedirs(os.path.dirname(self.config.persistence_path), exist_ok=True)

        data = {
            "successful": [r.to_dict() for r in self._successful],
            "failed": [r.to_dict() for r in self._failed],
            "all_records": [r.to_dict() for r in self._all_records[-100:]],  # Keep recent
            "counter": self._record_counter,
        }

        with open(self.config.persistence_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load memory from disk."""
        if not self.config.persistence_path or not os.path.exists(
            self.config.persistence_path
        ):
            return

        with open(self.config.persistence_path) as f:
            data = json.load(f)

        self._successful = [CircuitRecord.from_dict(r) for r in data.get("successful", [])]
        self._failed = [CircuitRecord.from_dict(r) for r in data.get("failed", [])]
        self._all_records = [CircuitRecord.from_dict(r) for r in data.get("all_records", [])]
        self._record_counter = data.get("counter", 0)

    def save(self):
        """Explicitly save memory to disk."""
        self._save()

    def clear(self):
        """Clear all memory."""
        self._successful = []
        self._failed = []
        self._all_records = []
        self._record_counter = 0

    def __len__(self) -> int:
        """Get total number of records."""
        return len(self._all_records)

    def __iter__(self) -> Iterator[CircuitRecord]:
        """Iterate over all records."""
        return iter(self._all_records)


def create_feedback(
    is_valid: bool,
    is_trainable: bool,
    verification_errors: list[str] | None = None,
    bp_details: dict | None = None,
    energy_error: float | None = None,
    depth: int | None = None,
    target_error: float = 0.01,
) -> str:
    """
    Generate feedback string for a circuit evaluation.

    Args:
        is_valid: Whether circuit passed verification
        is_trainable: Whether circuit is trainable
        verification_errors: List of verification errors
        bp_details: Barren plateau detection details
        energy_error: Energy error vs exact
        depth: Circuit depth
        target_error: Target energy error

    Returns:
        Feedback string
    """
    feedback_parts = []

    if not is_valid:
        feedback_parts.append("VERIFICATION FAILED:")
        if verification_errors:
            for error in verification_errors[:3]:  # Limit to 3 errors
                feedback_parts.append(f"  - {error}")
        return "\n".join(feedback_parts)

    if not is_trainable:
        feedback_parts.append("TRAINABILITY ISSUE (Barren Plateau Risk):")
        if bp_details:
            if "gradient_variance" in bp_details:
                feedback_parts.append(
                    f"  - Gradient variance: {bp_details['gradient_variance']:.2e} (too low)"
                )
            if "risk_factors" in bp_details:
                for factor in bp_details["risk_factors"][:2]:
                    feedback_parts.append(f"  - {factor}")
        feedback_parts.append("  - Consider reducing circuit depth or entanglement")
        return "\n".join(feedback_parts)

    # Circuit is valid and trainable - provide performance feedback
    feedback_parts.append("CIRCUIT VALID AND TRAINABLE")

    if energy_error is not None:
        if energy_error <= target_error:
            feedback_parts.append(f"  - Energy error {energy_error:.6f} meets target!")
        else:
            feedback_parts.append(
                f"  - Energy error {energy_error:.6f} > target {target_error}"
            )
            feedback_parts.append("  - Consider more expressive ansatz or different structure")

    if depth is not None:
        if depth > 20:
            feedback_parts.append(f"  - Depth {depth} is high, consider optimization")

    return "\n".join(feedback_parts)
