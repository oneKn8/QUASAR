"""
Result analyzer for QuantumMind agent.

Analyzes circuit evaluation results and generates structured feedback
to guide the LLM in improving circuit designs.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit import QuantumCircuit


@dataclass
class AnalysisResult:
    """Complete analysis of a circuit evaluation."""

    # Overall assessment
    is_successful: bool
    score: float  # 0-100 composite score

    # Component scores
    validity_score: float
    trainability_score: float
    performance_score: float
    efficiency_score: float

    # Detailed feedback
    summary: str
    improvements: list[str]
    strengths: list[str]
    weaknesses: list[str]

    # Raw data
    details: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Comparison between two circuits."""

    better_circuit: str  # "new" or "baseline"
    improvement_percentage: float
    comparison_details: dict


class ResultAnalyzer:
    """
    Analyzes VQE and verification results to generate feedback.

    Produces structured analysis that helps the LLM understand
    what worked, what didn't, and how to improve.
    """

    def __init__(
        self,
        target_energy_error: float = 0.01,
        max_acceptable_depth: int = 30,
        min_gradient_variance: float = 1e-6,
    ):
        """
        Initialize the analyzer.

        Args:
            target_energy_error: Target energy error (Hartree)
            max_acceptable_depth: Maximum acceptable circuit depth
            min_gradient_variance: Minimum gradient variance for trainability
        """
        self.target_energy_error = target_energy_error
        self.max_acceptable_depth = max_acceptable_depth
        self.min_gradient_variance = min_gradient_variance

        self._analysis_count = 0

    def analyze(
        self,
        circuit: QuantumCircuit | None,
        verification_result: Any | None,
        vqe_result: Any | None = None,
        bp_result: Any | None = None,
        exact_energy: float | None = None,
    ) -> AnalysisResult:
        """
        Perform complete analysis of a circuit evaluation.

        Args:
            circuit: The quantum circuit (if available)
            verification_result: Result from circuit verifier
            vqe_result: Result from VQE execution (optional)
            bp_result: Result from barren plateau detection (optional)
            exact_energy: Exact ground state energy (optional)

        Returns:
            Complete analysis result
        """
        self._analysis_count += 1

        # Compute component scores
        validity_score = self._compute_validity_score(verification_result)
        trainability_score = self._compute_trainability_score(bp_result)
        performance_score = self._compute_performance_score(
            vqe_result, exact_energy
        )
        efficiency_score = self._compute_efficiency_score(circuit, verification_result)

        # Compute composite score
        if validity_score == 0:
            # Invalid circuits get zero overall
            composite_score = 0
        elif trainability_score == 0:
            # Non-trainable circuits get low score
            composite_score = validity_score * 0.2
        else:
            # Weighted average
            composite_score = (
                validity_score * 0.2
                + trainability_score * 0.3
                + performance_score * 0.35
                + efficiency_score * 0.15
            )

        # Determine success
        is_successful = (
            validity_score >= 80
            and trainability_score >= 50
            and performance_score >= 70
        )

        # Generate feedback
        summary = self._generate_summary(
            validity_score,
            trainability_score,
            performance_score,
            efficiency_score,
            verification_result,
            vqe_result,
        )

        improvements = self._generate_improvements(
            validity_score,
            trainability_score,
            performance_score,
            efficiency_score,
            verification_result,
            vqe_result,
            bp_result,
        )

        strengths = self._identify_strengths(
            validity_score,
            trainability_score,
            performance_score,
            efficiency_score,
            circuit,
        )

        weaknesses = self._identify_weaknesses(
            validity_score,
            trainability_score,
            performance_score,
            efficiency_score,
            verification_result,
            bp_result,
        )

        return AnalysisResult(
            is_successful=is_successful,
            score=composite_score,
            validity_score=validity_score,
            trainability_score=trainability_score,
            performance_score=performance_score,
            efficiency_score=efficiency_score,
            summary=summary,
            improvements=improvements,
            strengths=strengths,
            weaknesses=weaknesses,
            details={
                "target_energy_error": self.target_energy_error,
                "max_acceptable_depth": self.max_acceptable_depth,
                "circuit_depth": circuit.depth() if circuit else None,
                "param_count": len(circuit.parameters) if circuit else None,
            },
        )

    def _compute_validity_score(self, verification_result: Any | None) -> float:
        """Compute validity score (0-100)."""
        if verification_result is None:
            return 0

        if not verification_result.is_valid:
            # Partial credit for passing some checks
            checks = [
                verification_result.syntax_ok,
                verification_result.function_ok,
                verification_result.depth_ok,
                verification_result.gates_ok,
                verification_result.params_ok,
                verification_result.qubit_count_ok,
            ]
            return sum(checks) / len(checks) * 40  # Max 40 for invalid

        return 100

    def _compute_trainability_score(self, bp_result: Any | None) -> float:
        """Compute trainability score (0-100)."""
        if bp_result is None:
            return 50  # Unknown, assume moderate

        if hasattr(bp_result, "is_trainable"):
            if not bp_result.is_trainable:
                return 0

        if hasattr(bp_result, "gradient_variance"):
            variance = bp_result.gradient_variance
            if variance < self.min_gradient_variance:
                return 0
            elif variance < 1e-4:
                return 30 + (variance / 1e-4) * 40  # 30-70
            elif variance < 1e-2:
                return 70 + (variance / 1e-2) * 20  # 70-90
            else:
                return 90 + min(10, variance * 10)  # 90-100

        return 70  # Default if trainable but no variance info

    def _compute_performance_score(
        self,
        vqe_result: Any | None,
        exact_energy: float | None,
    ) -> float:
        """Compute performance score based on energy error (0-100)."""
        if vqe_result is None or exact_energy is None:
            return 50  # Unknown

        if hasattr(vqe_result, "energy"):
            error = abs(vqe_result.energy - exact_energy)
            relative_error = error / abs(exact_energy) if exact_energy != 0 else error

            if relative_error <= 0.001:  # < 0.1%
                return 100
            elif relative_error <= 0.01:  # < 1%
                return 90 + (0.01 - relative_error) / 0.009 * 10
            elif relative_error <= 0.05:  # < 5%
                return 70 + (0.05 - relative_error) / 0.04 * 20
            elif relative_error <= 0.1:  # < 10%
                return 50 + (0.1 - relative_error) / 0.05 * 20
            else:
                return max(0, 50 - relative_error * 100)

        return 50

    def _compute_efficiency_score(
        self,
        circuit: QuantumCircuit | None,
        verification_result: Any | None,
    ) -> float:
        """Compute efficiency score based on circuit resources (0-100)."""
        if circuit is None and verification_result is None:
            return 50

        depth = None
        param_count = None
        gate_count = None

        if circuit is not None:
            depth = circuit.depth()
            param_count = len(circuit.parameters)
            gate_count = sum(circuit.count_ops().values())
        elif verification_result is not None and hasattr(verification_result, "details"):
            depth = verification_result.details.get("depth")
            param_count = verification_result.details.get("num_params")
            gate_count = verification_result.details.get("gate_count")

        score = 100

        # Penalize deep circuits
        if depth is not None:
            if depth > self.max_acceptable_depth:
                score -= 30
            elif depth > 20:
                score -= 15
            elif depth > 10:
                score -= 5

        # Penalize too many parameters
        if param_count is not None:
            num_qubits = circuit.num_qubits if circuit else 4
            params_per_qubit = param_count / num_qubits
            if params_per_qubit > 10:
                score -= 20
            elif params_per_qubit > 6:
                score -= 10

        return max(0, score)

    def _generate_summary(
        self,
        validity_score: float,
        trainability_score: float,
        performance_score: float,
        efficiency_score: float,
        verification_result: Any | None,
        vqe_result: Any | None,
    ) -> str:
        """Generate human-readable summary."""
        parts = []

        if validity_score < 80:
            if verification_result and hasattr(verification_result, "errors"):
                parts.append(f"Circuit invalid: {verification_result.errors[0]}")
            else:
                parts.append("Circuit failed verification.")
        elif trainability_score < 50:
            parts.append("Circuit valid but has trainability issues (barren plateau risk).")
        elif performance_score < 70:
            parts.append("Circuit valid and trainable but energy accuracy needs improvement.")
        else:
            parts.append("Circuit performs well.")

        # Add energy info if available
        if vqe_result and hasattr(vqe_result, "energy"):
            parts.append(f"Achieved energy: {vqe_result.energy:.6f}")

        # Add score summary
        parts.append(
            f"Scores: validity={validity_score:.0f}, "
            f"trainability={trainability_score:.0f}, "
            f"performance={performance_score:.0f}, "
            f"efficiency={efficiency_score:.0f}"
        )

        return " ".join(parts)

    def _generate_improvements(
        self,
        validity_score: float,
        trainability_score: float,
        performance_score: float,
        efficiency_score: float,
        verification_result: Any | None,
        vqe_result: Any | None,
        bp_result: Any | None,
    ) -> list[str]:
        """Generate specific improvement suggestions."""
        improvements = []

        if validity_score < 80:
            if verification_result and hasattr(verification_result, "errors"):
                for error in verification_result.errors[:2]:
                    improvements.append(f"Fix: {error}")

        if trainability_score < 50:
            improvements.append("Reduce circuit depth to avoid barren plateaus")
            improvements.append("Use local rotations instead of global entanglement")
            if bp_result and hasattr(bp_result, "risk_factors"):
                for factor in bp_result.risk_factors[:2]:
                    improvements.append(f"Address: {factor}")

        if performance_score < 70 and validity_score >= 80:
            improvements.append("Add more parameters for expressivity")
            improvements.append("Try different entanglement pattern")
            improvements.append("Consider problem-specific structure")

        if efficiency_score < 70 and validity_score >= 80:
            improvements.append("Reduce circuit depth")
            improvements.append("Minimize two-qubit gates")

        return improvements

    def _identify_strengths(
        self,
        validity_score: float,
        trainability_score: float,
        performance_score: float,
        efficiency_score: float,
        circuit: QuantumCircuit | None,
    ) -> list[str]:
        """Identify circuit strengths."""
        strengths = []

        if validity_score >= 90:
            strengths.append("Clean, valid circuit code")

        if trainability_score >= 80:
            strengths.append("Good trainability (healthy gradients)")

        if performance_score >= 90:
            strengths.append("Excellent energy accuracy")

        if efficiency_score >= 90:
            strengths.append("Efficient circuit (shallow depth)")

        if circuit is not None:
            if circuit.depth() <= 10:
                strengths.append(f"Shallow depth ({circuit.depth()})")
            if len(circuit.parameters) <= 10:
                strengths.append(f"Few parameters ({len(circuit.parameters)})")

        return strengths

    def _identify_weaknesses(
        self,
        validity_score: float,
        trainability_score: float,
        performance_score: float,
        efficiency_score: float,
        verification_result: Any | None,
        bp_result: Any | None,
    ) -> list[str]:
        """Identify circuit weaknesses."""
        weaknesses = []

        if validity_score < 80:
            weaknesses.append("Code validity issues")

        if trainability_score < 50:
            weaknesses.append("Barren plateau risk")
            if bp_result and hasattr(bp_result, "gradient_variance"):
                weaknesses.append(
                    f"Low gradient variance ({bp_result.gradient_variance:.2e})"
                )

        if performance_score < 70:
            weaknesses.append("Insufficient energy accuracy")

        if efficiency_score < 70:
            weaknesses.append("Inefficient (too deep or too many parameters)")

        return weaknesses

    def compare_circuits(
        self,
        new_result: AnalysisResult,
        baseline_result: AnalysisResult,
    ) -> ComparisonResult:
        """
        Compare a new circuit against a baseline.

        Args:
            new_result: Analysis of new circuit
            baseline_result: Analysis of baseline circuit

        Returns:
            Comparison result
        """
        new_score = new_result.score
        baseline_score = baseline_result.score

        if baseline_score > 0:
            improvement = (new_score - baseline_score) / baseline_score * 100
        else:
            improvement = 100 if new_score > 0 else 0

        better = "new" if new_score > baseline_score else "baseline"

        return ComparisonResult(
            better_circuit=better,
            improvement_percentage=improvement,
            comparison_details={
                "new_score": new_score,
                "baseline_score": baseline_score,
                "validity_diff": new_result.validity_score - baseline_result.validity_score,
                "trainability_diff": new_result.trainability_score
                - baseline_result.trainability_score,
                "performance_diff": new_result.performance_score
                - baseline_result.performance_score,
                "efficiency_diff": new_result.efficiency_score
                - baseline_result.efficiency_score,
            },
        )

    def format_feedback_for_llm(self, result: AnalysisResult) -> str:
        """
        Format analysis result as LLM-friendly feedback.

        Args:
            result: Analysis result

        Returns:
            Formatted feedback string
        """
        lines = [
            f"EVALUATION RESULT: {'SUCCESS' if result.is_successful else 'NEEDS IMPROVEMENT'}",
            f"Overall Score: {result.score:.1f}/100",
            "",
            "SUMMARY:",
            result.summary,
            "",
        ]

        if result.strengths:
            lines.append("STRENGTHS:")
            for s in result.strengths:
                lines.append(f"  + {s}")
            lines.append("")

        if result.weaknesses:
            lines.append("WEAKNESSES:")
            for w in result.weaknesses:
                lines.append(f"  - {w}")
            lines.append("")

        if result.improvements:
            lines.append("SUGGESTED IMPROVEMENTS:")
            for i, imp in enumerate(result.improvements, 1):
                lines.append(f"  {i}. {imp}")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get analyzer statistics."""
        return {"total_analyses": self._analysis_count}


def quick_analyze(
    is_valid: bool,
    is_trainable: bool,
    energy_error: float | None = None,
    depth: int | None = None,
) -> dict:
    """
    Quick analysis without full result objects.

    Args:
        is_valid: Whether circuit is valid
        is_trainable: Whether circuit is trainable
        energy_error: Energy error (optional)
        depth: Circuit depth (optional)

    Returns:
        Quick analysis dict
    """
    if not is_valid:
        return {
            "status": "invalid",
            "feedback": "Circuit failed verification. Check syntax and constraints.",
            "score": 0,
        }

    if not is_trainable:
        return {
            "status": "not_trainable",
            "feedback": "Circuit has barren plateau issues. Reduce depth.",
            "score": 20,
        }

    score = 70
    feedback_parts = ["Circuit valid and trainable."]

    if energy_error is not None:
        if energy_error < 0.01:
            score += 25
            feedback_parts.append(f"Excellent accuracy (error={energy_error:.4f}).")
        elif energy_error < 0.05:
            score += 15
            feedback_parts.append(f"Good accuracy (error={energy_error:.4f}).")
        else:
            feedback_parts.append(f"Accuracy needs work (error={energy_error:.4f}).")

    if depth is not None and depth <= 10:
        score += 5
        feedback_parts.append("Efficient depth.")

    return {
        "status": "success" if score >= 80 else "needs_improvement",
        "feedback": " ".join(feedback_parts),
        "score": min(100, score),
    }
