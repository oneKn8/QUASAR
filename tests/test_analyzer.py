"""
Tests for the result analyzer module.
"""

from dataclasses import dataclass
from typing import Any

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from src.agent.analyzer import (
    AnalysisResult,
    ComparisonResult,
    ResultAnalyzer,
    quick_analyze,
)


@dataclass
class MockVerificationResult:
    """Mock verification result for testing."""

    is_valid: bool
    syntax_ok: bool = True
    function_ok: bool = True
    depth_ok: bool = True
    gates_ok: bool = True
    params_ok: bool = True
    qubit_count_ok: bool = True
    errors: list = None
    details: dict = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.details is None:
            self.details = {}


@dataclass
class MockVQEResult:
    """Mock VQE result for testing."""

    energy: float
    optimal_params: list = None
    iterations: int = 100
    converged: bool = True


@dataclass
class MockBPResult:
    """Mock barren plateau result for testing."""

    is_trainable: bool
    gradient_variance: float
    risk_factors: list = None

    def __post_init__(self):
        if self.risk_factors is None:
            self.risk_factors = []


def create_test_circuit(num_qubits: int = 4, depth: int = 2) -> QuantumCircuit:
    """Create a simple test circuit."""
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for i in range(num_qubits):
            p = Parameter(f"theta_{len(qc.parameters)}")
            qc.ry(p, i)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
    return qc


class TestResultAnalyzer:
    """Tests for ResultAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ResultAnalyzer()

        assert analyzer.target_energy_error == 0.01
        assert analyzer.max_acceptable_depth == 30

    def test_custom_initialization(self):
        """Test custom initialization."""
        analyzer = ResultAnalyzer(
            target_energy_error=0.005,
            max_acceptable_depth=20,
            min_gradient_variance=1e-5,
        )

        assert analyzer.target_energy_error == 0.005
        assert analyzer.max_acceptable_depth == 20

    def test_analyze_valid_trainable_circuit(self):
        """Test analysis of a valid, trainable circuit."""
        analyzer = ResultAnalyzer()
        circuit = create_test_circuit(4)

        verification = MockVerificationResult(is_valid=True)
        vqe_result = MockVQEResult(energy=-2.82)
        bp_result = MockBPResult(is_trainable=True, gradient_variance=0.01)
        exact_energy = -2.83

        result = analyzer.analyze(
            circuit=circuit,
            verification_result=verification,
            vqe_result=vqe_result,
            bp_result=bp_result,
            exact_energy=exact_energy,
        )

        assert isinstance(result, AnalysisResult)
        assert result.validity_score == 100
        assert result.trainability_score > 50
        assert result.score > 0

    def test_analyze_invalid_circuit(self):
        """Test analysis of an invalid circuit."""
        analyzer = ResultAnalyzer()

        verification = MockVerificationResult(
            is_valid=False,
            syntax_ok=False,
            errors=["Syntax error on line 5"],
        )

        result = analyzer.analyze(
            circuit=None,
            verification_result=verification,
        )

        assert result.validity_score < 100
        assert result.is_successful is False
        assert "invalid" in result.summary.lower() or "failed" in result.summary.lower()

    def test_analyze_non_trainable_circuit(self):
        """Test analysis of a non-trainable circuit."""
        analyzer = ResultAnalyzer()
        circuit = create_test_circuit(4)

        verification = MockVerificationResult(is_valid=True)
        bp_result = MockBPResult(
            is_trainable=False,
            gradient_variance=1e-10,
            risk_factors=["Deep circuit", "Too much entanglement"],
        )

        result = analyzer.analyze(
            circuit=circuit,
            verification_result=verification,
            bp_result=bp_result,
        )

        assert result.trainability_score == 0
        assert result.is_successful is False
        assert "trainability" in result.summary.lower() or "barren" in result.summary.lower()

    def test_analyze_excellent_performance(self):
        """Test analysis of circuit with excellent performance."""
        analyzer = ResultAnalyzer(target_energy_error=0.01)
        circuit = create_test_circuit(4)

        verification = MockVerificationResult(is_valid=True)
        vqe_result = MockVQEResult(energy=-2.8284)
        bp_result = MockBPResult(is_trainable=True, gradient_variance=0.01)
        exact_energy = -2.8284

        result = analyzer.analyze(
            circuit=circuit,
            verification_result=verification,
            vqe_result=vqe_result,
            bp_result=bp_result,
            exact_energy=exact_energy,
        )

        assert result.performance_score >= 90

    def test_analyze_poor_performance(self):
        """Test analysis of circuit with poor performance."""
        analyzer = ResultAnalyzer()
        circuit = create_test_circuit(4)

        verification = MockVerificationResult(is_valid=True)
        vqe_result = MockVQEResult(energy=-2.0)  # Far from exact
        bp_result = MockBPResult(is_trainable=True, gradient_variance=0.01)
        exact_energy = -2.83

        result = analyzer.analyze(
            circuit=circuit,
            verification_result=verification,
            vqe_result=vqe_result,
            bp_result=bp_result,
            exact_energy=exact_energy,
        )

        assert result.performance_score < 70

    def test_improvements_generated(self):
        """Test that improvements are generated."""
        analyzer = ResultAnalyzer()

        verification = MockVerificationResult(
            is_valid=False,
            errors=["Missing import statement"],
        )

        result = analyzer.analyze(
            circuit=None,
            verification_result=verification,
        )

        assert len(result.improvements) > 0
        assert any("fix" in imp.lower() or "missing" in imp.lower() for imp in result.improvements)

    def test_strengths_identified(self):
        """Test that strengths are identified for good circuits."""
        analyzer = ResultAnalyzer()
        circuit = create_test_circuit(4, depth=1)  # Shallow circuit

        verification = MockVerificationResult(is_valid=True)
        vqe_result = MockVQEResult(energy=-2.828)
        bp_result = MockBPResult(is_trainable=True, gradient_variance=0.05)
        exact_energy = -2.828

        result = analyzer.analyze(
            circuit=circuit,
            verification_result=verification,
            vqe_result=vqe_result,
            bp_result=bp_result,
            exact_energy=exact_energy,
        )

        assert len(result.strengths) > 0

    def test_weaknesses_identified(self):
        """Test that weaknesses are identified for problematic circuits."""
        analyzer = ResultAnalyzer()

        verification = MockVerificationResult(is_valid=True)
        bp_result = MockBPResult(
            is_trainable=False,
            gradient_variance=1e-10,
        )

        result = analyzer.analyze(
            circuit=None,
            verification_result=verification,
            bp_result=bp_result,
        )

        assert len(result.weaknesses) > 0
        assert any("barren" in w.lower() or "gradient" in w.lower() for w in result.weaknesses)

    def test_stats_tracking(self):
        """Test that analysis count is tracked."""
        analyzer = ResultAnalyzer()

        verification = MockVerificationResult(is_valid=True)
        analyzer.analyze(circuit=None, verification_result=verification)
        analyzer.analyze(circuit=None, verification_result=verification)

        stats = analyzer.get_stats()
        assert stats["total_analyses"] == 2


class TestCompareCircuits:
    """Tests for circuit comparison."""

    def test_new_better_than_baseline(self):
        """Test comparison when new circuit is better."""
        analyzer = ResultAnalyzer()

        new_result = AnalysisResult(
            is_successful=True,
            score=85,
            validity_score=100,
            trainability_score=90,
            performance_score=80,
            efficiency_score=90,
            summary="Good circuit",
            improvements=[],
            strengths=["Excellent"],
            weaknesses=[],
        )

        baseline_result = AnalysisResult(
            is_successful=True,
            score=70,
            validity_score=100,
            trainability_score=70,
            performance_score=60,
            efficiency_score=80,
            summary="Baseline",
            improvements=[],
            strengths=[],
            weaknesses=[],
        )

        comparison = analyzer.compare_circuits(new_result, baseline_result)

        assert comparison.better_circuit == "new"
        assert comparison.improvement_percentage > 0

    def test_baseline_better_than_new(self):
        """Test comparison when baseline is better."""
        analyzer = ResultAnalyzer()

        new_result = AnalysisResult(
            is_successful=False,
            score=50,
            validity_score=100,
            trainability_score=40,
            performance_score=50,
            efficiency_score=70,
            summary="Worse",
            improvements=[],
            strengths=[],
            weaknesses=[],
        )

        baseline_result = AnalysisResult(
            is_successful=True,
            score=80,
            validity_score=100,
            trainability_score=80,
            performance_score=75,
            efficiency_score=85,
            summary="Baseline",
            improvements=[],
            strengths=[],
            weaknesses=[],
        )

        comparison = analyzer.compare_circuits(new_result, baseline_result)

        assert comparison.better_circuit == "baseline"
        assert comparison.improvement_percentage < 0

    def test_comparison_details(self):
        """Test that comparison includes detailed diffs."""
        analyzer = ResultAnalyzer()

        new_result = AnalysisResult(
            is_successful=True,
            score=90,
            validity_score=100,
            trainability_score=95,
            performance_score=85,
            efficiency_score=90,
            summary="New",
            improvements=[],
            strengths=[],
            weaknesses=[],
        )

        baseline_result = AnalysisResult(
            is_successful=True,
            score=70,
            validity_score=100,
            trainability_score=70,
            performance_score=60,
            efficiency_score=80,
            summary="Baseline",
            improvements=[],
            strengths=[],
            weaknesses=[],
        )

        comparison = analyzer.compare_circuits(new_result, baseline_result)

        assert "new_score" in comparison.comparison_details
        assert "baseline_score" in comparison.comparison_details
        assert "validity_diff" in comparison.comparison_details
        assert "trainability_diff" in comparison.comparison_details


class TestFormatFeedbackForLLM:
    """Tests for LLM feedback formatting."""

    def test_format_successful_result(self):
        """Test formatting of successful result."""
        analyzer = ResultAnalyzer()

        result = AnalysisResult(
            is_successful=True,
            score=90,
            validity_score=100,
            trainability_score=90,
            performance_score=85,
            efficiency_score=95,
            summary="Excellent circuit with good accuracy",
            improvements=["Minor: could optimize depth further"],
            strengths=["Clean code", "Good trainability"],
            weaknesses=[],
        )

        feedback = analyzer.format_feedback_for_llm(result)

        assert "SUCCESS" in feedback
        assert "90" in feedback
        assert "STRENGTHS" in feedback
        assert "Clean code" in feedback

    def test_format_failed_result(self):
        """Test formatting of failed result."""
        analyzer = ResultAnalyzer()

        result = AnalysisResult(
            is_successful=False,
            score=30,
            validity_score=40,
            trainability_score=0,
            performance_score=50,
            efficiency_score=60,
            summary="Circuit has barren plateau issues",
            improvements=["Reduce depth", "Use local entanglement"],
            strengths=[],
            weaknesses=["Barren plateau risk", "Too deep"],
        )

        feedback = analyzer.format_feedback_for_llm(result)

        assert "NEEDS IMPROVEMENT" in feedback
        assert "WEAKNESSES" in feedback
        assert "Barren plateau" in feedback
        assert "SUGGESTED IMPROVEMENTS" in feedback


class TestQuickAnalyze:
    """Tests for quick_analyze function."""

    def test_quick_invalid(self):
        """Test quick analysis for invalid circuit."""
        result = quick_analyze(is_valid=False, is_trainable=True)

        assert result["status"] == "invalid"
        assert result["score"] == 0

    def test_quick_not_trainable(self):
        """Test quick analysis for non-trainable circuit."""
        result = quick_analyze(is_valid=True, is_trainable=False)

        assert result["status"] == "not_trainable"
        assert result["score"] == 20

    def test_quick_excellent(self):
        """Test quick analysis for excellent circuit."""
        result = quick_analyze(
            is_valid=True,
            is_trainable=True,
            energy_error=0.005,
            depth=8,
        )

        assert result["status"] == "success"
        assert result["score"] >= 80

    def test_quick_needs_improvement(self):
        """Test quick analysis for circuit needing improvement."""
        result = quick_analyze(
            is_valid=True,
            is_trainable=True,
            energy_error=0.1,
        )

        assert result["status"] == "needs_improvement"


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = AnalysisResult(
            is_successful=True,
            score=85.5,
            validity_score=100,
            trainability_score=90,
            performance_score=80,
            efficiency_score=85,
            summary="Good circuit",
            improvements=["Optimize depth"],
            strengths=["Valid", "Trainable"],
            weaknesses=["Slightly deep"],
        )

        assert result.is_successful is True
        assert result.score == 85.5
        assert len(result.improvements) == 1
        assert len(result.strengths) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
