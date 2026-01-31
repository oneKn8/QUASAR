"""
Main discovery agent for QuantumMind.

Orchestrates the autonomous circuit discovery loop by combining:
- LLM proposer for circuit generation
- Circuit verifier for validation
- Barren plateau detector for trainability
- VQE executor for evaluation
- Memory system for learning
- Result analyzer for feedback
"""

import time
from dataclasses import dataclass, field
from typing import Callable

from qiskit import QuantumCircuit

from src.agent.analyzer import AnalysisResult, ResultAnalyzer
from src.agent.memory import AgentMemory, MemoryConfig, create_feedback
from src.agent.proposer import CircuitProposal, MockProposer, ProposerConfig
from src.quantum.barren_plateau import detect_barren_plateau, quick_bp_heuristic
from src.quantum.executor import ExecutorConfig, QuantumExecutor, VQEResult
from src.quantum.hamiltonians import HamiltonianResult, get_hamiltonian
from src.quantum.verifier import VerificationResult, verify_circuit_code


@dataclass
class DiscoveryConfig:
    """Configuration for the discovery agent."""

    max_iterations: int = 50
    max_failed_attempts: int = 10
    target_energy_error: float = 0.01
    use_mock_proposer: bool = True
    run_vqe: bool = True
    vqe_max_iterations: int = 200
    check_barren_plateau: bool = True
    bp_num_samples: int = 20
    verbose: bool = True
    early_stop_on_success: bool = True
    memory_persistence_path: str | None = None


@dataclass
class DiscoveryResult:
    """Result of a discovery run."""

    success: bool
    best_circuit_code: str | None
    best_energy: float | None
    best_energy_error: float | None
    total_iterations: int
    successful_circuits: int
    failed_circuits: int
    best_analysis: AnalysisResult | None
    all_records: list = field(default_factory=list)
    runtime_seconds: float = 0


class DiscoveryAgent:
    """
    Autonomous quantum circuit discovery agent.

    Uses an LLM to propose circuits, evaluates them through verification,
    barren plateau detection, and VQE, then provides feedback to guide
    the next proposal.
    """

    def __init__(
        self,
        config: DiscoveryConfig | None = None,
        proposer_config: ProposerConfig | None = None,
        memory_config: MemoryConfig | None = None,
    ):
        """
        Initialize the discovery agent.

        Args:
            config: Discovery configuration
            proposer_config: Proposer configuration
            memory_config: Memory configuration
        """
        self.config = config or DiscoveryConfig()

        # Initialize components
        if self.config.use_mock_proposer:
            self._proposer = MockProposer()
        else:
            from src.agent.proposer import CircuitProposer

            self._proposer = CircuitProposer(proposer_config)

        mem_config = memory_config or MemoryConfig(
            persistence_path=self.config.memory_persistence_path
        )
        self._memory = AgentMemory(mem_config)
        self._analyzer = ResultAnalyzer(
            target_energy_error=self.config.target_energy_error
        )
        self._executor = QuantumExecutor(
            ExecutorConfig(max_iterations=self.config.vqe_max_iterations)
        )

        self._stats = {
            "total_discoveries": 0,
            "successful_discoveries": 0,
            "total_iterations": 0,
        }

    def discover(
        self,
        hamiltonian_type: str,
        num_qubits: int,
        goal_description: str | None = None,
        constraints: dict | None = None,
        callback: Callable[[int, str, AnalysisResult | None], None] | None = None,
    ) -> DiscoveryResult:
        """
        Run the discovery loop for a specific physics problem.

        Args:
            hamiltonian_type: Type of Hamiltonian (xy_chain, heisenberg, etc.)
            num_qubits: Number of qubits
            goal_description: Optional custom goal description
            constraints: Optional circuit constraints
            callback: Optional callback(iteration, status, analysis)

        Returns:
            Discovery result with best circuit and statistics
        """
        start_time = time.time()
        self._stats["total_discoveries"] += 1

        if goal_description is None:
            goal_description = (
                f"Design a variational quantum circuit (ansatz) to find the "
                f"ground state energy of the {hamiltonian_type} model with "
                f"{num_qubits} qubits."
            )

        # Get Hamiltonian
        ham_result = get_hamiltonian(hamiltonian_type, num_qubits=num_qubits)

        if self.config.verbose:
            print(f"Starting discovery for {hamiltonian_type} with {num_qubits} qubits")
            print(f"Exact energy: {ham_result.exact_energy:.6f}")
            print(f"Target error: {self.config.target_energy_error}")
            print("-" * 50)

        best_result = None
        best_energy_error = float("inf")
        consecutive_failures = 0
        iteration = 0

        while iteration < self.config.max_iterations:
            iteration += 1
            self._stats["total_iterations"] += 1

            if self.config.verbose:
                print(f"\nIteration {iteration}/{self.config.max_iterations}")

            # Get context from memory
            context = self._memory.get_context_for_proposal(
                hamiltonian_type=hamiltonian_type,
                num_qubits=num_qubits,
            )

            # Get feedback from last attempt
            feedback = None
            if best_result is not None:
                feedback = self._analyzer.format_feedback_for_llm(best_result)

            # Generate proposal
            proposal = self._proposer.propose(
                goal_description=goal_description,
                num_qubits=num_qubits,
                hamiltonian_type=hamiltonian_type,
                constraints=constraints,
                feedback=feedback,
            )

            if not proposal.code:
                if self.config.verbose:
                    print("  No code generated, skipping")
                consecutive_failures += 1
                if consecutive_failures >= self.config.max_failed_attempts:
                    break
                continue

            # Evaluate proposal
            analysis = self._evaluate_proposal(
                proposal=proposal,
                ham_result=ham_result,
                num_qubits=num_qubits,
            )

            # Update memory
            self._update_memory(
                proposal=proposal,
                analysis=analysis,
                hamiltonian_type=hamiltonian_type,
                num_qubits=num_qubits,
            )

            # Check for improvement
            energy_error = analysis.details.get("energy_error")

            # Track best even if not fully successful (for returning something)
            if energy_error is not None and energy_error < best_energy_error:
                best_energy_error = energy_error
                best_result = analysis
                best_result.details["code"] = proposal.code

                if self.config.verbose:
                    print(f"  New best! Energy error: {energy_error:.6f}")

            # Check for target reached (independent of is_successful)
            target_reached = (
                energy_error is not None
                and energy_error <= self.config.target_energy_error
                and analysis.validity_score >= 80
            )

            if target_reached:
                if self.config.verbose:
                    print(f"  Target reached!")
                if self.config.early_stop_on_success:
                    break

            if analysis.is_successful or analysis.validity_score >= 80:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            # Callback
            if callback:
                callback(iteration, "iteration_complete", analysis)

            # Check failure limit
            if consecutive_failures >= self.config.max_failed_attempts:
                if self.config.verbose:
                    print(f"  Too many consecutive failures, stopping")
                break

        # Compile results
        runtime = time.time() - start_time
        stats = self._memory.get_statistics(hamiltonian_type=hamiltonian_type)

        success = best_result is not None and best_energy_error <= self.config.target_energy_error

        if success:
            self._stats["successful_discoveries"] += 1

        if self.config.verbose:
            print("\n" + "=" * 50)
            print(f"Discovery complete in {runtime:.1f}s")
            print(f"Success: {success}")
            print(f"Best energy error: {best_energy_error:.6f}")
            print(f"Iterations: {iteration}")

        return DiscoveryResult(
            success=success,
            best_circuit_code=best_result.details.get("code") if best_result else None,
            best_energy=best_result.details.get("energy") if best_result else None,
            best_energy_error=best_energy_error if best_energy_error != float("inf") else None,
            total_iterations=iteration,
            successful_circuits=stats.get("successful_count", 0),
            failed_circuits=stats.get("failed_count", 0),
            best_analysis=best_result,
            runtime_seconds=runtime,
        )

    def _evaluate_proposal(
        self,
        proposal: CircuitProposal,
        ham_result: HamiltonianResult,
        num_qubits: int,
    ) -> AnalysisResult:
        """
        Evaluate a circuit proposal through the full pipeline.

        Args:
            proposal: The circuit proposal
            ham_result: Hamiltonian result with operator and exact energy
            num_qubits: Expected number of qubits

        Returns:
            Complete analysis result
        """
        # Step 1: Verify code
        verification = verify_circuit_code(
            proposal.code,
            expected_qubits=num_qubits,
            require_params=True,
        )

        if not verification.is_valid:
            if self.config.verbose:
                print(f"  Verification failed: {verification.errors[0]}")
            return self._analyzer.analyze(
                circuit=None,
                verification_result=verification,
            )

        circuit = verification.circuit

        # Step 2: Check barren plateau (quick heuristic first)
        bp_result = None
        if self.config.check_barren_plateau:
            has_risk, risk_reason, risk_details = quick_bp_heuristic(circuit)
            if has_risk:
                if self.config.verbose:
                    print(f"  BP risk: {risk_reason}")
                # Do full check
                bp_result = detect_barren_plateau(
                    circuit,
                    ham_result.operator,
                    num_samples=self.config.bp_num_samples,
                )
                if bp_result.has_barren_plateau:
                    return self._analyzer.analyze(
                        circuit=circuit,
                        verification_result=verification,
                        bp_result=bp_result,
                    )
            else:
                # Assume trainable if quick check passes
                bp_result = type("BP", (), {
                    "has_barren_plateau": False,
                    "is_trainable": True,  # Keep for analyzer compatibility
                    "gradient_variance": 0.01,
                    "risk_factors": [],
                })()

        # Step 3: Run VQE
        vqe_result = None
        if self.config.run_vqe:
            try:
                vqe_result = self._executor.run_vqe(circuit, ham_result.operator)
                if self.config.verbose:
                    print(f"  VQE energy: {vqe_result.energy:.6f}")
            except Exception as e:
                if self.config.verbose:
                    print(f"  VQE failed: {e}")

        # Step 4: Analyze
        analysis = self._analyzer.analyze(
            circuit=circuit,
            verification_result=verification,
            vqe_result=vqe_result,
            bp_result=bp_result,
            exact_energy=ham_result.exact_energy,
        )

        # Add extra details
        if vqe_result:
            analysis.details["energy"] = vqe_result.energy
            analysis.details["energy_error"] = abs(
                vqe_result.energy - ham_result.exact_energy
            )

        return analysis

    def _update_memory(
        self,
        proposal: CircuitProposal,
        analysis: AnalysisResult,
        hamiltonian_type: str,
        num_qubits: int,
    ):
        """Update memory with the evaluation result."""
        self._memory.add_record(
            code=proposal.code,
            hamiltonian_type=hamiltonian_type,
            num_qubits=num_qubits,
            is_valid=analysis.validity_score >= 80,
            is_trainable=analysis.trainability_score >= 50,
            depth=analysis.details.get("circuit_depth", 0) or 0,
            param_count=analysis.details.get("param_count", 0) or 0,
            gate_count=analysis.details.get("gate_count", 0) or 0,
            reasoning=proposal.reasoning,
            feedback=analysis.summary,
            energy=analysis.details.get("energy"),
            energy_error=analysis.details.get("energy_error"),
        )

    def get_memory(self) -> AgentMemory:
        """Get the memory system."""
        return self._memory

    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            **self._stats,
            "memory_stats": self._memory.get_statistics(),
            "analyzer_stats": self._analyzer.get_stats(),
            "proposer_stats": self._proposer.get_stats(),
        }

    def reset(self):
        """Reset the agent state."""
        self._memory.clear()
        self._stats = {
            "total_discoveries": 0,
            "successful_discoveries": 0,
            "total_iterations": 0,
        }


def run_discovery(
    hamiltonian_type: str,
    num_qubits: int,
    max_iterations: int = 20,
    target_error: float = 0.01,
    verbose: bool = True,
) -> DiscoveryResult:
    """
    Convenience function to run a discovery campaign.

    Args:
        hamiltonian_type: Type of Hamiltonian
        num_qubits: Number of qubits
        max_iterations: Maximum iterations
        target_error: Target energy error
        verbose: Print progress

    Returns:
        Discovery result
    """
    config = DiscoveryConfig(
        max_iterations=max_iterations,
        target_energy_error=target_error,
        verbose=verbose,
        use_mock_proposer=True,  # Use mock for now
    )

    agent = DiscoveryAgent(config)
    return agent.discover(hamiltonian_type, num_qubits)


def run_multi_discovery(
    problems: list[tuple[str, int]],
    max_iterations_per_problem: int = 20,
    target_error: float = 0.01,
    verbose: bool = True,
) -> dict[str, DiscoveryResult]:
    """
    Run discovery on multiple problems.

    Args:
        problems: List of (hamiltonian_type, num_qubits) tuples
        max_iterations_per_problem: Max iterations per problem
        target_error: Target energy error
        verbose: Print progress

    Returns:
        Dictionary mapping problem names to results
    """
    results = {}

    for hamiltonian_type, num_qubits in problems:
        problem_name = f"{hamiltonian_type}_{num_qubits}q"

        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting: {problem_name}")
            print(f"{'='*60}")

        result = run_discovery(
            hamiltonian_type=hamiltonian_type,
            num_qubits=num_qubits,
            max_iterations=max_iterations_per_problem,
            target_error=target_error,
            verbose=verbose,
        )

        results[problem_name] = result

    return results
