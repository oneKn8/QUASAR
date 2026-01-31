"""
Agent module for QuantumMind.

Provides the autonomous circuit discovery agent components.
"""

from src.agent.analyzer import (
    AnalysisResult,
    ComparisonResult,
    ResultAnalyzer,
    quick_analyze,
)
from src.agent.discovery import (
    DiscoveryAgent,
    DiscoveryConfig,
    DiscoveryResult,
    run_discovery,
    run_multi_discovery,
)
from src.agent.memory import (
    AgentMemory,
    CircuitRecord,
    MemoryConfig,
    create_feedback,
)
from src.agent.proposer import (
    CircuitProposal,
    CircuitProposer,
    MockProposer,
    ProposerConfig,
    build_circuit_prompt,
    extract_code_from_response,
    extract_reasoning_from_response,
)

__all__ = [
    # Analyzer
    "AnalysisResult",
    "ComparisonResult",
    "ResultAnalyzer",
    "quick_analyze",
    # Discovery
    "DiscoveryAgent",
    "DiscoveryConfig",
    "DiscoveryResult",
    "run_discovery",
    "run_multi_discovery",
    # Memory
    "AgentMemory",
    "CircuitRecord",
    "MemoryConfig",
    "create_feedback",
    # Proposer
    "CircuitProposal",
    "CircuitProposer",
    "MockProposer",
    "ProposerConfig",
    "build_circuit_prompt",
    "extract_code_from_response",
    "extract_reasoning_from_response",
]
