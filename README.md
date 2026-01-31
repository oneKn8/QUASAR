# QuantumMind: Autonomous Quantum Algorithm Discovery Agent

> An LLM-based agent that autonomously discovers novel quantum circuits by proposing, testing, analyzing, and iterating on real quantum hardware.

---

## What This Is

QuantumMind is not another chatbot. It's an **autonomous research agent** that:

1. Takes a physics problem as input (e.g., "Find ground state of 6-qubit XY spin chain")
2. Uses a fine-tuned LLM to propose quantum circuits
3. Detects and avoids barren plateaus before they kill training
4. Tests circuits on IBM quantum hardware
5. Learns from results and iterates
6. Discovers novel circuits that outperform human-designed ones

**The goal**: Build an AI system that does quantum research autonomously.

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical system design, components, data flow |
| [APPROACH.md](APPROACH.md) | Research methodology and scientific approach |
| [BUILD_PLAN.md](BUILD_PLAN.md) | Step-by-step build instructions |
| [EXECUTION_GUARDRAILS.md](EXECUTION_GUARDRAILS.md) | **STRICT RULES** - Must follow, no exceptions |
| [MODELS.md](MODELS.md) | Model selection, fine-tuning configuration |
| [DATASETS.md](DATASETS.md) | Data sources, preparation, formatting |
| [EVALUATION.md](EVALUATION.md) | Benchmarks, metrics, success criteria |
| [RESEARCH_PAPER.md](RESEARCH_PAPER.md) | Paper outline, writing plan |

---

## The Claim

> "I built QuantumMind, an autonomous agent that discovered novel quantum circuits for simulating spin chains. The system fine-tunes an LLM to propose circuits, detects barren plateaus before they kill training, and iterates based on real IBM quantum hardware results. The discovered circuits are X% shallower than human-designed ansatzes while achieving the same accuracy."

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
python --version

# Required packages
pip install qiskit qiskit-ibm-runtime qiskit-aer
pip install torch transformers datasets
pip install unsloth  # For fast fine-tuning
```

### IBM Quantum Setup

1. Sign up: https://quantum.cloud.ibm.com
2. Get API token from dashboard
3. Save to environment:
```bash
export IBM_QUANTUM_TOKEN="your_token_here"
```

### Directory Structure

```
quantum-mind/
├── README.md                    # This file
├── ARCHITECTURE.md              # System design
├── APPROACH.md                  # Research methodology
├── BUILD_PLAN.md                # Build instructions
├── EXECUTION_GUARDRAILS.md      # STRICT RULES
├── MODELS.md                    # Model configuration
├── DATASETS.md                  # Data sources
├── EVALUATION.md                # Benchmarks
├── RESEARCH_PAPER.md            # Paper outline
├── src/
│   ├── __init__.py
│   ├── agent/                   # Core agent logic
│   │   ├── __init__.py
│   │   ├── proposer.py          # LLM circuit proposer
│   │   ├── verifier.py          # Circuit verification
│   │   ├── executor.py          # Quantum execution
│   │   ├── analyzer.py          # Result analysis
│   │   └── memory.py            # Learning memory
│   ├── quantum/                 # Quantum utilities
│   │   ├── __init__.py
│   │   ├── hamiltonians.py      # Hamiltonian definitions
│   │   ├── circuits.py          # Circuit utilities
│   │   ├── barren_plateau.py    # BP detection
│   │   └── hardware.py          # IBM backend interface
│   ├── training/                # LLM fine-tuning
│   │   ├── __init__.py
│   │   ├── dataset.py           # Dataset preparation
│   │   ├── finetune.py          # Fine-tuning script
│   │   └── config.py            # Training config
│   └── evaluation/              # Benchmarking
│       ├── __init__.py
│       ├── metrics.py           # Evaluation metrics
│       └── baselines.py         # Baseline comparisons
├── data/
│   ├── raw/                     # Downloaded datasets
│   ├── processed/               # Prepared training data
│   └── results/                 # Experiment results
├── models/
│   └── checkpoints/             # Saved model weights
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_fine_tuning.ipynb
│   ├── 03_agent_testing.ipynb
│   └── 04_hardware_validation.ipynb
├── experiments/
│   └── logs/                    # Experiment logs
├── paper/
│   ├── main.tex
│   ├── figures/
│   └── references.bib
└── tests/
    └── test_*.py
```

---

## Core Principle

**This is research, not a toy project.**

Every decision is documented. Every experiment is logged. Every claim is verifiable.

The output is:
1. A working autonomous discovery system
2. Novel discovered circuits (the actual contribution)
3. A publishable research paper
4. An internship-worthy portfolio piece

---

## Key Resources

### Papers
- [Agent-Q: Fine-Tuning LLMs for Quantum Circuits](https://arxiv.org/abs/2504.11109)
- [LLM-Discovered Ansatzes](https://arxiv.org/html/2505.06347)
- [Barren Plateaus Two-Step Solution](https://arxiv.org/abs/2601.18060)
- [QuantumLLMInstruct Dataset](https://arxiv.org/abs/2412.20956)

### Tools
- [IBM Quantum Platform](https://quantum.cloud.ibm.com)
- [Qiskit Documentation](https://quantum.cloud.ibm.com/docs)
- [Unsloth Fine-Tuning](https://unsloth.ai)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

### Datasets
- [QuantumLLMInstruct (500k pairs)](https://huggingface.co/datasets/BoltzmannEntropy/QuantumLLMInstruct)
- [IBM Quantum Tutorials](https://quantum.cloud.ibm.com/docs/en/tutorials)
- [Qiskit Textbook Examples](https://github.com/Qiskit/textbook)

---

## The Energy

This project exists because:

1. **No one has built this** - Autonomous quantum algorithm discovery doesn't exist
2. **It's achievable** - LLMs can generate circuits (proven), IBM hardware is free
3. **It's publishable** - Novel system + potential novel discoveries = paper
4. **It's memorable** - "I built AI that discovers physics" beats any chatbot

The goal is not to complete a project. The goal is to **discover something new**.

---

## Navigation

Start with:
1. [EXECUTION_GUARDRAILS.md](EXECUTION_GUARDRAILS.md) - Read the rules first
2. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand the system
3. [BUILD_PLAN.md](BUILD_PLAN.md) - Start building

---

## Status

- [ ] Phase 1: Foundation (Environment, IBM setup, basic agent)
- [ ] Phase 2: Fine-Tuning (Dataset prep, LLM training)
- [ ] Phase 3: Discovery (Run agent, find novel circuits)
- [ ] Phase 4: Validation (Hardware testing, benchmarking)
- [ ] Phase 5: Publication (Paper writing, release)

---

*"The best way to predict the future is to invent it." - Alan Kay*

Let's discover some physics.
