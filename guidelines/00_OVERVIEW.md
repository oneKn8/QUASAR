# QuantumMind Guideline: Overview

> Quick reference for project state. Read only this file to know where we are.

---

## Project Summary

**What**: QuantumMind is an autonomous agent that discovers novel quantum circuits using a fine-tuned LLM, barren plateau detection, iterative feedback, and IBM quantum hardware validation.

**Goal**: Discover circuits for spin chain ground states (XY, Heisenberg, TFIM) that outperform human-designed ansatzes by 10%+ in energy error while using fewer gates.

**Claim**: "I built an autonomous agent that discovered novel quantum circuits for simulating spin chains. The system fine-tunes an LLM to propose circuits, detects barren plateaus before they kill training, and iterates based on real IBM quantum hardware results."

---

## Current State (as of 2026-01-30)

### Completed Components

| Component | Location | Tests | Status |
|-----------|----------|-------|--------|
| Project structure | `src/` directories | - | DONE |
| Virtual environment | `venv/` | - | DONE |
| Qiskit verification | `test_qiskit.py` | 5 | DONE |
| Hamiltonians | `src/quantum/hamiltonians.py` | 29 | DONE |
| Barren plateau detector | `src/quantum/barren_plateau.py` | 17 | DONE |
| Circuit verifier | `src/quantum/verifier.py` | 25 | DONE |
| Quantum executor | `src/quantum/executor.py` | 21 | DONE |
| Dataset utilities | `src/training/dataset.py` | 24 | DONE |
| LLM Proposer | `src/agent/proposer.py` | 25 | DONE |
| Memory system | `src/agent/memory.py` | 23 | DONE |
| Result analyzer | `src/agent/analyzer.py` | 21 | DONE |
| Discovery agent | `src/agent/discovery.py` | 18 | DONE |

**Total Tests Passing: 203**

### Pending Components

| Component | Guideline Doc | Depends On | Status |
|-----------|---------------|------------|--------|
| IBM token setup | `01_ENVIRONMENT.md` | User action | BLOCKED |
| Fine-tuned model | `03_FINETUNING.md` | Dataset + GPU | PENDING |
| Model validation | `03_FINETUNING.md` | Fine-tuned model | PENDING |
| Baseline ansatzes | `05_EVALUATION.md` | - | PENDING |
| Metrics module | `05_EVALUATION.md` | - | PENDING |
| Comparative evaluation | `05_EVALUATION.md` | Baselines + Metrics | PENDING |
| Statistical tests | `05_EVALUATION.md` | Comparative eval | PENDING |
| IBM hardware runner | `06_HARDWARE.md` | IBM token | PENDING |
| Error mitigation | `06_HARDWARE.md` | Hardware runner | PENDING |
| Noise analysis | `06_HARDWARE.md` | Hardware runner | PENDING |
| XY Chain discovery | `07_DISCOVERY.md` | All above | PENDING |
| Heisenberg discovery | `07_DISCOVERY.md` | All above | PENDING |
| TFIM discovery | `07_DISCOVERY.md` | All above | PENDING |
| Results documentation | `08_PAPER.md` | Discoveries | PENDING |
| Paper draft | `08_PAPER.md` | Results | PENDING |

---

## Phase Progress

| Phase | Progress | Next Guideline |
|-------|----------|----------------|
| Phase 1: Environment | 3/4 | `01_ENVIRONMENT.md` |
| Phase 2: Quantum Components | 4/4 COMPLETE | - |
| Phase 3: Fine-Tuning | 3/5 | `03_FINETUNING.md` |
| Phase 4: Agent | 4/4 COMPLETE | - |
| Phase 5: Evaluation | 0/4 | `05_EVALUATION.md` |
| Phase 6: Hardware | 0/3 | `06_HARDWARE.md` |
| Phase 7: Discovery | 0/3 | `07_DISCOVERY.md` |
| Phase 8: Documentation | 0/2 | `08_PAPER.md` |
| **TOTAL** | **14/29** | |

---

## Sequential Execution Order

1. **Phase 1.3** - IBM token setup (user action required)
2. **Phase 3.4** - Fine-tune model
3. **Phase 3.5** - Validate fine-tuned model
4. **Phase 5.1** - Baseline ansatzes (can run parallel with 3.4)
5. **Phase 5.2** - Metrics module
6. **Phase 5.3** - Comparative evaluation
7. **Phase 5.4** - Statistical tests
8. **Phase 6.1** - IBM hardware runner
9. **Phase 6.2** - Error mitigation
10. **Phase 6.3** - Noise analysis
11. **Phase 7.1** - XY Chain discovery
12. **Phase 7.2** - Heisenberg discovery
13. **Phase 7.3** - TFIM discovery
14. **Phase 8.1** - Results documentation
15. **Phase 8.2** - Paper draft

---

## Rules (Non-Negotiable)

1. **100% complete or not done** - No MVPs, no skeletons
2. **All checks must pass** - Before moving to next step
3. **Sequential execution** - No skipping ahead
4. **Log everything** - Every experiment, every result
5. **Test everything** - No untested code
6. **Document as you go** - Not at the end

---

## How to Use These Guidelines

1. Check this file (`00_OVERVIEW.md`) to see current progress
2. Read ONLY the guideline doc for the current phase
3. Complete ALL verification checks before moving on
4. Update this overview when a phase is complete

---

*This file is the index. Individual phase docs contain complete code and verification checklists.*
