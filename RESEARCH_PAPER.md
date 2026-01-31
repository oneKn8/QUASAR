# Research Paper Plan

## Target Venues

| Venue | Type | Deadline | Fit |
|-------|------|----------|-----|
| **QIP** | Workshop | ~Oct | Top quantum venue |
| **IEEE QCE** | Conference | ~Apr | Applied quantum |
| **NeurIPS (ML4Phys)** | Workshop | ~Sep | ML for science |
| **arXiv** | Preprint | Anytime | Immediate visibility |
| **Quantum** | Journal | Rolling | Open access |

**Strategy**: Submit to arXiv first (visibility), then IEEE QCE or QIP workshop.

---

## Paper Title Options

1. "QuantumMind: Autonomous Discovery of Quantum Circuits via LLM-Guided Search"
2. "LLM-Driven Quantum Algorithm Discovery: An Autonomous Approach"
3. "Beyond Human Design: Autonomous Discovery of Variational Quantum Ansatzes"
4. "Self-Improving Quantum Circuit Design with Large Language Models"

**Recommended**: Option 1 (clear, memorable, describes contribution)

---

## Abstract Template

```
Designing variational quantum circuits (ansatzes) for specific physics
problems remains a significant bottleneck in quantum computing research.
Current approaches rely heavily on human expertise or computationally
expensive architecture search methods. We present QuantumMind, an
autonomous agent that discovers novel quantum circuits using a fine-tuned
large language model (LLM) combined with iterative feedback from quantum
simulation and hardware execution.

QuantumMind integrates (1) an LLM fine-tuned on quantum computing datasets
to propose circuit architectures, (2) automated barren plateau detection
to screen untrainable candidates, (3) a memory system that accumulates
knowledge from successful and failed experiments, and (4) validation on
real IBM quantum hardware.

We demonstrate QuantumMind on spin chain ground state preparation problems
(XY, Heisenberg, TFIM models). On the 6-qubit XY chain, our discovered
circuit achieves [X]% lower energy error than the standard hardware-efficient
ansatz while using [Y]% fewer gates. The discovered architecture exhibits
[novel property], suggesting [insight]. All discovered circuits are validated
on IBM's 127-qubit Eagle processor, confirming simulation results.

Our work demonstrates that LLMs can serve as effective hypothesis generators
for quantum algorithm discovery, opening new avenues for AI-assisted
quantum research.
```

---

## Paper Structure

### 1. Introduction (1-1.5 pages)

**Opening**: The promise and challenge of variational quantum algorithms
- NISQ era requires efficient ansatzes
- Manual design is bottleneck
- Existing automated methods are expensive

**Gap**: No autonomous discovery system exists
- Architecture search requires many evaluations
- LLMs can generate code but don't "discover"
- Need closed-loop system with learning

**Contribution**: QuantumMind
- First fully autonomous quantum circuit discovery agent
- Combines LLM + barren plateau detection + memory + hardware
- Discovers novel circuits that beat baselines
- Validated on real quantum hardware

**Paper outline**: Brief description of sections

---

### 2. Related Work (1 page)

**Variational Quantum Algorithms**
- VQE, QAOA fundamentals
- Ansatz design importance
- Cite: Peruzzo 2014, Cerezo 2021

**Quantum Architecture Search**
- Evolutionary approaches
- Reinforcement learning methods
- Cite: Du 2020, Zhang 2021

**LLMs for Quantum Computing**
- Agent-Q (circuit generation)
- QGAS (ansatz search)
- QuantumLLMInstruct (dataset)
- Cite: relevant papers

**Our Distinction**: First to close the loop with autonomous iteration + hardware validation

---

### 3. Method (2-3 pages)

#### 3.1 System Overview

```
Figure 1: QuantumMind Architecture
[Diagram from ARCHITECTURE.md]
```

- Physics goal input
- LLM circuit proposal
- Verification + BP detection
- VQE execution
- Result analysis
- Memory update
- Loop

#### 3.2 LLM Fine-Tuning

- Base model: Qwen2.5-Coder-7B
- Dataset: QuantumLLMInstruct (filtered)
- Method: QLoRA
- Prompt engineering for circuit generation

#### 3.3 Barren Plateau Detection

- Gradient variance sampling
- Early rejection of untrainable circuits
- Cite: McClean 2018, Cerezo 2021

#### 3.4 Memory System

- Success/failure tracking
- Pattern extraction
- Context injection for LLM

#### 3.5 Hardware Validation

- IBM Quantum Runtime
- Error mitigation (resilience level 2)
- Multiple runs for statistics

---

### 4. Experiments (2-3 pages)

#### 4.1 Experimental Setup

- Target problems: XY chain, Heisenberg, TFIM
- Qubit counts: 4, 6, 8
- Baselines: HEA, EfficientSU2
- Metrics: Energy error, depth, gate count

#### 4.2 Results: XY Spin Chain

```
Table 1: XY Chain Results
[Results from experiments]
```

```
Figure 2: Energy Error Comparison
[Bar chart]
```

- QuantumMind achieves X% improvement
- Discovered circuit has Y properties
- Hardware validation confirms

#### 4.3 Results: Heisenberg Model

Similar structure

#### 4.4 Results: Transverse-Field Ising

Similar structure

#### 4.5 Analysis of Discovered Circuits

```
Figure 3: Discovered Circuit Architectures
[Circuit diagrams]
```

- What patterns emerge?
- Why do they work?
- Novel architectural insights

---

### 5. Discussion (1 page)

**Key Findings**
- LLMs can discover effective circuits
- Memory enables learning without retraining
- BP detection is crucial
- Hardware validation confirms simulation

**Limitations**
- Limited to specific problem types
- Requires significant compute for fine-tuning
- Discovery depends on training data quality

**Broader Impact**
- AI-assisted quantum research
- Potential for other physics problems
- Ethical considerations minimal (basic research)

---

### 6. Conclusion (0.5 pages)

- Summary of contributions
- Discovered circuits available at [repo]
- Future work: larger systems, other algorithms, multi-objective

---

### References

Cite all papers mentioned in documentation:
- McClean et al. (Barren plateaus)
- Agent-Q paper
- QuantumLLMInstruct
- Qiskit papers
- VQE/QAOA fundamentals

---

## Figures and Tables

### Required Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| Fig 1 | System architecture | Show method |
| Fig 2 | Energy error comparison | Main result |
| Fig 3 | Discovered circuits | Show novelty |
| Fig 4 | Convergence curves | Show VQE behavior |
| Fig 5 | Hardware vs simulation | Validate |

### Required Tables

| Table | Content | Purpose |
|-------|---------|---------|
| Tab 1 | Main results | Quantitative comparison |
| Tab 2 | Circuit statistics | Depth, gates, params |
| Tab 3 | Ablation study | Component importance |
| Tab 4 | Hardware results | Real QPU numbers |

---

## Writing Guidelines

### Style
- Clear, concise, technical
- Present tense for methods, past tense for experiments
- Active voice preferred
- Define all notation

### Length
- Conference: 8 pages + refs
- Workshop: 4-6 pages + refs
- arXiv: No limit, aim for 10-12

### LaTeX Template

```latex
\documentclass{article}
\usepackage{neurips_2026}  % or appropriate venue style

\title{QuantumMind: Autonomous Discovery of Quantum Circuits\\
via LLM-Guided Search}

\author{
  Shifat Islam Santo \\
  University of Texas at Dallas \\
  \texttt{sxs220461@utdallas.edu}
}

\begin{document}
\maketitle
\begin{abstract}
...
\end{abstract}

\section{Introduction}
...

\end{document}
```

---

## Writing Schedule

| Phase | Task | Output |
|-------|------|--------|
| 1 | Outline complete | This document |
| 2 | Methods written | Section 3 draft |
| 3 | Experiments done | All results |
| 4 | Results written | Sections 4-5 draft |
| 5 | Intro/conclusion | Full draft |
| 6 | Figures finalized | Publication quality |
| 7 | Internal review | Revised draft |
| 8 | Polish | Camera ready |

---

## Checklist Before Submission

- [ ] All results are reproducible
- [ ] Code is available (GitHub link)
- [ ] Models are available (HuggingFace)
- [ ] Figures are 300+ DPI
- [ ] All claims have supporting data
- [ ] Statistical significance reported
- [ ] Hardware results included
- [ ] Related work is complete
- [ ] No typos or grammatical errors
- [ ] Follows venue formatting

---

## Supplementary Material

### Appendix A: Discovered Circuits

Full circuit diagrams and Qiskit code for all discovered ansatzes.

### Appendix B: Hyperparameters

Complete training configuration, VQE settings, hardware parameters.

### Appendix C: Additional Results

Results on other qubit counts, ablation studies, failure analysis.

### Appendix D: Prompt Examples

Sample prompts used for LLM circuit generation.

---

## GitHub Repository Structure

```
quantum-mind/
├── README.md              # Project overview
├── paper/
│   ├── main.tex           # Paper source
│   ├── figures/           # All figures
│   ├── references.bib     # Bibliography
│   └── supplementary.tex  # Appendix
├── src/                   # Code
├── data/                  # Datasets (or links)
├── models/                # Model weights (or HF link)
├── experiments/           # Experiment logs
└── notebooks/             # Reproducibility notebooks
```

---

## Key Claims to Support

Each claim in the paper needs evidence:

| Claim | Evidence Required |
|-------|-------------------|
| "First autonomous agent" | Literature review showing gap |
| "Beats baselines by X%" | Statistical comparison |
| "Novel circuit architecture" | Analysis of discovered patterns |
| "Works on hardware" | IBM job IDs, results |
| "Memory enables learning" | Ablation without memory |

---

## Potential Reviewer Concerns

| Concern | How to Address |
|---------|----------------|
| "Just prompt engineering" | Show fine-tuning improves, memory matters |
| "Small systems only" | Acknowledge, show scaling trends |
| "Not generalizable" | Multiple Hamiltonians, different structures |
| "Needs more baselines" | Include UCCSD if applicable |
| "Hardware noise" | Report both sim and hardware, discuss gap |

---

## Paper Writing Mantra

1. **Every claim needs data**
2. **Every figure needs a purpose**
3. **Every baseline needs to be fair**
4. **Every limitation needs acknowledgment**
5. **Every contribution needs to be clear**

Write the paper that you would want to read.
