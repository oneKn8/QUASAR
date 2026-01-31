# Research Approach

## The Scientific Question

**Can an LLM-based agent autonomously discover novel quantum circuits that outperform human-designed ansatzes?**

This is not "can LLMs generate quantum code" (proven: yes).
This is "can LLMs **discover** quantum algorithms" (open question).

---

## Hypothesis

**H1**: A fine-tuned LLM, when equipped with:
1. Quantum domain knowledge
2. Iterative feedback from execution
3. Memory of successful/failed patterns
4. Barren plateau awareness

Can discover quantum circuits that:
- Achieve lower energy error than standard ansatzes (HEA, UCCSD)
- Use fewer gates/shallower depth
- Generalize to larger system sizes

---

## Prior Work

### What Exists

| Work | What It Does | Limitation |
|------|--------------|------------|
| [Agent-Q](https://arxiv.org/abs/2504.11109) | Fine-tunes LLM for circuit generation | Reactive, not autonomous discovery |
| [QGAS](https://arxiv.org/abs/2307.08191) | LLM proposes ansatz architectures | Limited iteration, needs human guidance |
| [LLM-Guided Ansatz](https://arxiv.org/html/2509.08385) | Hardware-aware circuit design | Single-shot, no learning loop |
| [Scalable QSP](https://arxiv.org/html/2505.06347) | LLM discovers compact ansatzes | Validated but not fully autonomous |

### The Gap

No system exists that:
1. Runs continuously without human intervention
2. Learns from its own experiments
3. Validates on real quantum hardware
4. Discovers genuinely novel circuits

**QuantumMind fills this gap.**

---

## Research Methodology

### Phase 1: Establish Baselines

Before any discovery, we need baselines:

1. **Hardware-Efficient Ansatz (HEA)**
   - Standard parametrized circuit
   - Known to work but susceptible to barren plateaus
   - Benchmark: Energy error, depth, gate count

2. **UCCSD (Unitary Coupled Cluster)**
   - Chemistry-inspired ansatz
   - Better expressibility, worse hardware efficiency
   - Benchmark: Energy error, depth, gate count

3. **Random Circuit Sampling**
   - What happens if LLM has no guidance?
   - Establishes lower bound

4. **GPT-4/Claude Generation**
   - What can frontier models do without fine-tuning?
   - Establishes comparison to commercial models

### Phase 2: Fine-Tune LLM

Train Qwen2.5-Coder-7B on quantum data:

**Training Objective**: Generate valid, trainable quantum circuits

**Data Sources**:
- QuantumLLMInstruct (500k pairs)
- IBM Qiskit tutorials/examples
- Agent-Q circuit dataset
- Custom physics reasoning examples

**Validation**: The fine-tuned model must:
- Generate syntactically valid Qiskit code (>95%)
- Produce circuits that pass verification (>80%)
- Avoid barren plateaus more often than base model

### Phase 3: Agent Loop Development

Build the autonomous system:

1. **Proposer**: LLM generates circuits
2. **Verifier**: Checks validity
3. **BP Detector**: Screens for trainability
4. **Executor**: Runs VQE
5. **Analyzer**: Evaluates results
6. **Memory**: Stores learnings

**Key Innovation**: The memory system enables learning without weight updates.

### Phase 4: Discovery Campaigns

Run the agent on target problems:

**Target 1: XY Spin Chain (4-8 qubits)**
- Why: Analytically solvable, clear benchmark
- Goal: Beat HEA depth by 20%+ at same accuracy

**Target 2: Heisenberg Model (4-6 qubits)**
- Why: More complex entanglement structure
- Goal: Discover novel entanglement patterns

**Target 3: Transverse-Field Ising Model (4-8 qubits)**
- Why: Phase transitions, interesting physics
- Goal: Circuits that capture phase transition

### Phase 5: Hardware Validation

Critical step - simulation means nothing without hardware:

1. Select top 5 discovered circuits per target
2. Run on IBM Eagle processor
3. Compare simulated vs hardware performance
4. Document noise effects and mitigation

### Phase 6: Analysis and Publication

Scientific deliverables:

1. **Discovered Circuits**: Novel ansatz architectures
2. **Performance Data**: Systematic benchmarks
3. **Methodology**: Reproducible approach
4. **Insights**: What makes certain patterns work

---

## Control Variables

To ensure scientific validity:

| Variable | Control |
|----------|---------|
| Random seed | Fixed for reproducibility |
| Optimizer | Same (COBYLA) for all experiments |
| VQE iterations | Fixed max (200) |
| Error mitigation | Same settings for hardware |
| Qubit mapping | Consistent across runs |

---

## Expected Outcomes

### Minimum Viable Research
- Agent generates valid circuits
- Some circuits match baseline performance
- System is reproducible
- One target problem solved

### Target Outcome
- Discovered circuits beat baselines by 10-20%
- Novel circuit patterns identified
- Hardware validation confirms results
- Publishable contribution

### Stretch Outcome
- Discover genuinely new physics insight
- Circuits generalize to larger systems
- Paper accepted to quantum venue
- Framework adopted by others

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM generates garbage | Start with proven Agent-Q base |
| No improvement over baselines | Document negative results (still publishable) |
| Hardware too noisy | Use error mitigation, report both sim and HW |
| Too ambitious scope | Focus on 4-qubit systems first |
| Can't reproduce | Log everything, fix seeds |

---

## Theoretical Foundation

### Why LLMs Might Work for This

1. **Pattern Recognition**: LLMs excel at recognizing patterns in code/structure
2. **Knowledge Transfer**: Training data contains quantum computing knowledge
3. **Iterative Refinement**: Memory system enables learning without retraining
4. **Constraint Satisfaction**: LLMs can follow complex constraints in prompts

### Why This is Hard

1. **Quantum Mechanics is Non-Intuitive**: LLMs trained on classical data
2. **Barren Plateaus**: Most random circuits are untrainable
3. **Hardware Noise**: Simulation ≠ reality
4. **Optimization Landscape**: VQE is a hard optimization problem

### Our Approach to Hard Problems

1. **Fine-tuning**: Inject quantum knowledge
2. **BP Detection**: Screen bad circuits before wasting compute
3. **Memory**: Learn from failures
4. **Hardware-in-the-Loop**: Validate on real systems

---

## Contributions

If successful, QuantumMind contributes:

1. **System**: First fully autonomous quantum algorithm discovery agent
2. **Method**: LLM + feedback loop for quantum circuit design
3. **Circuits**: Novel ansatzes for spin chain Hamiltonians
4. **Insights**: What architectural patterns enable discovery

---

## Comparison to Related Fields

| Field | Similarity | Difference |
|-------|------------|------------|
| Neural Architecture Search | Automated design | Quantum constraints, physics goals |
| AlphaFold | AI discovers science | Protein folding vs quantum circuits |
| AutoML | Hyperparameter optimization | We optimize structure, not params |
| Program Synthesis | Generate code | Quantum code with physical validity |

---

## Success Criteria

### Quantitative

| Metric | Threshold |
|--------|-----------|
| Energy error | < 0.01 Ha (chemical accuracy) |
| Improvement over HEA | > 10% depth reduction |
| Hardware fidelity | > 80% |
| Valid circuit rate | > 60% |
| BP-free circuit rate | > 50% |

### Qualitative

- Novel circuit architecture not in literature
- Insights about what makes circuits trainable
- Reproducible methodology
- Clear presentation for publication

---

## Timeline-Free Milestones

(No dates, just sequence)

1. Baselines established and documented
2. LLM fine-tuned and validated
3. Agent loop working end-to-end
4. First discovery campaign completed
5. Hardware validation done
6. Analysis complete
7. Paper draft ready
8. Submission

---

## The Big Picture

We're not just building a tool. We're testing whether AI can do science.

If an LLM can discover quantum circuits that humans haven't found, that's evidence that AI can contribute to fundamental research - not just automate existing tasks.

This is the kind of work that matters.
