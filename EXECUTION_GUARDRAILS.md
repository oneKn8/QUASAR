# Execution Guardrails

## MANDATORY RULES - NO EXCEPTIONS

This document defines strict guardrails for executing the QuantumMind project. These rules are non-negotiable. Violating them means the project fails.

---

## Rule 1: Research Before Code

**NEVER write code without understanding what you're building.**

Before ANY implementation:
1. Read the relevant section in this documentation
2. Search for existing implementations
3. Understand the quantum mechanics involved
4. Document your approach FIRST

```
WRONG: "Let me just start coding the circuit generator"
RIGHT: "Let me research how Agent-Q generates circuits, understand the
       OpenQASM format, then implement based on proven patterns"
```

---

## Rule 2: End-to-End Execution

**Every feature must work end-to-end before moving on.**

Do NOT:
- Write skeleton code with TODOs
- Leave functions unimplemented
- Say "I'll come back to this later"
- Build partial systems

Do:
- Implement completely
- Test immediately
- Verify on real inputs
- Document results

```
WRONG: def generate_circuit(prompt): pass  # TODO: implement
RIGHT: Full implementation, tested with 3+ examples, documented
```

---

## Rule 3: No Complaints, No Excuses

**If something breaks, fix it. Don't explain why it's hard.**

Acceptable responses:
- "This failed because X. Here's the fix."
- "The error is Y. Implementing solution Z."
- "Found a better approach. Switching to it."

Unacceptable responses:
- "This is too complex"
- "I can't do this because..."
- "This might not work"
- "Let me try a simpler approach"

---

## Rule 4: Verify Everything

**Never assume code works. Prove it.**

Every function must have:
1. At least 3 test cases
2. Edge case handling
3. Error messages that explain failures
4. Documented expected behavior

```python
# WRONG
def detect_barren_plateau(circuit):
    return some_calculation(circuit)

# RIGHT
def detect_barren_plateau(circuit: QuantumCircuit) -> BarrenPlateauResult:
    """
    Detect barren plateau risk in a quantum circuit.

    Args:
        circuit: Qiskit QuantumCircuit to analyze

    Returns:
        BarrenPlateauResult with:
        - has_bp: bool - True if BP detected
        - gradient_variance: float - Measured variance
        - confidence: float - Detection confidence

    Raises:
        InvalidCircuitError: If circuit has no parameters

    Example:
        >>> circ = create_test_circuit()
        >>> result = detect_barren_plateau(circ)
        >>> assert result.gradient_variance > 0
    """
    # Full implementation here
    ...
```

---

## Rule 5: Log Everything

**Every experiment must be reproducible.**

Required logging:
1. All hyperparameters
2. Random seeds
3. Dataset versions
4. Model checkpoints
5. Results with timestamps

```python
experiment_log = {
    "timestamp": datetime.now().isoformat(),
    "seed": 42,
    "model": "Qwen2.5-Coder-7B",
    "dataset_version": "QuantumLLMInstruct-v1",
    "hyperparameters": {
        "learning_rate": 2e-4,
        "batch_size": 2,
        "lora_rank": 32,
        ...
    },
    "results": {
        "loss": 0.xxx,
        "accuracy": 0.xxx,
        ...
    }
}
```

---

## Rule 6: Sequential Thinking

**Complex problems require structured thinking.**

Before tackling any complex task:
1. Break into steps
2. Identify dependencies
3. Plan the order
4. Execute sequentially
5. Verify each step before proceeding

```
WRONG: Just start implementing the whole agent at once

RIGHT:
Step 1: Implement Hamiltonian builder -> Test -> Verify
Step 2: Implement circuit generator -> Test -> Verify
Step 3: Implement barren plateau detector -> Test -> Verify
Step 4: Implement simulator interface -> Test -> Verify
Step 5: Connect components -> Integration test -> Verify
Step 6: Add IBM hardware interface -> Test -> Verify
Step 7: Full system test -> Verify
```

---

## Rule 7: Use The Documentation

**This documentation exists for a reason. Use it.**

Before implementing anything:
1. Check if it's documented in this project
2. Follow the documented approach
3. Update docs if you find improvements
4. Never deviate without documenting why

The documentation files are:
- `ARCHITECTURE.md` - HOW the system works
- `APPROACH.md` - WHY we make these choices
- `BUILD_PLAN.md` - WHAT to build in what order
- `MODELS.md` - Model configuration details
- `DATASETS.md` - Data sources and formats
- `EVALUATION.md` - How to measure success

---

## Rule 8: Hardware Validation Required

**Simulation results mean nothing without hardware validation.**

The goal is real quantum hardware results. Every claimed result must:
1. Work in Qiskit Aer simulator first
2. Be validated on IBM quantum hardware
3. Include noise/error analysis
4. Be reproducible with provided configs

```
WRONG: "The circuit works in simulation"
RIGHT: "The circuit achieves X% fidelity on ibm_sherbrooke with
       error mitigation enabled. Here are the job IDs: [...]"
```

---

## Rule 9: Benchmark Against Baselines

**Novel doesn't mean good. Better means good.**

Every discovered circuit must be compared against:
1. Hardware-Efficient Ansatz (HEA)
2. UCCSD (Unitary Coupled Cluster)
3. Previous best published result
4. GPT-4/Claude circuit generation (if applicable)

Metrics to report:
- Energy error (vs exact solution)
- Circuit depth
- Gate count
- Fidelity
- Runtime

---

## Rule 10: Paper-Ready Output

**Everything you produce should be publishable.**

Standards:
- Figures: Publication quality (300+ DPI, clear labels)
- Tables: Complete with standard deviations
- Code: Clean, documented, reproducible
- Claims: Backed by data, not opinions

```
WRONG: "The results look good"
RIGHT: "Our method achieves 0.023 Ha energy error (std: 0.003)
       compared to 0.041 Ha (std: 0.005) for HEA baseline,
       representing a 44% improvement (p < 0.01)."
```

---

## Execution Checklist

Before starting ANY coding session:

- [ ] Read relevant documentation sections
- [ ] Understand what you're building and WHY
- [ ] Have test cases ready
- [ ] Set up logging
- [ ] Know the success criteria

Before ending ANY coding session:

- [ ] All new code is tested
- [ ] All tests pass
- [ ] Changes are documented
- [ ] Experiments are logged
- [ ] Next steps are clear

---

## The Mindset

This is not a school project where "it runs" is success.

This is research where:
- **Correctness** is mandatory
- **Reproducibility** is mandatory
- **Documentation** is mandatory
- **Novel contribution** is the goal

Every line of code should be written as if it's going into a paper.

Because it might.

---

## Enforcement

If you (the AI assistant) are about to:
- Skip testing
- Leave something unimplemented
- Make excuses about complexity
- Deviate from the documented approach

**STOP. Re-read this document. Follow the rules.**

The user has explicitly requested end-to-end execution with no complaints and no breaks. Honor that request.

---

## Summary

| Rule | One-Liner |
|------|-----------|
| 1 | Research before code |
| 2 | Complete end-to-end |
| 3 | Fix problems, don't complain |
| 4 | Test everything |
| 5 | Log everything |
| 6 | Think sequentially |
| 7 | Use the documentation |
| 8 | Validate on hardware |
| 9 | Beat baselines |
| 10 | Publish-ready output |

---

*These rules exist to ensure success. Follow them.*
