# Datasets

## Overview

| Dataset | Size | Purpose | Source |
|---------|------|---------|--------|
| QuantumLLMInstruct | 500k pairs | Main fine-tuning | HuggingFace |
| IBM Qiskit Tutorials | ~100 examples | Code patterns | GitHub |
| Custom Physics Examples | ~500 examples | Physics reasoning | Created |
| Agent-Q Circuits | 14k circuits | Circuit patterns | Paper |

---

## Primary: QuantumLLMInstruct

### Description

QuantumLLMInstruct (QLMMI) is the largest quantum computing instruction dataset:
- 500,000+ problem-solution pairs
- 90 quantum computing domains
- QASM code generation
- Hamiltonian transformations
- Circuit decompositions

### Download

```python
from datasets import load_dataset

# Load from HuggingFace
dataset = load_dataset("BoltzmannEntropy/QuantumLLMInstruct")

# Inspect
print(f"Training examples: {len(dataset['train'])}")
print(f"Columns: {dataset['train'].column_names}")
print(f"Sample:\n{dataset['train'][0]}")
```

### Data Format

```json
{
  "instruction": "Generate a quantum circuit that prepares the Bell state |00> + |11>",
  "input": "Number of qubits: 2",
  "output": "from qiskit import QuantumCircuit\n\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\n",
  "category": "circuit_generation",
  "difficulty": "basic"
}
```

### Filtering

Not all examples are useful. Filter for:

```python
def filter_dataset(example):
    """Filter for circuit generation examples."""
    # Must have code output
    if "from qiskit" not in example["output"]:
        return False

    # Must be circuit-related
    relevant_categories = [
        "circuit_generation",
        "ansatz_design",
        "vqe",
        "qaoa",
        "hamiltonian"
    ]
    if example.get("category") not in relevant_categories:
        return False

    # Must be reasonable length
    if len(example["output"]) > 5000:
        return False

    return True

filtered_dataset = dataset["train"].filter(filter_dataset)
print(f"Filtered: {len(filtered_dataset)} examples")
```

---

## Secondary: IBM Qiskit Tutorials

### Source

```bash
git clone https://github.com/Qiskit/textbook.git data/raw/qiskit-textbook
git clone https://github.com/qiskit-community/qiskit-community-tutorials.git data/raw/qiskit-tutorials
```

### Extract Code Examples

```python
import os
import json
import nbformat

def extract_from_notebooks(notebook_dir):
    """Extract code cells from Jupyter notebooks."""
    examples = []

    for root, dirs, files in os.walk(notebook_dir):
        for file in files:
            if file.endswith('.ipynb'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        nb = nbformat.read(f, as_version=4)

                    for cell in nb.cells:
                        if cell.cell_type == 'code':
                            code = cell.source
                            if 'QuantumCircuit' in code or 'qiskit' in code:
                                examples.append({
                                    "source": file,
                                    "code": code
                                })
                except:
                    continue

    return examples

qiskit_examples = extract_from_notebooks("data/raw/qiskit-textbook")
print(f"Extracted {len(qiskit_examples)} Qiskit examples")
```

---

## Custom: Physics Reasoning Examples

### Purpose

QuantumLLMInstruct has code but lacks physics reasoning. Create custom examples that teach the LLM to reason about:
- Why certain circuit structures work
- How to match ansatz to Hamiltonian
- Barren plateau avoidance strategies

### Format

```json
{
  "instruction": "Design a variational ansatz for the XY spin chain ground state",
  "input": "Number of qubits: 4, Hamiltonian: H = -sum(XX + YY), Target: Ground state energy",
  "reasoning": "The XY model has U(1) symmetry (total Z magnetization conserved). An effective ansatz should:\n1. Preserve this symmetry\n2. Create appropriate entanglement patterns\n3. Be shallow to avoid barren plateaus\n\nI'll use a hardware-efficient ansatz with RY rotations (preserve real amplitudes) followed by linear CX entanglement.",
  "output": "from qiskit import QuantumCircuit\nfrom qiskit.circuit import Parameter\n\ndef create_ansatz(num_qubits: int) -> QuantumCircuit:\n    qc = QuantumCircuit(num_qubits)\n    params = []\n    \n    # Layer 1: Local rotations\n    for i in range(num_qubits):\n        p = Parameter(f'theta_{len(params)}')\n        params.append(p)\n        qc.ry(p, i)\n    \n    # Entanglement: Linear chain\n    for i in range(num_qubits - 1):\n        qc.cx(i, i + 1)\n    \n    # Layer 2: More rotations\n    for i in range(num_qubits):\n        p = Parameter(f'theta_{len(params)}')\n        params.append(p)\n        qc.ry(p, i)\n    \n    return qc"
}
```

### Examples to Create

| Topic | Count | Description |
|-------|-------|-------------|
| XY Chain | 50 | Various qubit counts, depths |
| Heisenberg | 50 | XXX model circuits |
| TFIM | 50 | Transverse-field Ising |
| BP Avoidance | 50 | Examples showing BP issues |
| Hardware Efficient | 50 | HEA patterns |
| Symmetry Matching | 50 | Using problem symmetry |
| **Total** | 300 | |

### Creation Script

```python
import json

def create_physics_example(
    hamiltonian_type: str,
    num_qubits: int,
    reasoning: str,
    circuit_code: str
) -> dict:
    """Create a physics reasoning example."""
    return {
        "instruction": f"Design a variational ansatz for the {hamiltonian_type} ground state",
        "input": f"Number of qubits: {num_qubits}",
        "reasoning": reasoning,
        "output": circuit_code,
        "source": "custom_physics"
    }

# Example for XY chain
xy_example = create_physics_example(
    hamiltonian_type="4-qubit XY spin chain",
    num_qubits=4,
    reasoning="""The XY model conserves total magnetization. Key design principles:
1. Use RY rotations (real amplitudes, good for ground states)
2. Linear entanglement matches chain geometry
3. 2 layers is sufficient for 4 qubits to avoid barren plateaus
4. Place parameters before and after entanglement for expressivity""",
    circuit_code="""from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int = 4) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = []

    # Layer 1
    for i in range(num_qubits):
        p = Parameter(f'theta_{len(params)}')
        params.append(p)
        qc.ry(p, i)

    # Entanglement
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    # Layer 2
    for i in range(num_qubits):
        p = Parameter(f'theta_{len(params)}')
        params.append(p)
        qc.ry(p, i)

    return qc"""
)
```

---

## Data Preparation Pipeline

### Step 1: Download

```python
# scripts/download_data.py
from datasets import load_dataset
import os

os.makedirs("data/raw", exist_ok=True)

# QuantumLLMInstruct
ds = load_dataset("BoltzmannEntropy/QuantumLLMInstruct")
ds.save_to_disk("data/raw/quantum_llm_instruct")
```

### Step 2: Filter and Clean

```python
# scripts/prepare_data.py
from datasets import load_from_disk

ds = load_from_disk("data/raw/quantum_llm_instruct")

def clean_example(example):
    """Clean and standardize example."""
    # Remove markdown artifacts
    output = example["output"]
    output = output.replace("```python", "").replace("```", "")

    # Ensure imports
    if "from qiskit" not in output and "QuantumCircuit" in output:
        output = "from qiskit import QuantumCircuit\n" + output

    return {**example, "output": output.strip()}

def filter_example(example):
    """Filter for relevant examples."""
    output = example["output"]

    # Must have Qiskit code
    if "QuantumCircuit" not in output:
        return False

    # Must be syntactically valid
    try:
        compile(output, "<string>", "exec")
    except SyntaxError:
        return False

    return True

cleaned = ds["train"].map(clean_example)
filtered = cleaned.filter(filter_example)

print(f"Original: {len(ds['train'])}")
print(f"Filtered: {len(filtered)}")

filtered.save_to_disk("data/processed/quantum_filtered")
```

### Step 3: Format for Training

```python
# scripts/format_for_training.py
from datasets import load_from_disk

ds = load_from_disk("data/processed/quantum_filtered")

def format_for_sft(example):
    """Format for supervised fine-tuning."""
    prompt = f"""<|im_start|>system
You are a quantum computing expert. Generate Qiskit code for quantum circuits.
<|im_end|>
<|im_start|>user
{example['instruction']}

{example['input']}
<|im_end|>
<|im_start|>assistant
{example['output']}
<|im_end|>"""

    return {"text": prompt}

formatted = ds.map(format_for_sft)
formatted.save_to_disk("data/processed/quantum_sft_ready")
```

### Step 4: Create Train/Val Split

```python
from datasets import load_from_disk

ds = load_from_disk("data/processed/quantum_sft_ready")

# 90/10 split
split = ds.train_test_split(test_size=0.1, seed=42)
split["train"].save_to_disk("data/processed/train")
split["test"].save_to_disk("data/processed/val")

print(f"Train: {len(split['train'])}")
print(f"Val: {len(split['test'])}")
```

---

## Data Quality Checks

### Validation Script

```python
# scripts/validate_data.py
from datasets import load_from_disk
import random

ds = load_from_disk("data/processed/train")

# Sample check
samples = random.sample(range(len(ds)), 10)
for i in samples:
    example = ds[i]
    print(f"\n=== Example {i} ===")
    print(f"Length: {len(example['text'])}")

    # Check for required elements
    assert "<|im_start|>system" in example["text"]
    assert "<|im_start|>user" in example["text"]
    assert "<|im_start|>assistant" in example["text"]
    assert "QuantumCircuit" in example["text"]

    print("OK")

print("\nAll validation checks passed!")
```

---

## Data Statistics

After processing, log these statistics:

```yaml
# data/processed/stats.yaml
total_examples: xxx
train_examples: xxx
val_examples: xxx
avg_input_length: xxx
avg_output_length: xxx
categories:
  circuit_generation: xxx
  ansatz_design: xxx
  vqe: xxx
  other: xxx
unique_hamiltonians: xxx
qubit_range: [2, 20]
```

---

## Augmentation (Optional)

### Paraphrase Instructions

```python
def augment_instruction(instruction):
    """Create variations of instructions."""
    templates = [
        "Create a quantum circuit that {task}",
        "Generate Qiskit code to {task}",
        "Implement a variational ansatz for {task}",
        "Design a parameterized circuit that {task}",
    ]
    # Extract task from original instruction
    # Apply template
    ...
```

### Vary Qubit Counts

```python
def augment_qubit_count(example, new_count):
    """Adapt example to different qubit count."""
    # Parse original code
    # Replace num_qubits
    # Regenerate if needed
    ...
```

---

## Dataset Card

Create `data/processed/README.md`:

```markdown
# QuantumMind Training Dataset

## Description
Fine-tuning dataset for quantum circuit generation.

## Sources
- QuantumLLMInstruct (filtered)
- IBM Qiskit tutorials
- Custom physics examples

## Statistics
- Total: ~50,000 examples
- Train: ~45,000
- Val: ~5,000

## Format
Qwen chat format with system/user/assistant turns.

## License
Research use only.
```

---

## Summary

| Step | Input | Output |
|------|-------|--------|
| Download | HuggingFace | `data/raw/` |
| Filter | Raw | `data/processed/quantum_filtered/` |
| Format | Filtered | `data/processed/quantum_sft_ready/` |
| Split | Formatted | `data/processed/train/`, `data/processed/val/` |

Follow this pipeline exactly. Verify each step before proceeding.
