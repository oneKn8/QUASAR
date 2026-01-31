# Model Selection and Configuration

## Primary Model: Qwen2.5-Coder-7B-Instruct

### Why This Model

| Factor | Qwen2.5-Coder-7B | Alternatives | Decision |
|--------|------------------|--------------|----------|
| Code generation | State-of-the-art | Llama, CodeLlama | Qwen leads on benchmarks |
| Size | 7B (fits on single GPU) | 3B too weak, 14B too large | 7B is sweet spot |
| Fine-tuning support | Excellent with Unsloth | Good | Qwen + Unsloth is proven |
| Context length | 128K tokens | Varies | Plenty for our needs |
| Quantum code | No specific training | Same | We fine-tune anyway |

### Model Details

```yaml
Model: Qwen/Qwen2.5-Coder-7B-Instruct
Parameters: 7.61 billion
Architecture: Transformer (decoder-only)
Context Length: 128,000 tokens
Training Data: 5.5 trillion tokens (code + text)
Languages: 92 programming languages supported
Quantization: 4-bit compatible
```

### Download

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

---

## Fine-Tuning Configuration

### Method: QLoRA (Quantized Low-Rank Adaptation)

**Why QLoRA**:
- 4-bit base model (75% memory reduction)
- LoRA adapters (train only ~1% of params)
- Proven for code generation tasks
- Fits on single A100/H100

### Unsloth Configuration

```python
# Fine-tuning script
from unsloth import FastLanguageModel
import torch

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

### Training Hyperparameters

```python
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    # Output
    output_dir="./models/checkpoints",

    # Batch size
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 16

    # Learning rate
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # Training duration
    num_train_epochs=3,
    max_steps=-1,  # Use epochs

    # Precision
    bf16=True,  # Use bfloat16
    fp16=False,

    # Optimization
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Logging
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,

    # Evaluation
    eval_strategy="steps",
    eval_steps=500,

    # Misc
    seed=42,
    dataloader_num_workers=4,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    max_seq_length=2048,
    packing=True,  # Pack short sequences
)
```

### Expected Resources

| Resource | Requirement |
|----------|-------------|
| GPU | A100 40GB or H100 |
| VRAM Usage | ~20GB with QLoRA |
| Training Time | 2-4 hours (10k examples) |
| Cost (Brev) | ~$25-40 |

---

## Prompt Template

### System Prompt

```
You are a quantum computing expert specializing in variational quantum algorithms. Your task is to generate efficient, trainable quantum circuits in Qiskit.

Key principles:
1. Always use Parameters for trainable values
2. Prefer shallow circuits over deep ones
3. Use local rotations followed by entanglement
4. Avoid patterns that cause barren plateaus
5. Match circuit structure to problem symmetry
```

### User Prompt Template

```
TASK: Generate a variational ansatz for the following quantum physics problem.

PROBLEM: {problem_description}

HAMILTONIAN: {hamiltonian_description}

SPECIFICATIONS:
- Number of qubits: {num_qubits}
- Maximum depth: {max_depth}
- Allowed gates: {allowed_gates}

CONTEXT:
{memory_context}

REQUIREMENTS:
1. Define function: create_ansatz(num_qubits: int) -> QuantumCircuit
2. Use qiskit.circuit.Parameter for all trainable parameters
3. Return a QuantumCircuit object
4. Include necessary imports

Generate only the Python code.
```

### Response Format

```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int) -> QuantumCircuit:
    """
    Variational ansatz for {problem}.

    Architecture:
    - {description of layers}

    Parameters: {count}
    Depth: {depth}
    """
    qc = QuantumCircuit(num_qubits)
    # ... circuit construction
    return qc
```

---

## Inference Configuration

### For Discovery Loop

```python
# config.py

INFERENCE_CONFIG = {
    # Generation parameters
    "max_new_tokens": 1024,
    "temperature": 0.7,  # Some creativity
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
    "repetition_penalty": 1.1,

    # Stopping
    "stop_sequences": ["```\n", "\n\n\n"],

    # Device
    "device": "cuda",
    "torch_dtype": "float16",
}
```

### For Batch Generation

```python
# Generate multiple candidates per iteration
BATCH_CONFIG = {
    "num_return_sequences": 3,  # Generate 3 circuits per call
    "temperature": [0.5, 0.7, 0.9],  # Varying creativity
}
```

---

## Model Checkpoints

### Save Strategy

```python
# Save adapter weights only (small files)
model.save_pretrained("./models/checkpoints/quantum-mind-v1")
tokenizer.save_pretrained("./models/checkpoints/quantum-mind-v1")
```

### Checkpoint Naming

```
models/
└── checkpoints/
    ├── quantum-mind-v1/      # Initial fine-tuning
    ├── quantum-mind-v2/      # After more data
    └── quantum-mind-final/   # Best performing
```

### Version Tracking

```yaml
# models/checkpoints/quantum-mind-v1/config.yaml
version: v1
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
training_data: QuantumLLMInstruct-filtered
num_examples: 50000
epochs: 3
final_loss: 0.xxx
eval_accuracy: 0.xxx
date: 2026-xx-xx
```

---

## Evaluation Metrics for LLM

### During Training

| Metric | Target |
|--------|--------|
| Training Loss | < 0.5 |
| Eval Loss | < 0.6 |
| Perplexity | < 5.0 |

### Post-Training Validation

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Syntax Validity | > 95% | % of generated code that executes |
| Verification Pass | > 80% | % passing circuit verifier |
| BP-Free | > 60% | % without barren plateaus |
| Novel Patterns | > 20% | % different from training data |

### Validation Script

```python
def evaluate_model(model, tokenizer, test_prompts, verifier):
    """Evaluate fine-tuned model quality."""
    results = {
        "syntax_valid": 0,
        "verification_pass": 0,
        "bp_free": 0,
        "total": len(test_prompts)
    }

    for prompt in test_prompts:
        code = generate(model, tokenizer, prompt)

        # Syntax check
        try:
            exec(code)
            results["syntax_valid"] += 1
        except:
            continue

        # Verification
        ver_result = verifier.verify(code)
        if ver_result.is_valid:
            results["verification_pass"] += 1

        # BP check
        if ver_result.circuit:
            bp_result = detect_barren_plateau(ver_result.circuit)
            if not bp_result.has_barren_plateau:
                results["bp_free"] += 1

    return {
        k: v / results["total"] for k, v in results.items() if k != "total"
    }
```

---

## Alternative Models (Backup)

If Qwen doesn't work well:

| Model | Pros | Cons |
|-------|------|------|
| DeepSeek-Coder-6.7B | Good at code | Less documentation |
| CodeLlama-7B | Meta backing | Older architecture |
| Llama-3.2-8B | Latest Llama | Not code-specific |
| Phi-4 (14B) | Excellent reasoning | Larger, Microsoft license |

---

## GRPO Fine-Tuning (Optional Enhancement)

After SFT, optionally add GRPO for reasoning:

```python
# GRPO configuration
from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    output_dir="./models/checkpoints/quantum-mind-grpo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    reward_model=None,  # Use verifiable rewards
)

# Reward function: Does the circuit achieve low energy?
def reward_fn(generated_code, goal):
    try:
        circuit = execute_code(generated_code)
        energy = run_vqe(circuit, goal.hamiltonian)
        error = abs(energy - goal.exact_energy)
        # Reward: Higher for lower error
        return max(0, 1 - error * 10)
    except:
        return 0  # Invalid code gets zero reward
```

This is optional but could significantly improve circuit quality.

---

## Summary

```yaml
Primary Model: Qwen2.5-Coder-7B-Instruct
Method: QLoRA via Unsloth
LoRA Rank: 32
Learning Rate: 2e-4
Batch Size: 16 (effective)
Epochs: 3
Expected Training Time: 2-4 hours
Expected Cost: $25-40
```

Follow these configurations exactly. Do not experiment with hyperparameters until baseline is established.
