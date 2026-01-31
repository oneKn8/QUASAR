# QuantumMind Guideline: Phase 3 - LLM Fine-Tuning

> Complete Steps 3.4-3.5. Steps 3.1-3.3 already done.

---

## Step 3.4: Fine-Tune Model

**Status**: PENDING

**Prerequisites**:
- GPU access (A100 40GB+ recommended)
- Dataset prepared (Steps 3.1-3.3 DONE)
- ~$25-40 compute budget

---

### Action 1: Download Base Model

```python
# src/training/download_model.py
from unsloth import FastLanguageModel

def download_base_model():
    """Download and prepare Qwen2.5-Coder-7B."""

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    print(f"Model loaded: {model.config.name_or_path}")
    print(f"Parameters: {model.num_parameters():,}")
    print(f"Max sequence length: 2048")
    print(f"Quantization: 4-bit")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = download_base_model()
    print("Download complete!")
```

---

### Action 2: Add LoRA Adapters

```python
# src/training/prepare_lora.py
from unsloth import FastLanguageModel

def add_lora_adapters(model):
    """Add LoRA adapters for fine-tuning."""

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

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable %: {100 * trainable / total:.2f}%")

    return model
```

---

### Action 3: Load Dataset

```python
# src/training/load_dataset.py
from datasets import load_from_disk

def load_training_data():
    """Load prepared training and validation datasets."""

    train_ds = load_from_disk("data/processed/train")
    val_ds = load_from_disk("data/processed/val")

    print(f"Training examples: {len(train_ds):,}")
    print(f"Validation examples: {len(val_ds):,}")

    # Show sample
    print(f"\nSample training example:")
    print(train_ds[0]["text"][:500] + "...")

    return train_ds, val_ds
```

---

### Action 4: Configure and Run Training

```python
# src/training/finetune.py
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from datasets import load_from_disk
import yaml
from datetime import datetime
import torch

def run_finetuning():
    """Complete fine-tuning pipeline."""

    # 1. Load model
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # 2. Add LoRA
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
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

    # 3. Load data
    print("Loading datasets...")
    train_ds = load_from_disk("data/processed/train")
    val_ds = load_from_disk("data/processed/val")

    # 4. Configure training
    output_dir = "./models/checkpoints/quantum-mind-v1"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch size: 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=3,
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=500,
        seed=42,
        dataloader_num_workers=4,
        report_to="none",  # or "wandb" if using W&B
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        max_seq_length=2048,
        packing=True,
    )

    # 5. Train
    print("Starting training...")
    start_time = datetime.now()
    train_result = trainer.train()
    end_time = datetime.now()

    # 6. Save model
    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 7. Log metrics
    training_time = (end_time - start_time).total_seconds() / 3600

    log = {
        "version": "v1",
        "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "training_data": "data/processed/train",
        "num_train_examples": len(train_ds),
        "num_val_examples": len(val_ds),
        "epochs": 3,
        "final_train_loss": train_result.training_loss,
        "training_time_hours": round(training_time, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
        "date": datetime.now().isoformat(),
        "lora_r": 32,
        "lora_alpha": 64,
        "learning_rate": 2e-4,
        "batch_size_effective": 16,
    }

    with open(f"{output_dir}/training_log.yaml", "w") as f:
        yaml.dump(log, f, default_flow_style=False)

    print(f"\nTraining complete!")
    print(f"  Final loss: {train_result.training_loss:.4f}")
    print(f"  Time: {training_time:.2f} hours")
    print(f"  Model saved to: {output_dir}")

    return train_result


if __name__ == "__main__":
    run_finetuning()
```

---

### Action 5: Verify Training Success

```python
# src/training/verify_training.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml

def verify_training():
    """Verify fine-tuned model loads and generates."""

    model_path = "./models/checkpoints/quantum-mind-v1"

    # Load training log
    with open(f"{model_path}/training_log.yaml") as f:
        log = yaml.safe_load(f)

    print("Training log:")
    for k, v in log.items():
        print(f"  {k}: {v}")

    # Check loss
    if log.get("final_train_loss", 1.0) >= 0.5:
        print("\nWARNING: Training loss >= 0.5, model may be undertrained")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Test generation
    print("\nTesting generation...")
    prompt = """<|im_start|>system
You are a quantum computing expert that generates Qiskit circuits.
<|im_end|>
<|im_start|>user
Create a 4-qubit variational circuit for the XY chain Hamiltonian ground state.
<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated response:\n{response}")

    # Check for code
    if "QuantumCircuit" in response or "qc" in response:
        print("\nModel generates quantum circuit code!")
        return True
    else:
        print("\nWARNING: Response may not contain circuit code")
        return False


if __name__ == "__main__":
    verify_training()
```

---

## Step 3.4 Verification Checklist

- [ ] Model downloaded successfully (Qwen2.5-Coder-7B-Instruct)
- [ ] LoRA adapters added (r=32, alpha=64)
- [ ] Training completes without OOM errors
- [ ] Model saved to `models/checkpoints/quantum-mind-v1/`
- [ ] Training loss < 0.5
- [ ] `training_log.yaml` saved with all metrics
- [ ] Model loads and generates text

**DO NOT PROCEED TO STEP 3.5 UNTIL ALL 7 CHECKS PASS**

---

## Step 3.5: Validate Fine-Tuned Model

**Status**: PENDING

**Prerequisites**: Step 3.4 complete

---

### Action 1: Create Validation Test Suite

```python
# src/training/validate_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from typing import List, Tuple
from src.quantum.verifier import verify_circuit_code
from src.quantum.barren_plateau import detect_barren_plateau
from src.quantum.hamiltonians import xy_chain, heisenberg_chain, tfim_chain

MODEL_PATH = "./models/checkpoints/quantum-mind-v1"
NUM_TESTS = 100

def get_test_prompts() -> List[Tuple[str, dict]]:
    """Generate 100 diverse test prompts."""

    prompts = []

    hamiltonians = [
        ("XY chain", "xy_chain"),
        ("Heisenberg chain", "heisenberg"),
        ("Transverse-field Ising model", "tfim"),
    ]

    qubit_counts = [4, 6, 8]

    constraints = [
        "Use only RY and CX gates",
        "Minimize circuit depth",
        "Use a hardware-efficient structure",
        "Include entangling layers",
        "",  # No constraint
    ]

    for ham_name, ham_type in hamiltonians:
        for n_qubits in qubit_counts:
            for constraint in constraints:
                prompt = f"""<|im_start|>system
You are a quantum computing expert that generates Qiskit circuits.
<|im_end|>
<|im_start|>user
Create a {n_qubits}-qubit variational circuit for the {ham_name} Hamiltonian ground state. {constraint}
<|im_end|>
<|im_start|>assistant
"""
                prompts.append((prompt, {
                    "hamiltonian": ham_type,
                    "num_qubits": n_qubits
                }))

    # Add more until we have 100
    while len(prompts) < NUM_TESTS:
        prompts.append(prompts[len(prompts) % len(prompts[:45])])

    return prompts[:NUM_TESTS]


def extract_code(response: str) -> str:
    """Extract Python code from model response."""

    # Try to find code block
    patterns = [
        r"```python\n(.*?)```",
        r"```\n(.*?)```",
        r"(from qiskit.*?)(?:\n\n|\Z)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Return everything after "assistant" marker if no code block
    if "<|im_start|>assistant" in response:
        return response.split("<|im_start|>assistant")[-1].strip()

    return response


def validate_model():
    """Run full validation suite."""

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompts = get_test_prompts()
    print(f"Running {len(prompts)} validation tests...")

    results = {
        "syntax_valid": 0,
        "verification_pass": 0,
        "bp_free": 0,
        "total": len(prompts)
    }

    for i, (prompt, meta) in enumerate(prompts):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(prompts)}")

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = extract_code(response)

        # Syntax check
        try:
            compile(code, "<string>", "exec")
            results["syntax_valid"] += 1
        except SyntaxError:
            continue

        # Verification check
        try:
            ver_result = verify_circuit_code(
                code,
                expected_qubits=meta["num_qubits"],
                max_depth=50
            )
            if ver_result.is_valid:
                results["verification_pass"] += 1

                # BP check
                ham_fn = {
                    "xy_chain": xy_chain,
                    "heisenberg": heisenberg_chain,
                    "tfim": tfim_chain
                }[meta["hamiltonian"]]

                ham = ham_fn(meta["num_qubits"]).operator
                bp_result = detect_barren_plateau(
                    ver_result.circuit,
                    ham,
                    num_samples=30
                )

                if not bp_result.has_barren_plateau:
                    results["bp_free"] += 1
        except Exception:
            continue

    # Calculate rates
    syntax_rate = results["syntax_valid"] / results["total"] * 100
    verify_rate = results["verification_pass"] / results["total"] * 100
    bp_rate = results["bp_free"] / results["total"] * 100

    print(f"\n{'='*50}")
    print("VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Syntax validity: {syntax_rate:.1f}% (target: >95%)")
    print(f"Verification pass: {verify_rate:.1f}% (target: >80%)")
    print(f"BP-free rate: {bp_rate:.1f}% (target: >60%)")
    print(f"{'='*50}")

    passed = syntax_rate > 95 and verify_rate > 80 and bp_rate > 60

    if passed:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED - targets not met")

    # Save results
    validation_log = {
        "syntax_rate": syntax_rate,
        "verification_rate": verify_rate,
        "bp_free_rate": bp_rate,
        "num_tests": results["total"],
        "passed": passed
    }

    import yaml
    with open(f"{MODEL_PATH}/validation_log.yaml", "w") as f:
        yaml.dump(validation_log, f)

    return passed


if __name__ == "__main__":
    success = validate_model()
    exit(0 if success else 1)
```

---

### Action 2: Compare to Base Model

```python
# src/training/compare_base_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.training.validate_model import get_test_prompts, extract_code
from src.quantum.verifier import verify_circuit_code

def compare_to_base():
    """Compare fine-tuned model to base model."""

    base_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
    finetuned_path = "./models/checkpoints/quantum-mind-v1"

    # Only run 20 tests for comparison
    prompts = get_test_prompts()[:20]

    results = {"base": {"syntax": 0, "verify": 0}, "finetuned": {"syntax": 0, "verify": 0}}

    for model_name, model_path in [("base", base_path), ("finetuned", finetuned_path)]:
        print(f"\nEvaluating {model_name} model...")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        for prompt, meta in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            code = extract_code(response)

            try:
                compile(code, "<string>", "exec")
                results[model_name]["syntax"] += 1

                ver = verify_circuit_code(code, meta["num_qubits"], 50)
                if ver.is_valid:
                    results[model_name]["verify"] += 1
            except:
                pass

        # Free memory
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print("COMPARISON RESULTS")
    print(f"{'='*50}")
    print(f"{'Metric':<20} {'Base':>10} {'Fine-tuned':>12}")
    print(f"{'Syntax valid':<20} {results['base']['syntax']:>10} {results['finetuned']['syntax']:>12}")
    print(f"{'Verification pass':<20} {results['base']['verify']:>10} {results['finetuned']['verify']:>12}")

    improvement = results['finetuned']['verify'] - results['base']['verify']
    print(f"\nImprovement: +{improvement} verified circuits")

    return results['finetuned']['verify'] > results['base']['verify']


if __name__ == "__main__":
    success = compare_to_base()
    print("Fine-tuned model is better!" if success else "No improvement detected")
```

---

## Step 3.5 Verification Checklist

- [ ] `validate_model.py` runs on 100 test prompts
- [ ] Syntax validity rate > 95%
- [ ] Verification pass rate > 80%
- [ ] BP-free rate > 60%
- [ ] `validation_log.yaml` saved
- [ ] Fine-tuned model beats base model on comparison
- [ ] All tests documented

**DO NOT PROCEED TO PHASE 5 UNTIL ALL 7 CHECKS PASS**

---

## After Completion

Update `00_OVERVIEW.md`:
- Change Phase 3 progress to 5/5
- Mark Steps 3.4 and 3.5 as DONE

Next: Proceed to `05_EVALUATION.md` (Step 5.1)

---

*Phase 3 provides the trained LLM that powers circuit discovery.*
