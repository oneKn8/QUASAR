"""
Dataset preparation module for QuantumMind.

This module handles:
1. Downloading QuantumLLMInstruct dataset from HuggingFace
2. Filtering for circuit generation examples
3. Cleaning and validating data
4. Formatting for Qwen fine-tuning

Usage:
    python -m src.training.dataset --download
    python -m src.training.dataset --prepare
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tqdm import tqdm


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    min_output_length: int = 50
    max_output_length: int = 5000
    require_qiskit: bool = True
    require_function: bool = True
    train_split: float = 0.9
    seed: int = 42
    categories: list[str] = field(
        default_factory=lambda: [
            "circuit_generation",
            "ansatz_design",
            "vqe",
            "qaoa",
            "hamiltonian",
            "quantum_algorithm",
        ]
    )


def download_quantum_datasets(config: DatasetConfig | None = None) -> dict:
    """
    Download QuantumLLMInstruct dataset from HuggingFace.

    Args:
        config: Dataset configuration

    Returns:
        Dictionary with download statistics
    """
    from datasets import load_dataset

    config = config or DatasetConfig()
    os.makedirs(config.raw_dir, exist_ok=True)

    print("Downloading QuantumLLMInstruct dataset...")
    print("This may take a few minutes...")

    try:
        dataset = load_dataset("BoltzmannEntropy/QuantumLLMInstruct")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure you have internet access and the datasets library installed.")
        raise

    # Save to disk
    save_path = os.path.join(config.raw_dir, "quantum_llm_instruct")
    dataset.save_to_disk(save_path)

    stats = {
        "total_examples": len(dataset["train"]),
        "columns": dataset["train"].column_names,
        "save_path": save_path,
    }

    print(f"Downloaded {stats['total_examples']} examples")
    print(f"Saved to: {save_path}")

    return stats


def filter_example(example: dict, config: DatasetConfig) -> bool:
    """
    Filter a single example based on configuration.

    Args:
        example: Dataset example
        config: Filter configuration

    Returns:
        True if example should be kept
    """
    output = example.get("output", "")

    # Length checks
    if len(output) < config.min_output_length:
        return False
    if len(output) > config.max_output_length:
        return False

    # Content checks
    if config.require_qiskit:
        if "QuantumCircuit" not in output and "from qiskit" not in output.lower():
            return False

    if config.require_function:
        if "def " not in output:
            return False

    # Category check (if available)
    if "category" in example:
        category = example["category"]
        if config.categories and category not in config.categories:
            return False

    # Syntax validation
    try:
        compile(output, "<string>", "exec")
    except SyntaxError:
        return False

    return True


def clean_example(example: dict) -> dict:
    """
    Clean and standardize a single example.

    Args:
        example: Raw example

    Returns:
        Cleaned example
    """
    output = example.get("output", "")

    # Remove markdown code blocks
    if "```python" in output:
        start = output.find("```python")
        end = output.find("```", start + 9)
        if start != -1 and end != -1:
            output = output[start + 9 : end].strip()
    elif "```" in output:
        output = output.replace("```", "").strip()

    # Ensure imports
    if "QuantumCircuit" in output and "from qiskit" not in output.lower():
        output = "from qiskit import QuantumCircuit\n" + output

    if "Parameter" in output and "from qiskit.circuit import Parameter" not in output:
        if "from qiskit import" in output:
            # Add to existing import
            output = output.replace(
                "from qiskit import QuantumCircuit",
                "from qiskit import QuantumCircuit\nfrom qiskit.circuit import Parameter",
            )
        else:
            output = "from qiskit.circuit import Parameter\n" + output

    return {
        **example,
        "output": output.strip(),
    }


def filter_dataset(
    config: DatasetConfig | None = None, verbose: bool = True
) -> dict:
    """
    Filter the raw dataset based on configuration.

    Args:
        config: Filter configuration
        verbose: Print progress

    Returns:
        Dictionary with filter statistics
    """
    from datasets import load_from_disk

    config = config or DatasetConfig()

    raw_path = os.path.join(config.raw_dir, "quantum_llm_instruct")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. Run download first."
        )

    if verbose:
        print(f"Loading raw dataset from {raw_path}...")

    dataset = load_from_disk(raw_path)
    original_size = len(dataset["train"])

    if verbose:
        print(f"Original size: {original_size}")
        print("Filtering and cleaning...")

    # Filter
    filtered = dataset["train"].filter(
        lambda x: filter_example(x, config),
        desc="Filtering" if verbose else None,
    )

    # Clean
    cleaned = filtered.map(
        clean_example,
        desc="Cleaning" if verbose else None,
    )

    filtered_size = len(cleaned)

    # Save
    os.makedirs(config.processed_dir, exist_ok=True)
    save_path = os.path.join(config.processed_dir, "quantum_filtered")
    cleaned.save_to_disk(save_path)

    stats = {
        "original_size": original_size,
        "filtered_size": filtered_size,
        "retention_rate": filtered_size / original_size,
        "save_path": save_path,
    }

    if verbose:
        print(f"Filtered size: {filtered_size} ({stats['retention_rate']:.1%} retained)")
        print(f"Saved to: {save_path}")

    return stats


def format_for_training(
    config: DatasetConfig | None = None,
    system_prompt: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Format filtered dataset for Qwen fine-tuning.

    Args:
        config: Dataset configuration
        system_prompt: Custom system prompt (uses default if None)
        verbose: Print progress

    Returns:
        Dictionary with format statistics
    """
    from datasets import load_from_disk

    config = config or DatasetConfig()

    if system_prompt is None:
        system_prompt = """You are a quantum computing expert specializing in variational quantum algorithms. Your task is to generate efficient, trainable quantum circuits in Qiskit.

Key principles:
1. Always use Parameters for trainable values
2. Prefer shallow circuits over deep ones
3. Use local rotations followed by entanglement
4. Avoid patterns that cause barren plateaus
5. Match circuit structure to problem symmetry"""

    filtered_path = os.path.join(config.processed_dir, "quantum_filtered")
    if not os.path.exists(filtered_path):
        raise FileNotFoundError(
            f"Filtered dataset not found at {filtered_path}. Run filter first."
        )

    if verbose:
        print(f"Loading filtered dataset from {filtered_path}...")

    dataset = load_from_disk(filtered_path)

    def format_example(example):
        """Format a single example for Qwen chat format."""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        # Combine instruction and input
        user_message = instruction
        if input_text:
            user_message += f"\n\n{input_text}"

        # Qwen chat format
        formatted = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>"""

        return {"text": formatted}

    if verbose:
        print("Formatting for training...")

    formatted = dataset.map(
        format_example,
        desc="Formatting" if verbose else None,
    )

    # Create train/val split
    split = formatted.train_test_split(
        test_size=1 - config.train_split,
        seed=config.seed,
    )

    # Save
    train_path = os.path.join(config.processed_dir, "train")
    val_path = os.path.join(config.processed_dir, "val")

    split["train"].save_to_disk(train_path)
    split["test"].save_to_disk(val_path)

    stats = {
        "train_size": len(split["train"]),
        "val_size": len(split["test"]),
        "train_path": train_path,
        "val_path": val_path,
        "system_prompt_length": len(system_prompt),
    }

    if verbose:
        print(f"Train size: {stats['train_size']}")
        print(f"Val size: {stats['val_size']}")
        print(f"Saved train to: {train_path}")
        print(f"Saved val to: {val_path}")

    # Save stats
    stats_path = os.path.join(config.processed_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def prepare_training_data(
    config: DatasetConfig | None = None, verbose: bool = True
) -> dict:
    """
    Run full data preparation pipeline.

    This downloads, filters, and formats the dataset.

    Args:
        config: Dataset configuration
        verbose: Print progress

    Returns:
        Dictionary with all statistics
    """
    config = config or DatasetConfig()

    # Check if already downloaded
    raw_path = os.path.join(config.raw_dir, "quantum_llm_instruct")
    if not os.path.exists(raw_path):
        if verbose:
            print("=" * 50)
            print("Step 1: Downloading dataset")
            print("=" * 50)
        download_stats = download_quantum_datasets(config)
    else:
        if verbose:
            print("Dataset already downloaded, skipping download.")
        download_stats = {"skipped": True}

    if verbose:
        print("\n" + "=" * 50)
        print("Step 2: Filtering dataset")
        print("=" * 50)
    filter_stats = filter_dataset(config, verbose)

    if verbose:
        print("\n" + "=" * 50)
        print("Step 3: Formatting for training")
        print("=" * 50)
    format_stats = format_for_training(config, verbose=verbose)

    return {
        "download": download_stats,
        "filter": filter_stats,
        "format": format_stats,
    }


def load_processed_dataset(split: str = "train", config: DatasetConfig | None = None):
    """
    Load processed dataset.

    Args:
        split: "train" or "val"
        config: Dataset configuration

    Returns:
        HuggingFace Dataset object
    """
    from datasets import load_from_disk

    config = config or DatasetConfig()

    if split not in ["train", "val"]:
        raise ValueError(f"split must be 'train' or 'val', got {split}")

    path = os.path.join(config.processed_dir, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            "Run prepare_training_data first."
        )

    return load_from_disk(path)


def validate_dataset(config: DatasetConfig | None = None) -> dict:
    """
    Validate processed dataset quality.

    Args:
        config: Dataset configuration

    Returns:
        Validation statistics
    """
    config = config or DatasetConfig()

    train_data = load_processed_dataset("train", config)
    val_data = load_processed_dataset("val", config)

    stats = {
        "train_size": len(train_data),
        "val_size": len(val_data),
        "sample_lengths": [],
        "has_required_tags": 0,
    }

    # Check samples
    for i in range(min(100, len(train_data))):
        example = train_data[i]
        text = example["text"]
        stats["sample_lengths"].append(len(text))

        # Check for required format tags
        has_tags = (
            "<|im_start|>system" in text
            and "<|im_start|>user" in text
            and "<|im_start|>assistant" in text
        )
        if has_tags:
            stats["has_required_tags"] += 1

    stats["avg_length"] = sum(stats["sample_lengths"]) / len(stats["sample_lengths"])
    stats["tag_compliance"] = stats["has_required_tags"] / min(100, len(train_data))

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset preparation for QuantumMind")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--filter", action="store_true", help="Filter dataset")
    parser.add_argument("--format", action="store_true", help="Format for training")
    parser.add_argument("--prepare", action="store_true", help="Run full pipeline")
    parser.add_argument("--validate", action="store_true", help="Validate dataset")

    args = parser.parse_args()

    config = DatasetConfig()

    if args.download:
        download_quantum_datasets(config)
    elif args.filter:
        filter_dataset(config)
    elif args.format:
        format_for_training(config)
    elif args.prepare:
        prepare_training_data(config)
    elif args.validate:
        stats = validate_dataset(config)
        print("\nValidation Results:")
        print(f"  Train size: {stats['train_size']}")
        print(f"  Val size: {stats['val_size']}")
        print(f"  Avg sample length: {stats['avg_length']:.0f}")
        print(f"  Tag compliance: {stats['tag_compliance']:.1%}")
    else:
        parser.print_help()
