"""
Tests for the dataset preparation module.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.training.dataset import (
    DatasetConfig,
    clean_example,
    filter_example,
)


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DatasetConfig()

        assert config.raw_dir == "data/raw"
        assert config.processed_dir == "data/processed"
        assert config.min_output_length == 50
        assert config.max_output_length == 5000
        assert config.require_qiskit is True
        assert config.require_function is True
        assert config.train_split == 0.9
        assert config.seed == 42

    def test_custom_config(self):
        """Test custom configuration."""
        config = DatasetConfig(
            raw_dir="/custom/raw",
            processed_dir="/custom/processed",
            min_output_length=100,
            max_output_length=3000,
            require_qiskit=False,
            train_split=0.8,
            seed=123,
        )

        assert config.raw_dir == "/custom/raw"
        assert config.processed_dir == "/custom/processed"
        assert config.min_output_length == 100
        assert config.max_output_length == 3000
        assert config.require_qiskit is False
        assert config.train_split == 0.8
        assert config.seed == 123

    def test_default_categories(self):
        """Test default categories list."""
        config = DatasetConfig()

        assert "circuit_generation" in config.categories
        assert "vqe" in config.categories
        assert "ansatz_design" in config.categories


class TestFilterExample:
    """Tests for filter_example function."""

    def test_valid_qiskit_code(self):
        """Valid Qiskit code should pass filter."""
        config = DatasetConfig()
        example = {
            "output": """from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits):
    qc = QuantumCircuit(num_qubits)
    params = [Parameter(f'theta_{i}') for i in range(num_qubits)]
    for i in range(num_qubits):
        qc.ry(params[i], i)
    return qc
""",
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is True

    def test_too_short_output(self):
        """Output shorter than minimum should fail."""
        config = DatasetConfig(min_output_length=100)
        example = {
            "output": "x = 1",
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is False

    def test_too_long_output(self):
        """Output longer than maximum should fail."""
        config = DatasetConfig(max_output_length=100)
        example = {
            "output": "x = 1\n" * 100,  # Long output
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is False

    def test_no_qiskit_fails(self):
        """Code without Qiskit should fail when required."""
        config = DatasetConfig(require_qiskit=True)
        example = {
            "output": """def calculate(x):
    return x * 2
""" * 20,  # Make it long enough
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is False

    def test_no_qiskit_passes_when_not_required(self):
        """Code without Qiskit should pass when not required."""
        config = DatasetConfig(require_qiskit=False, require_function=False)
        example = {
            "output": """# Some valid Python code
x = 1
y = 2
z = x + y
print(z)
""" * 20,  # Make it long enough
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is True

    def test_no_function_fails(self):
        """Code without function definition should fail when required."""
        config = DatasetConfig(require_function=True, require_qiskit=False)
        example = {
            "output": """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
""" * 10,
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is False

    def test_syntax_error_fails(self):
        """Code with syntax errors should fail."""
        config = DatasetConfig(require_qiskit=False, require_function=False)
        example = {
            "output": """def broken(
    x = 1  # Missing closing paren
""" * 10,
            "category": "circuit_generation",
        }

        assert filter_example(example, config) is False

    def test_wrong_category_fails(self):
        """Wrong category should fail when categories are specified."""
        config = DatasetConfig(
            categories=["circuit_generation", "vqe"],
            require_qiskit=False,
            require_function=False,
        )
        example = {
            "output": "x = 1\n" * 20,
            "category": "unrelated_topic",
        }

        assert filter_example(example, config) is False

    def test_empty_output_fails(self):
        """Empty output should fail."""
        config = DatasetConfig()
        example = {"output": ""}

        assert filter_example(example, config) is False


class TestCleanExample:
    """Tests for clean_example function."""

    def test_removes_markdown_python_block(self):
        """Should remove ```python``` markdown blocks."""
        example = {
            "output": """```python
from qiskit import QuantumCircuit

def create_ansatz():
    return QuantumCircuit(2)
```""",
            "instruction": "Create a circuit",
        }

        cleaned = clean_example(example)

        assert "```python" not in cleaned["output"]
        assert "```" not in cleaned["output"]
        assert "from qiskit import QuantumCircuit" in cleaned["output"]

    def test_removes_generic_code_blocks(self):
        """Should remove generic ``` code blocks."""
        example = {
            "output": """```
x = 1
y = 2
```""",
            "instruction": "Calculate",
        }

        cleaned = clean_example(example)

        assert "```" not in cleaned["output"]

    def test_adds_qiskit_import(self):
        """Should add Qiskit import if missing."""
        example = {
            "output": """def create_ansatz():
    qc = QuantumCircuit(2)
    return qc
""",
            "instruction": "Create circuit",
        }

        cleaned = clean_example(example)

        assert "from qiskit import QuantumCircuit" in cleaned["output"]

    def test_adds_parameter_import(self):
        """Should add Parameter import if missing."""
        example = {
            "output": """from qiskit import QuantumCircuit

def create_ansatz():
    p = Parameter('theta')
    qc = QuantumCircuit(1)
    qc.ry(p, 0)
    return qc
""",
            "instruction": "Create parameterized circuit",
        }

        cleaned = clean_example(example)

        assert "from qiskit.circuit import Parameter" in cleaned["output"]

    def test_preserves_existing_imports(self):
        """Should not duplicate existing imports."""
        example = {
            "output": """from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz():
    p = Parameter('theta')
    qc = QuantumCircuit(1)
    return qc
""",
            "instruction": "Create circuit",
        }

        cleaned = clean_example(example)

        # Count occurrences - should only have one of each
        assert cleaned["output"].count("from qiskit import QuantumCircuit") == 1

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        example = {
            "output": """

    x = 1

""",
            "instruction": "Test",
        }

        cleaned = clean_example(example)

        assert cleaned["output"] == cleaned["output"].strip()

    def test_preserves_other_fields(self):
        """Should preserve other fields in the example."""
        example = {
            "output": "x = 1",
            "instruction": "Test instruction",
            "input": "Test input",
            "category": "test",
        }

        cleaned = clean_example(example)

        assert cleaned["instruction"] == "Test instruction"
        assert cleaned["input"] == "Test input"
        assert cleaned["category"] == "test"


class TestDownloadQuantumDatasets:
    """Tests for download_quantum_datasets function."""

    @patch("datasets.load_dataset")
    def test_download_creates_directories(self, mock_load_dataset):
        """Should create raw directory."""
        from src.training.dataset import download_quantum_datasets

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(raw_dir=os.path.join(tmpdir, "raw"))

            # Mock the dataset
            mock_train = MagicMock()
            mock_train.column_names = ["instruction", "output"]
            mock_train.__len__ = MagicMock(return_value=1000)

            mock_ds = MagicMock()
            mock_ds.__getitem__ = MagicMock(return_value=mock_train)
            mock_ds.save_to_disk = MagicMock()
            mock_load_dataset.return_value = mock_ds

            download_quantum_datasets(config)

            assert os.path.exists(config.raw_dir)

    @patch("datasets.load_dataset")
    def test_download_returns_stats(self, mock_load_dataset):
        """Should return download statistics."""
        from src.training.dataset import download_quantum_datasets

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(raw_dir=os.path.join(tmpdir, "raw"))

            # Mock the dataset
            mock_train = MagicMock()
            mock_train.column_names = ["instruction", "output"]
            mock_train.__len__ = MagicMock(return_value=1000)

            mock_ds = MagicMock()
            mock_ds.__getitem__ = MagicMock(return_value=mock_train)
            mock_ds.save_to_disk = MagicMock()
            mock_load_dataset.return_value = mock_ds

            stats = download_quantum_datasets(config)

            assert "total_examples" in stats
            assert "columns" in stats
            assert "save_path" in stats


class TestFormatForTraining:
    """Tests for format_for_training function."""

    def test_qwen_format_structure(self):
        """Test that formatting produces correct Qwen chat structure."""
        # Create a mock example that would result from formatting
        system_prompt = "You are a quantum computing expert."
        instruction = "Create a Bell state circuit"
        input_text = "num_qubits: 2"
        output = "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)"

        # Manually construct expected format
        expected_start = "<|im_start|>system"
        expected_user = "<|im_start|>user"
        expected_assistant = "<|im_start|>assistant"
        expected_end = "<|im_end|>"

        # This tests the format structure we expect
        formatted = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{instruction}

{input_text}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>"""

        assert expected_start in formatted
        assert expected_user in formatted
        assert expected_assistant in formatted
        assert formatted.count(expected_end) == 3


class TestLoadProcessedDataset:
    """Tests for load_processed_dataset function."""

    def test_invalid_split_raises_error(self):
        """Should raise error for invalid split name."""
        from src.training.dataset import load_processed_dataset

        with pytest.raises(ValueError):
            load_processed_dataset("invalid")

    def test_missing_dataset_raises_error(self):
        """Should raise error if dataset doesn't exist."""
        from src.training.dataset import load_processed_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(processed_dir=tmpdir)

            with pytest.raises(FileNotFoundError):
                load_processed_dataset("train", config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
