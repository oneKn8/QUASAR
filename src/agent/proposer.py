"""
LLM-based circuit proposer for QuantumMind.

This module generates quantum circuit code using a language model,
either via local inference or API calls.
"""

import re
from dataclasses import dataclass, field
from typing import Callable

from qiskit import QuantumCircuit


@dataclass
class ProposerConfig:
    """Configuration for the circuit proposer."""

    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1
    use_local: bool = True
    api_key: str | None = None
    api_base: str | None = None
    device: str = "auto"
    load_in_4bit: bool = True
    seed: int | None = None


@dataclass
class CircuitProposal:
    """A proposed quantum circuit from the LLM."""

    code: str
    reasoning: str
    raw_response: str
    metadata: dict = field(default_factory=dict)


DEFAULT_SYSTEM_PROMPT = """You are a quantum computing expert specializing in variational quantum algorithms. Your task is to design efficient, trainable quantum circuits (ansatze) in Qiskit.

CRITICAL REQUIREMENTS:
1. Always define a function called `create_ansatz` that takes `num_qubits: int` as the first argument
2. The function MUST return a QuantumCircuit object
3. Use qiskit.circuit.Parameter for trainable parameters (name them theta_0, theta_1, etc.)
4. Import QuantumCircuit from qiskit and Parameter from qiskit.circuit

KEY DESIGN PRINCIPLES:
- Prefer shallow circuits (2-4 layers) over deep ones to avoid barren plateaus
- Use local rotations (RY, RZ) followed by entanglement (CX)
- Match entanglement pattern to problem geometry (linear for chains, all-to-all for full coupling)
- Ensure gradients are non-vanishing by keeping depth proportional to log(num_qubits)

EXAMPLE STRUCTURE:
```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = []

    # Layer 1: Local rotations
    for i in range(num_qubits):
        p = Parameter(f'theta_{len(params)}')
        params.append(p)
        qc.ry(p, i)

    # Entanglement
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    # Layer 2: More rotations
    for i in range(num_qubits):
        p = Parameter(f'theta_{len(params)}')
        params.append(p)
        qc.ry(p, i)

    return qc
```

Respond with:
1. REASONING: Brief explanation of your design choices
2. CODE: The complete Python code implementing create_ansatz"""


def build_circuit_prompt(
    goal_description: str,
    num_qubits: int,
    hamiltonian_type: str,
    constraints: dict | None = None,
    feedback: str | None = None,
) -> str:
    """
    Build a prompt for circuit generation.

    Args:
        goal_description: Description of the physics goal
        num_qubits: Number of qubits for the circuit
        hamiltonian_type: Type of Hamiltonian (e.g., "xy_chain", "heisenberg")
        constraints: Optional constraints (max_depth, allowed_gates, etc.)
        feedback: Optional feedback from previous attempts

    Returns:
        Formatted prompt string
    """
    prompt = f"""Design a variational quantum circuit (ansatz) for the following problem:

PHYSICS GOAL: {goal_description}

SPECIFICATIONS:
- Number of qubits: {num_qubits}
- Hamiltonian type: {hamiltonian_type}
"""

    if constraints:
        prompt += "\nCONSTRAINTS:\n"
        if "max_depth" in constraints:
            prompt += f"- Maximum circuit depth: {constraints['max_depth']}\n"
        if "max_params" in constraints:
            prompt += f"- Maximum parameters: {constraints['max_params']}\n"
        if "allowed_gates" in constraints:
            gates = ", ".join(constraints["allowed_gates"])
            prompt += f"- Allowed gates: {gates}\n"
        if "entanglement" in constraints:
            prompt += f"- Entanglement pattern: {constraints['entanglement']}\n"

    if feedback:
        prompt += f"""
FEEDBACK FROM PREVIOUS ATTEMPT:
{feedback}

Please address the issues mentioned above in your new design.
"""

    prompt += """
Provide your design with:
1. REASONING: Explain your design choices
2. CODE: Complete Python implementation of create_ansatz function
"""

    return prompt


def extract_code_from_response(response: str) -> str | None:
    """
    Extract Python code from LLM response.

    Args:
        response: Raw LLM response

    Returns:
        Extracted code or None if not found
    """
    # Try to find code block
    code_patterns = [
        r"```python\s*(.*?)```",
        r"```\s*(.*?)```",
        r"CODE:\s*(.*?)(?=\n\n|\Z)",
    ]

    for pattern in code_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            # Verify it looks like circuit code
            if "def create_ansatz" in code and "QuantumCircuit" in code:
                return code

    # Fallback: look for function definition directly
    if "def create_ansatz" in response:
        # Find the function and extract it
        lines = response.split("\n")
        code_lines = []
        in_function = False
        indent_level = 0

        for line in lines:
            if "def create_ansatz" in line:
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                code_lines.append(line)
            elif in_function:
                # Check if we're still in the function
                if line.strip() and not line.startswith(" " * (indent_level + 1)):
                    if not line.strip().startswith("#"):
                        break
                code_lines.append(line)

        if code_lines:
            # Add imports if missing
            code = "\n".join(code_lines)
            if "from qiskit import QuantumCircuit" not in code:
                code = "from qiskit import QuantumCircuit\n" + code
            if "Parameter" in code and "from qiskit.circuit import Parameter" not in code:
                code = "from qiskit.circuit import Parameter\n" + code
            return code

    return None


def extract_reasoning_from_response(response: str) -> str:
    """
    Extract reasoning/explanation from LLM response.

    Args:
        response: Raw LLM response

    Returns:
        Extracted reasoning
    """
    patterns = [
        r"REASONING:\s*(.*?)(?=CODE:|```|$)",
        r"Explanation:\s*(.*?)(?=CODE:|```|$)",
        r"Design choices:\s*(.*?)(?=CODE:|```|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Return first paragraph as fallback
    paragraphs = response.split("\n\n")
    if paragraphs:
        return paragraphs[0].strip()

    return "No reasoning provided"


class CircuitProposer:
    """
    LLM-based quantum circuit proposer.

    Uses a language model to generate variational quantum circuits
    based on physics goals and constraints.
    """

    def __init__(
        self,
        config: ProposerConfig | None = None,
        system_prompt: str | None = None,
    ):
        """
        Initialize the proposer.

        Args:
            config: Proposer configuration
            system_prompt: Custom system prompt (uses default if None)
        """
        self.config = config or ProposerConfig()
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        self._model = None
        self._tokenizer = None
        self._stats = {
            "total_proposals": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
        }

    def _load_local_model(self):
        """Load the local model for inference."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        print(f"Loading model: {self.config.model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        if self.config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import torch

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    device_map=self.config.device,
                    trust_remote_code=True,
                )
            except ImportError:
                print("bitsandbytes not available, loading in full precision")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    device_map=self.config.device,
                    trust_remote_code=True,
                )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map=self.config.device,
                trust_remote_code=True,
            )

        print("Model loaded successfully")

    def _generate_local(self, prompt: str) -> str:
        """Generate response using local model."""
        self._load_local_model()

        # Format for chat
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        import torch

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return response

    def _generate_api(self, prompt: str) -> str:
        """Generate response using API."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai library required for API calls. Install with: pip install openai"
            )

        client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
        )

        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        return response.choices[0].message.content

    def propose(
        self,
        goal_description: str,
        num_qubits: int,
        hamiltonian_type: str,
        constraints: dict | None = None,
        feedback: str | None = None,
    ) -> CircuitProposal:
        """
        Propose a quantum circuit for the given physics goal.

        Args:
            goal_description: Description of what the circuit should achieve
            num_qubits: Number of qubits
            hamiltonian_type: Type of Hamiltonian
            constraints: Optional design constraints
            feedback: Optional feedback from previous attempts

        Returns:
            CircuitProposal containing code and reasoning
        """
        prompt = build_circuit_prompt(
            goal_description=goal_description,
            num_qubits=num_qubits,
            hamiltonian_type=hamiltonian_type,
            constraints=constraints,
            feedback=feedback,
        )

        # Generate response
        if self.config.use_local:
            raw_response = self._generate_local(prompt)
        else:
            raw_response = self._generate_api(prompt)

        # Extract code and reasoning
        code = extract_code_from_response(raw_response)
        reasoning = extract_reasoning_from_response(raw_response)

        self._stats["total_proposals"] += 1
        if code:
            self._stats["successful_extractions"] += 1
        else:
            self._stats["failed_extractions"] += 1

        return CircuitProposal(
            code=code or "",
            reasoning=reasoning,
            raw_response=raw_response,
            metadata={
                "num_qubits": num_qubits,
                "hamiltonian_type": hamiltonian_type,
                "has_feedback": feedback is not None,
            },
        )

    def propose_variations(
        self,
        base_code: str,
        num_variations: int = 3,
        variation_type: str = "depth",
    ) -> list[CircuitProposal]:
        """
        Generate variations of an existing circuit.

        Args:
            base_code: The base circuit code to vary
            num_variations: Number of variations to generate
            variation_type: Type of variation (depth, entanglement, gates)

        Returns:
            List of circuit proposals
        """
        variation_prompts = {
            "depth": "Create a variation with different depth (shallower or deeper)",
            "entanglement": "Create a variation with a different entanglement pattern",
            "gates": "Create a variation using different rotation gates",
            "parameters": "Create a variation with different parameter placement",
        }

        prompt = f"""Given this quantum circuit:

```python
{base_code}
```

{variation_prompts.get(variation_type, 'Create a variation of this circuit')}.

Generate {num_variations} different variations, each as a complete create_ansatz function.
"""

        proposals = []
        for i in range(num_variations):
            if self.config.use_local:
                raw_response = self._generate_local(prompt)
            else:
                raw_response = self._generate_api(prompt)

            code = extract_code_from_response(raw_response)
            reasoning = extract_reasoning_from_response(raw_response)

            proposals.append(
                CircuitProposal(
                    code=code or "",
                    reasoning=reasoning,
                    raw_response=raw_response,
                    metadata={
                        "variation_type": variation_type,
                        "variation_index": i,
                    },
                )
            )

        return proposals

    def get_stats(self) -> dict:
        """Get proposer statistics."""
        return self._stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self._stats = {
            "total_proposals": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
        }


class MockProposer:
    """
    Mock proposer for testing without LLM.

    Returns predefined circuit templates.
    """

    def __init__(self):
        self._templates = {
            "xy_chain": self._xy_template,
            "heisenberg": self._heisenberg_template,
            "transverse_ising": self._tfim_template,
            "default": self._default_template,
        }
        self._stats = {"total_proposals": 0}

    def _default_template(self, num_qubits: int) -> str:
        return f'''from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int = {num_qubits}) -> QuantumCircuit:
    """Hardware-efficient ansatz with RY rotations and linear CX."""
    qc = QuantumCircuit(num_qubits)
    params = []

    # Layer 1: RY rotations
    for i in range(num_qubits):
        p = Parameter(f'theta_{{len(params)}}')
        params.append(p)
        qc.ry(p, i)

    # Entanglement: Linear CX chain
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    # Layer 2: RY rotations
    for i in range(num_qubits):
        p = Parameter(f'theta_{{len(params)}}')
        params.append(p)
        qc.ry(p, i)

    return qc
'''

    def _xy_template(self, num_qubits: int) -> str:
        return f'''from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int = {num_qubits}) -> QuantumCircuit:
    """Ansatz for XY chain: RY-CX-RY structure with linear entanglement."""
    qc = QuantumCircuit(num_qubits)
    params = []

    for layer in range(2):
        # Rotation layer
        for i in range(num_qubits):
            p = Parameter(f'theta_{{len(params)}}')
            params.append(p)
            qc.ry(p, i)

        # Entanglement (skip on last layer)
        if layer < 1:
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)

    return qc
'''

    def _heisenberg_template(self, num_qubits: int) -> str:
        return f'''from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int = {num_qubits}) -> QuantumCircuit:
    """Ansatz for Heisenberg model: RY-RZ rotations with circular entanglement."""
    qc = QuantumCircuit(num_qubits)
    params = []

    for layer in range(2):
        # RY rotations
        for i in range(num_qubits):
            p = Parameter(f'theta_{{len(params)}}')
            params.append(p)
            qc.ry(p, i)

        # RZ rotations
        for i in range(num_qubits):
            p = Parameter(f'theta_{{len(params)}}')
            params.append(p)
            qc.rz(p, i)

        # Circular entanglement
        for i in range(num_qubits):
            qc.cx(i, (i + 1) % num_qubits)

    return qc
'''

    def _tfim_template(self, num_qubits: int) -> str:
        return f'''from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def create_ansatz(num_qubits: int = {num_qubits}) -> QuantumCircuit:
    """Ansatz for transverse-field Ising: RX-RZ with ZZ interaction layers."""
    qc = QuantumCircuit(num_qubits)
    params = []

    for layer in range(2):
        # RX rotations (for transverse field)
        for i in range(num_qubits):
            p = Parameter(f'theta_{{len(params)}}')
            params.append(p)
            qc.rx(p, i)

        # RZ rotations
        for i in range(num_qubits):
            p = Parameter(f'theta_{{len(params)}}')
            params.append(p)
            qc.rz(p, i)

        # ZZ interaction via CX-RZ-CX
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

    return qc
'''

    def propose(
        self,
        goal_description: str,
        num_qubits: int,
        hamiltonian_type: str,
        constraints: dict | None = None,
        feedback: str | None = None,
    ) -> CircuitProposal:
        """Generate a mock proposal using templates."""
        template_fn = self._templates.get(hamiltonian_type, self._templates["default"])
        code = template_fn(num_qubits)

        self._stats["total_proposals"] += 1

        return CircuitProposal(
            code=code,
            reasoning=f"Template-based ansatz for {hamiltonian_type} with {num_qubits} qubits",
            raw_response=code,
            metadata={
                "num_qubits": num_qubits,
                "hamiltonian_type": hamiltonian_type,
                "is_mock": True,
            },
        )

    def get_stats(self) -> dict:
        """Get proposer statistics."""
        return self._stats.copy()
