# QuantumMind Guideline: Phase 7 - Discovery Campaigns

> Steps 7.1-7.3: XY Chain, Heisenberg, TFIM Discovery

---

## Step 7.1: XY Chain Discovery

**Status**: PENDING

**Prerequisites**: All previous phases complete (1-6)

---

### Create Experiment Directory Structure

```
experiments/
└── xy_chain/
    ├── config.yaml
    ├── run_discovery.py
    ├── results/
    │   ├── 4q/
    │   ├── 6q/
    │   └── 8q/
    └── logs/
```

---

### Create Experiment Config

```yaml
# experiments/xy_chain/config.yaml

experiment:
  name: "XY Chain Discovery"
  hamiltonian: "XY_CHAIN"
  description: "Discover novel ansatzes for XY chain ground state preparation"

targets:
  - num_qubits: 4
    max_iterations: 100
    target_error: 0.01
    vqe_max_iter: 200

  - num_qubits: 6
    max_iterations: 150
    target_error: 0.01
    vqe_max_iter: 250

  - num_qubits: 8
    max_iterations: 200
    target_error: 0.02
    vqe_max_iter: 300

agent:
  model_path: "./models/checkpoints/quantum-mind-v1"
  circuits_per_iteration: 3
  temperature: 0.7
  max_retries: 5

vqe:
  optimizer: "COBYLA"
  max_iterations: 200

hardware:
  enabled: true  # Set false for initial runs
  backend: "ibm_sherbrooke"
  resilience_level: 2
  shots: 4000
  runs_per_circuit: 5

logging:
  log_dir: "./experiments/xy_chain/logs"
  save_every: 10
  verbose: true

seeds: [42, 123, 456, 789, 101112]
```

---

### Create Discovery Runner

```python
# experiments/xy_chain/run_discovery.py

import yaml
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from src.agent.discovery import DiscoveryAgent
from src.quantum.hamiltonians import xy_chain
from src.evaluation.compare import compare_to_baselines, save_comparison, format_comparison_report
from src.evaluation.statistics import test_improvement_significance, format_significance_report
from src.quantum.hardware import IBMHardwareRunner, format_hardware_results


def load_config():
    """Load experiment configuration."""
    with open("experiments/xy_chain/config.yaml") as f:
        return yaml.safe_load(f)


def run_single_target(config: dict, target: dict) -> dict:
    """
    Run discovery for a single qubit count.

    Args:
        config: Full config dict
        target: Target configuration (num_qubits, max_iterations, etc.)

    Returns:
        Results dict for this target
    """

    num_qubits = target["num_qubits"]
    results_dir = Path(f"experiments/xy_chain/results/{num_qubits}q")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DISCOVERY: {num_qubits}-qubit XY Chain")
    print(f"{'='*60}")

    # Get exact energy
    ham_result = xy_chain(num_qubits)
    exact_energy = ham_result.exact_energy
    print(f"Exact energy: {exact_energy:.6f}")
    print(f"Target error: {target['target_error']}")

    # Initialize agent
    agent = DiscoveryAgent(
        model_path=config["agent"]["model_path"],
        hamiltonian_name="XY_CHAIN",
        num_qubits=num_qubits,
        max_iterations=target["max_iterations"]
    )

    # Run discovery
    print(f"\nStarting discovery (max {target['max_iterations']} iterations)...")
    discovery_result = agent.run()

    if discovery_result.best_circuit is None:
        print("WARNING: No valid circuit discovered!")
        return {
            "num_qubits": num_qubits,
            "success": False,
            "error": "No valid circuit found"
        }

    # Save circuit code
    circuit_path = results_dir / "best_circuit.py"
    with open(circuit_path, "w") as f:
        f.write(discovery_result.circuit_code)
    print(f"Saved circuit to {circuit_path}")

    # Compare to baselines
    print("\nComparing to baselines...")
    comparison = compare_to_baselines(
        discovered_circuit=discovery_result.best_circuit,
        discovered_name=f"QuantumMind-XY-{num_qubits}q",
        hamiltonian_name="XY_CHAIN",
        num_qubits=num_qubits,
        num_trials=10
    )

    # Save comparison
    save_comparison(comparison, str(results_dir / "comparison.json"))

    # Save comparison report
    report = format_comparison_report(comparison)
    with open(results_dir / "comparison_report.md", "w") as f:
        f.write(report)

    # Statistical significance tests
    print("\nRunning statistical tests...")
    sig_results = []
    for baseline_name, baseline_stats in comparison.baselines.items():
        discovered_errors = [r.energy_error for r in comparison.discovered.all_results]
        baseline_errors = [r.energy_error for r in baseline_stats.all_results]

        sig = test_improvement_significance(discovered_errors, baseline_errors)
        sig_results.append((baseline_name, sig))

    # Save significance report
    sig_report = format_significance_report(sig_results)
    with open(results_dir / "significance_report.md", "w") as f:
        f.write(sig_report)

    # Hardware validation (if enabled)
    hw_results = None
    if config["hardware"]["enabled"]:
        print("\nRunning hardware validation...")
        runner = IBMHardwareRunner(
            backend_name=config["hardware"]["backend"],
            resilience_level=config["hardware"]["resilience_level"]
        )

        hw_results = runner.run_multiple(
            circuit=discovery_result.best_circuit,
            hamiltonian=ham_result.operator,
            parameter_values=discovery_result.optimal_params,
            num_runs=config["hardware"]["runs_per_circuit"],
            shots=config["hardware"]["shots"]
        )

        # Save hardware results
        hw_report = format_hardware_results(hw_results, exact_energy)
        with open(results_dir / "hardware_results.md", "w") as f:
            f.write(hw_report)

        hw_data = {
            "energies": [r.energy for r in hw_results],
            "std_errors": [r.std_error for r in hw_results],
            "job_ids": [r.job_id for r in hw_results],
            "mean_energy": float(np.mean([r.energy for r in hw_results])),
            "mean_error": float(np.mean([abs(r.energy - exact_energy) for r in hw_results]))
        }
        with open(results_dir / "hardware_data.json", "w") as f:
            json.dump(hw_data, f, indent=2)

    # Compile final results
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "num_qubits": num_qubits,
        "exact_energy": exact_energy,
        "success": True,

        # Discovery metrics
        "discovery": {
            "iterations": discovery_result.total_iterations,
            "best_energy": discovery_result.best_energy,
            "best_error": discovery_result.best_error,
            "circuits_evaluated": discovery_result.circuits_evaluated
        },

        # Circuit metrics
        "circuit": {
            "depth": discovery_result.best_circuit.depth(),
            "gates": sum(discovery_result.best_circuit.count_ops().values()),
            "parameters": len(discovery_result.best_circuit.parameters)
        },

        # Comparison metrics
        "comparison": {
            "beats_all_baselines": comparison.beats_all_baselines,
            "best_improvement_percent": comparison.best_improvement_percent,
            "best_baseline": comparison.best_baseline_name,
            "mean_error": comparison.discovered.mean_error
        },

        # Hardware metrics (if run)
        "hardware": hw_data if hw_results else None
    }

    # Save final results
    with open(results_dir / "results.json", "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULT: {num_qubits}-qubit XY Chain")
    print(f"{'='*60}")
    print(f"Energy error: {discovery_result.best_error:.6f}")
    print(f"Beats baselines: {'YES' if comparison.beats_all_baselines else 'NO'}")
    print(f"Best improvement: {comparison.best_improvement_percent:.1f}%")
    if hw_results:
        print(f"Hardware mean error: {result_data['hardware']['mean_error']:.6f}")

    return result_data


def run_xy_chain_discovery():
    """Run full XY chain discovery campaign."""

    config = load_config()

    print("="*60)
    print("XY CHAIN DISCOVERY CAMPAIGN")
    print("="*60)
    print(f"Targets: {[t['num_qubits'] for t in config['targets']]} qubits")
    print(f"Model: {config['agent']['model_path']}")
    print(f"Hardware: {'enabled' if config['hardware']['enabled'] else 'disabled'}")
    print("="*60)

    all_results = []

    for target in config["targets"]:
        result = run_single_target(config, target)
        all_results.append(result)

    # Save campaign summary
    summary_dir = Path("experiments/xy_chain/results")
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "campaign_summary.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": all_results
        }, f, indent=2)

    # Print final summary
    print("\n" + "="*60)
    print("CAMPAIGN COMPLETE")
    print("="*60)

    successful = [r for r in all_results if r.get("success", False)]
    beats_all = [r for r in successful if r.get("comparison", {}).get("beats_all_baselines", False)]

    print(f"Successful discoveries: {len(successful)}/{len(all_results)}")
    print(f"Beat all baselines: {len(beats_all)}/{len(successful)}")

    for r in successful:
        n = r["num_qubits"]
        err = r["discovery"]["best_error"]
        beats = "BEATS" if r["comparison"]["beats_all_baselines"] else "below"
        imp = r["comparison"]["best_improvement_percent"]
        print(f"  {n}q: error={err:.6f}, {beats} baselines ({imp:.1f}% improvement)")


if __name__ == "__main__":
    run_xy_chain_discovery()
```

---

## Step 7.1 Verification Checklist

- [ ] `experiments/xy_chain/` directory created
- [ ] `config.yaml` with all parameters
- [ ] `run_discovery.py` executes without error
- [ ] Discovery runs for all qubit counts (4, 6, 8)
- [ ] At least one discovered circuit beats all baselines
- [ ] All results saved to `experiments/xy_chain/results/`
- [ ] Circuit code saved for all successful discoveries
- [ ] Comparison reports generated (markdown)
- [ ] Statistical significance tests completed
- [ ] Hardware validation completed for top circuits
- [ ] `campaign_summary.json` generated

**DO NOT PROCEED TO STEP 7.2 UNTIL ALL 11 CHECKS PASS**

---

## Step 7.2: Heisenberg Discovery

**Status**: PENDING

**Prerequisites**: Step 7.1 complete

---

### Create Experiment Structure

Same structure as XY Chain, in `experiments/heisenberg/`

### Config Differences

```yaml
# experiments/heisenberg/config.yaml

experiment:
  name: "Heisenberg Chain Discovery"
  hamiltonian: "HEISENBERG"
  description: "Heisenberg model is more complex, use smaller qubit counts"

targets:
  - num_qubits: 4
    max_iterations: 150  # More iterations needed
    target_error: 0.015

  - num_qubits: 6
    max_iterations: 200
    target_error: 0.02

# Note: 8 qubits too expensive for Heisenberg
```

### Key Differences

1. Use `heisenberg_chain()` from hamiltonians module
2. Expect higher errors (more complex Hamiltonian)
3. May need more VQE iterations
4. Deeper circuits likely needed

---

## Step 7.2 Verification Checklist

- [ ] `experiments/heisenberg/` directory created
- [ ] Discovery runs for 4 and 6 qubits
- [ ] Results saved to `experiments/heisenberg/results/`
- [ ] Comparison to baselines complete
- [ ] Statistical tests complete
- [ ] Hardware validation complete
- [ ] `campaign_summary.json` generated

**DO NOT PROCEED TO STEP 7.3 UNTIL ALL 7 CHECKS PASS**

---

## Step 7.3: TFIM Discovery

**Status**: PENDING

**Prerequisites**: Step 7.2 complete

---

### Create Experiment Structure

Same structure as XY Chain, in `experiments/tfim/`

### Config Differences

```yaml
# experiments/tfim/config.yaml

experiment:
  name: "TFIM Discovery"
  hamiltonian: "TFIM"
  description: "Transverse-field Ising model with varying field strength"

targets:
  - num_qubits: 4
    max_iterations: 100
    target_error: 0.01
    h: 1.0  # Transverse field strength

  - num_qubits: 6
    max_iterations: 150
    target_error: 0.015
    h: 1.0

  - num_qubits: 8
    max_iterations: 200
    target_error: 0.02
    h: 1.0

  # Phase transition point
  - num_qubits: 6
    max_iterations: 200
    target_error: 0.02
    h: 0.5  # Near critical point
```

### Key Differences

1. Use `tfim_chain(n, h=1.0)` from hamiltonians module
2. Include variation of field strength h
3. Near critical point (h ~ J) is hardest
4. Compare performance across field strengths

---

## Step 7.3 Verification Checklist

- [ ] `experiments/tfim/` directory created
- [ ] Discovery runs for all qubit counts (4, 6, 8)
- [ ] Discovery runs for multiple field strengths
- [ ] Results saved to `experiments/tfim/results/`
- [ ] Comparison to baselines complete
- [ ] Statistical tests complete
- [ ] Hardware validation complete
- [ ] `campaign_summary.json` generated

**DO NOT PROCEED TO PHASE 8 UNTIL ALL 8 CHECKS PASS**

---

## Final Phase 7 Summary

After all three discovery campaigns:

```python
# scripts/aggregate_discovery_results.py

import json
from pathlib import Path

def aggregate_results():
    experiments = ["xy_chain", "heisenberg", "tfim"]

    summary = {
        "total_circuits_discovered": 0,
        "beats_all_baselines": 0,
        "hardware_validated": 0,
        "experiments": {}
    }

    for exp in experiments:
        summary_path = Path(f"experiments/{exp}/results/campaign_summary.json")
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)

            results = data.get("results", [])
            successful = [r for r in results if r.get("success")]
            beats = [r for r in successful if r.get("comparison", {}).get("beats_all_baselines")]

            summary["experiments"][exp] = {
                "targets": len(results),
                "successful": len(successful),
                "beats_baselines": len(beats)
            }

            summary["total_circuits_discovered"] += len(successful)
            summary["beats_all_baselines"] += len(beats)

    with open("experiments/overall_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Overall Discovery Summary")
    print("="*40)
    print(f"Total circuits discovered: {summary['total_circuits_discovered']}")
    print(f"Beat all baselines: {summary['beats_all_baselines']}")
    for exp, data in summary["experiments"].items():
        print(f"  {exp}: {data['beats_baselines']}/{data['successful']} beat baselines")

if __name__ == "__main__":
    aggregate_results()
```

---

## After Completion

Update `00_OVERVIEW.md`:
- Change Phase 7 progress to 3/3
- Mark all Phase 7 steps as DONE

Next: Proceed to `08_PAPER.md` (Step 8.1)

---

*Phase 7 produces the core discovery results for the research paper.*
