# QuantumMind Guideline: Phase 8 - Documentation & Paper

> Steps 8.1-8.2: Results Documentation, Paper Draft

---

## Step 8.1: Results Documentation

**Status**: PENDING

**Prerequisites**: Phase 7 complete (all discovery campaigns)

---

### Action 1: Aggregate All Results

```python
# scripts/aggregate_results.py

import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def aggregate_all_results():
    """Aggregate results from all discovery campaigns."""

    experiments = ["xy_chain", "heisenberg", "tfim"]
    all_data = []

    for exp in experiments:
        results_dir = Path(f"experiments/{exp}/results")

        if not results_dir.exists():
            print(f"Warning: {results_dir} not found")
            continue

        # Find all qubit-specific results
        for subdir in results_dir.iterdir():
            if subdir.is_dir() and subdir.name.endswith("q"):
                result_file = subdir / "results.json"
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                        data["experiment"] = exp
                        all_data.append(data)

    if not all_data:
        print("No results found!")
        return None

    # Create DataFrame
    rows = []
    for d in all_data:
        row = {
            "experiment": d["experiment"],
            "num_qubits": d["num_qubits"],
            "exact_energy": d["exact_energy"],
            "best_energy": d["discovery"]["best_energy"],
            "best_error": d["discovery"]["best_error"],
            "iterations": d["discovery"]["iterations"],
            "circuit_depth": d["circuit"]["depth"],
            "circuit_gates": d["circuit"]["gates"],
            "circuit_params": d["circuit"]["parameters"],
            "beats_baselines": d["comparison"]["beats_all_baselines"],
            "improvement_percent": d["comparison"]["best_improvement_percent"],
            "mean_error": d["comparison"]["mean_error"],
        }

        # Add hardware if available
        if d.get("hardware"):
            row["hw_mean_energy"] = d["hardware"]["mean_energy"]
            row["hw_mean_error"] = d["hardware"]["mean_error"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save as CSV
    output_dir = Path("experiments")
    df.to_csv(output_dir / "all_results.csv", index=False)
    print(f"Saved to {output_dir / 'all_results.csv'}")

    # Generate summary statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    for exp in experiments:
        exp_df = df[df["experiment"] == exp]
        if len(exp_df) > 0:
            print(f"\n{exp.upper()}")
            print("-"*40)
            print(f"  Targets: {len(exp_df)}")
            print(f"  Beat baselines: {exp_df['beats_baselines'].sum()}")
            print(f"  Mean error: {exp_df['best_error'].mean():.6f}")
            print(f"  Mean improvement: {exp_df['improvement_percent'].mean():.1f}%")

    return df


def generate_latex_table(df: pd.DataFrame, experiment: str) -> str:
    """Generate LaTeX table for paper."""

    exp_df = df[df["experiment"] == experiment].copy()

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{Results for {experiment.replace('_', ' ').title()} Hamiltonian}}",
        r"\begin{tabular}{|c|c|c|c|c|c|}",
        r"\hline",
        r"Qubits & Error & Depth & Gates & Improvement & Beats All \\",
        r"\hline",
    ]

    for _, row in exp_df.iterrows():
        beats = "Yes" if row["beats_baselines"] else "No"
        lines.append(
            f"{row['num_qubits']} & {row['best_error']:.4f} & {row['circuit_depth']} & "
            f"{row['circuit_gates']} & {row['improvement_percent']:.1f}\\% & {beats} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    df = aggregate_all_results()

    if df is not None:
        # Generate LaTeX tables
        for exp in ["xy_chain", "heisenberg", "tfim"]:
            if len(df[df["experiment"] == exp]) > 0:
                latex = generate_latex_table(df, exp)
                with open(f"paper/tables/{exp}_results.tex", "w") as f:
                    f.write(latex)
                print(f"Generated paper/tables/{exp}_results.tex")
```

---

### Action 2: Generate Publication-Quality Figures

```python
# scripts/generate_figures.py

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd


# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_energy_error_comparison(df: pd.DataFrame, experiment: str, output_dir: Path):
    """
    Bar chart comparing energy errors across methods.

    Figure 2 in paper.
    """

    exp_df = df[df["experiment"] == experiment]

    fig, ax = plt.subplots()

    qubits = exp_df["num_qubits"].values
    discovered_errors = exp_df["best_error"].values

    # Load baseline errors from comparison files
    baseline_errors = {}
    for _, row in exp_df.iterrows():
        n = row["num_qubits"]
        comp_file = Path(f"experiments/{experiment}/results/{n}q/comparison.json")
        if comp_file.exists():
            with open(comp_file) as f:
                comp = json.load(f)
                for baseline, data in comp["baselines"].items():
                    if baseline not in baseline_errors:
                        baseline_errors[baseline] = {}
                    baseline_errors[baseline][n] = data["mean_error"]

    x = np.arange(len(qubits))
    width = 0.15
    offset = 0

    # Plot discovered
    ax.bar(x + offset, discovered_errors, width, label="QuantumMind", color="#2ecc71")
    offset += width

    # Plot baselines
    colors = ["#3498db", "#e74c3c", "#9b59b6", "#f39c12"]
    for i, (baseline, errors) in enumerate(list(baseline_errors.items())[:4]):
        values = [errors.get(q, 0) for q in qubits]
        ax.bar(x + offset, values, width, label=baseline, color=colors[i % len(colors)])
        offset += width

    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Energy Error (Ha)")
    ax.set_title(f"Energy Error Comparison: {experiment.replace('_', ' ').title()}")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(qubits)
    ax.legend(loc="upper left")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment}_error_comparison.png")
    plt.savefig(output_dir / f"{experiment}_error_comparison.pdf")
    plt.close()

    print(f"Saved {experiment}_error_comparison figures")


def plot_circuit_depth_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Circuit depth vs energy error scatter plot.

    Figure 3 in paper.
    """

    fig, ax = plt.subplots()

    colors = {"xy_chain": "#2ecc71", "heisenberg": "#3498db", "tfim": "#e74c3c"}
    markers = {"xy_chain": "o", "heisenberg": "s", "tfim": "^"}

    for exp in df["experiment"].unique():
        exp_df = df[df["experiment"] == exp]
        ax.scatter(
            exp_df["circuit_depth"],
            exp_df["best_error"],
            c=colors.get(exp, "#666"),
            marker=markers.get(exp, "o"),
            s=100,
            label=exp.replace("_", " ").title(),
            alpha=0.7
        )

    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel("Energy Error (Ha)")
    ax.set_title("Circuit Depth vs Energy Error")
    ax.legend()
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_dir / "depth_vs_error.png")
    plt.savefig(output_dir / "depth_vs_error.pdf")
    plt.close()

    print("Saved depth_vs_error figures")


def plot_hardware_vs_simulation(df: pd.DataFrame, output_dir: Path):
    """
    Hardware vs simulation comparison.

    Figure 5 in paper.
    """

    # Filter rows with hardware data
    hw_df = df[df["hw_mean_error"].notna()].copy()

    if len(hw_df) == 0:
        print("No hardware data available")
        return

    fig, ax = plt.subplots()

    x = range(len(hw_df))
    sim_errors = hw_df["best_error"].values
    hw_errors = hw_df["hw_mean_error"].values
    labels = [f"{row['experiment'][:2]}-{row['num_qubits']}q" for _, row in hw_df.iterrows()]

    width = 0.35
    ax.bar([i - width/2 for i in x], sim_errors, width, label="Simulation", color="#3498db")
    ax.bar([i + width/2 for i in x], hw_errors, width, label="Hardware", color="#e74c3c")

    ax.set_xlabel("Circuit")
    ax.set_ylabel("Energy Error (Ha)")
    ax.set_title("Simulation vs Hardware Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "hardware_vs_simulation.png")
    plt.savefig(output_dir / "hardware_vs_simulation.pdf")
    plt.close()

    print("Saved hardware_vs_simulation figures")


def main():
    # Load data
    df = pd.read_csv("experiments/all_results.csv")

    # Create output directory
    output_dir = Path("paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all figures
    for exp in df["experiment"].unique():
        plot_energy_error_comparison(df, exp, output_dir)

    plot_circuit_depth_comparison(df, output_dir)
    plot_hardware_vs_simulation(df, output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
```

---

### Action 3: Document Novel Patterns

```python
# scripts/analyze_patterns.py

from pathlib import Path
import json
from collections import Counter


def analyze_discovered_circuits():
    """Analyze patterns in discovered circuits."""

    experiments = ["xy_chain", "heisenberg", "tfim"]

    patterns = {
        "gate_types": Counter(),
        "entanglement_patterns": [],
        "depth_ranges": [],
        "param_counts": []
    }

    for exp in experiments:
        results_dir = Path(f"experiments/{exp}/results")

        for subdir in results_dir.iterdir():
            if subdir.is_dir() and (subdir / "best_circuit.py").exists():
                circuit_code = (subdir / "best_circuit.py").read_text()

                # Extract gate usage
                gates = ["rx", "ry", "rz", "cx", "cz", "swap", "h", "x", "y", "z"]
                for gate in gates:
                    count = circuit_code.lower().count(f".{gate}(")
                    if count > 0:
                        patterns["gate_types"][gate] += count

                # Load metrics
                if (subdir / "results.json").exists():
                    with open(subdir / "results.json") as f:
                        data = json.load(f)
                        patterns["depth_ranges"].append(data["circuit"]["depth"])
                        patterns["param_counts"].append(data["circuit"]["parameters"])

    # Generate report
    report = []
    report.append("# Discovered Circuit Pattern Analysis\n")

    report.append("## Gate Usage\n")
    report.append("| Gate | Usage Count |")
    report.append("|------|-------------|")
    for gate, count in patterns["gate_types"].most_common():
        report.append(f"| {gate.upper()} | {count} |")

    report.append("\n## Circuit Statistics\n")
    if patterns["depth_ranges"]:
        report.append(f"- **Depth range**: {min(patterns['depth_ranges'])} - {max(patterns['depth_ranges'])}")
        report.append(f"- **Average depth**: {sum(patterns['depth_ranges'])/len(patterns['depth_ranges']):.1f}")

    if patterns["param_counts"]:
        report.append(f"- **Parameter range**: {min(patterns['param_counts'])} - {max(patterns['param_counts'])}")
        report.append(f"- **Average parameters**: {sum(patterns['param_counts'])/len(patterns['param_counts']):.1f}")

    report.append("\n## Key Observations\n")
    report.append("- [TO BE FILLED: Identify recurring patterns]")
    report.append("- [TO BE FILLED: Compare to known ansatz structures]")
    report.append("- [TO BE FILLED: Explain why patterns work]")

    output = "\n".join(report)

    with open("paper/pattern_analysis.md", "w") as f:
        f.write(output)

    print(output)
    print("\nSaved to paper/pattern_analysis.md")


if __name__ == "__main__":
    analyze_discovered_circuits()
```

---

## Step 8.1 Verification Checklist

- [ ] All results aggregated in `experiments/all_results.csv`
- [ ] LaTeX tables generated for each experiment
- [ ] Energy error comparison figures generated (Fig 2)
- [ ] Circuit depth vs error figure generated (Fig 3)
- [ ] Hardware vs simulation figure generated (Fig 5)
- [ ] All figures saved as PNG (300 DPI) and PDF
- [ ] Pattern analysis documented
- [ ] All claims have supporting data in results files

**DO NOT PROCEED TO STEP 8.2 UNTIL ALL 8 CHECKS PASS**

---

## Step 8.2: Paper Draft

**Status**: PENDING

**Prerequisites**: Step 8.1 complete

---

### Paper Directory Structure

```
paper/
├── main.tex
├── references.bib
├── figures/
│   ├── architecture.pdf
│   ├── xy_chain_error_comparison.pdf
│   ├── depth_vs_error.pdf
│   └── hardware_vs_simulation.pdf
├── tables/
│   ├── xy_chain_results.tex
│   ├── heisenberg_results.tex
│   └── tfim_results.tex
└── supplementary.tex
```

---

### Main Paper Template

```latex
% paper/main.tex

\documentclass[11pt]{article}
\usepackage{neurips_2026}  % or appropriate venue style
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{hyperref}

\title{QuantumMind: Autonomous Discovery of Quantum Circuits via LLM-Guided Search}

\author{
  [Author Name] \\
  [Institution] \\
  \texttt{[email]}
}

\begin{document}
\maketitle

\begin{abstract}
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
circuit achieves [X]\% lower energy error than the standard hardware-efficient
ansatz while using [Y]\% fewer gates. All discovered circuits are validated
on IBM's 127-qubit Eagle processor.

Our work demonstrates that LLMs can serve as effective hypothesis generators
for quantum algorithm discovery, opening new avenues for AI-assisted
quantum research.
\end{abstract}

%======================================================================
\section{Introduction}
%======================================================================

[Content follows RESEARCH_PAPER.md structure]

% Opening: Promise and challenge of VQAs
% Gap: No autonomous discovery system
% Contribution: QuantumMind
% Paper outline

%======================================================================
\section{Related Work}
%======================================================================

\subsection{Variational Quantum Algorithms}
% VQE, QAOA fundamentals
% Ansatz design importance

\subsection{Quantum Architecture Search}
% Evolutionary approaches
% Reinforcement learning methods

\subsection{LLMs for Quantum Computing}
% Agent-Q
% QGAS
% QuantumLLMInstruct

%======================================================================
\section{Method}
%======================================================================

\subsection{System Overview}

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/architecture.pdf}
\caption{QuantumMind architecture. The agent iteratively proposes circuits,
filters barren plateaus, evaluates via VQE, and accumulates knowledge.}
\label{fig:architecture}
\end{figure}

\subsection{LLM Fine-Tuning}

\subsection{Barren Plateau Detection}

\subsection{Memory System}

\subsection{Hardware Validation}

%======================================================================
\section{Experiments}
%======================================================================

\subsection{Experimental Setup}

\subsection{Results: XY Spin Chain}

\input{tables/xy_chain_results.tex}

\begin{figure}[t]
\centering
\includegraphics[width=0.8\textwidth]{figures/xy_chain_error_comparison.pdf}
\caption{Energy error comparison on XY chain.}
\label{fig:xy_results}
\end{figure}

\subsection{Results: Heisenberg Model}

\input{tables/heisenberg_results.tex}

\subsection{Results: Transverse-Field Ising}

\input{tables/tfim_results.tex}

\subsection{Analysis of Discovered Circuits}

\begin{figure}[t]
\centering
\includegraphics[width=0.8\textwidth]{figures/depth_vs_error.pdf}
\caption{Circuit depth vs energy error across all experiments.}
\label{fig:depth_error}
\end{figure}

%======================================================================
\section{Discussion}
%======================================================================

\subsection{Key Findings}

\subsection{Limitations}

\subsection{Broader Impact}

%======================================================================
\section{Conclusion}
%======================================================================

% Summary of contributions
% Future work

%======================================================================
\section*{Acknowledgments}
%======================================================================

% IBM Quantum access, compute resources, etc.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
```

---

### References Template

```bibtex
% paper/references.bib

@article{mcclean2018barren,
  title={Barren plateaus in quantum neural network training landscapes},
  author={McClean, Jarrod R and Boixo, Sergio and Smelyanskiy, Vadim N and Babbush, Ryan and Neven, Hartmut},
  journal={Nature Communications},
  volume={9},
  pages={4812},
  year={2018}
}

@article{cerezo2021variational,
  title={Variational quantum algorithms},
  author={Cerezo, Marco and Arrasmith, Andrew and Babbush, Ryan and others},
  journal={Nature Reviews Physics},
  volume={3},
  pages={625--644},
  year={2021}
}

@article{peruzzo2014variational,
  title={A variational eigenvalue solver on a photonic quantum processor},
  author={Peruzzo, Alberto and McClean, Jarrod and Shadbolt, Peter and others},
  journal={Nature Communications},
  volume={5},
  pages={4213},
  year={2014}
}

% Add all other references from RESEARCH_PAPER.md
```

---

### Writing Checklist Per Section

**Abstract**
- [ ] Problem statement (1-2 sentences)
- [ ] Approach (2-3 sentences)
- [ ] Key results with numbers
- [ ] Impact statement

**Introduction**
- [ ] Opening hook
- [ ] Problem description
- [ ] Gap in existing work
- [ ] Contributions (bulleted)
- [ ] Paper outline

**Related Work**
- [ ] VQA background
- [ ] Architecture search methods
- [ ] LLM for quantum
- [ ] Clear differentiation

**Method**
- [ ] System architecture figure
- [ ] LLM fine-tuning details
- [ ] BP detection explanation
- [ ] Memory system description
- [ ] Hardware validation process

**Experiments**
- [ ] Setup clearly described
- [ ] Results tables with std
- [ ] Comparison figures
- [ ] Statistical significance
- [ ] Hardware validation

**Discussion**
- [ ] Key findings summarized
- [ ] Limitations acknowledged
- [ ] Broader impact considered

**Conclusion**
- [ ] Contributions restated
- [ ] Future work outlined

---

## Step 8.2 Verification Checklist

- [ ] `paper/main.tex` created with full structure
- [ ] Abstract complete with actual numbers from results
- [ ] All 6 main sections written
- [ ] All figures embedded and referenced
- [ ] All tables embedded and referenced
- [ ] `references.bib` complete (minimum 20 citations)
- [ ] Paper compiles without errors: `pdflatex main.tex`
- [ ] Page count appropriate for venue (8-12 pages)
- [ ] Spell check passed
- [ ] Internal consistency verified (numbers match throughout)

**PROJECT COMPLETE WHEN ALL 10 CHECKS PASS**

---

## Final Checklist Before Submission

- [ ] All code is in GitHub repository
- [ ] README.md updated with final results
- [ ] Model weights uploaded to HuggingFace (if sharing)
- [ ] Reproducibility notebooks created
- [ ] Paper PDF generated and reviewed
- [ ] Supplementary materials prepared
- [ ] All claims verified against data
- [ ] Co-author approval (if applicable)

---

## After Completion

Update `00_OVERVIEW.md`:
- Change Phase 8 progress to 2/2
- Mark all Phase 8 steps as DONE
- Update total progress to 29/29

---

*Phase 8 produces the research paper documenting your discoveries.*

**CONGRATULATIONS - PROJECT COMPLETE**
