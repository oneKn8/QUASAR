# Strategic 3-Project Plan for ML/AI Internship

## The Narrative: "LLMs for Scientific Computing"

Your portfolio tells one story: **You make AI work for science.**

This differentiates you from:
- Generic "I built a chatbot" developers
- Pure ML researchers with no systems skills
- Systems engineers who don't understand models

You are the rare person who can **train models AND deploy them for real scientific workflows**.

---

## F1 Visa Strategic Targeting

### Companies You CAN Target (Accept CPT/OPT, Sponsor H1B)

| Company | Why Good Fit | Relevant Teams |
|---------|--------------|----------------|
| **IBM** | Quantum internships open, sponsors heavily | Quantum Research, IBM Research |
| **NVIDIA** | LLM inference role exists, fits your profile | LLM Performance, Applied Research |
| **Google** | Cloud AI, DeepMind, sponsors many | Cloud AI, Google Research |
| **Microsoft** | Phi team, Azure AI, research labs | MSR, Azure AI, GitHub Copilot |
| **Meta** | FAIR research, PyTorch team | FAIR, AI Research |
| **Amazon** | AWS AI, science teams | AWS AI, Amazon Science |
| **Scale AI** | ML research intern role, data-centric AI | ML Research |
| **Adobe** | AI/ML intern roles open | Adobe Research |
| **Startups** | Anyscale, Modal, Predibase, Oxen.ai | Various |

### Companies to AVOID (F1 Restrictions)

| Company | Reason |
|---------|--------|
| SpaceX | ITAR restricted, requires citizenship/green card |
| Lockheed Martin | Defense contractor, export controls |
| Raytheon | Defense contractor |
| Most NASA positions | Citizenship required (but JPL has some options) |
| Palantir | Government contracts, often restricted |

### Best Bet for Your Profile

**Tier 1 Targets (Best Fit)**:
1. **IBM Quantum** - Your physics interest + they sponsor + internships open
2. **NVIDIA** - LLM Inference Performance role matches your observability background
3. **Google Cloud AI** - Sponsors heavily, scientific computing teams exist

**Tier 2 Targets (Strong Fit)**:
1. Scale AI - ML Research
2. Microsoft Research
3. Predibase - They literally teach GRPO

**Tier 3 (Reach)**:
1. Anthropic (very competitive but worth trying)
2. DeepMind
3. OpenAI

---

## The 3 Projects

### Overview

| Project | Focus | Target Companies | Time |
|---------|-------|------------------|------|
| **AstroCode** | Scientific code generation | IBM, Google, research | Weeks 1-5 |
| **TransitNet** | Exoplanet detection + explainability | NASA JPL, IBM Quantum | Weeks 3-8 |
| **HPCMind** | HPC observability with fine-tuned LLM | NVIDIA, cloud providers | Weeks 6-10 |

---

## Project 1: AstroCode - Scientific Computing Code Generator

### The Pitch
"I fine-tuned an 8B model to outperform GPT-4 on astronomy code generation, benchmarked on AstroVisBench - the first benchmark in this space."

### Why It's Unique
- **AstroVisBench** is a NEW benchmark (2025) - "significant gap in current LLM capabilities"
- No one has fine-tuned specifically for this yet
- Combines your physics interest with ML depth
- Clear, measurable results

### Technical Approach

```
Data Sources:
- AstroPy codebase examples
- Lightkurve library examples
- NASA TESS/Kepler processing scripts
- Published astronomy notebooks from papers

Base Model: Qwen2.5-Coder-7B or DeepSeek-Coder-6.7B

Fine-tuning Method:
1. SFT on astronomy code pairs (natural language -> code)
2. GRPO with execution-based rewards (does code run?)
3. Evaluation on AstroVisBench

Deliverables:
- Model: santo/astrocode-7b on HuggingFace
- Benchmark results beating GPT-4 on astronomy tasks
- Technical report (potential arxiv submission)
- Interactive demo (paste astronomy task, get code)
```

### Benchmark Strategy

AstroVisBench tests:
1. Scientific computing workflows (data processing)
2. Visualization generation (plots, charts)
3. Domain-specific library usage (AstroPy, etc.)

Current SOTA models show "significant gap" - meaning room to beat them.

### GPU Budget: ~$40-60 (3-4 hours A100)

### Target Outcomes
- [ ] Beat GPT-4o on AstroVisBench by 5%+
- [ ] HuggingFace model with 100+ downloads
- [ ] Technical report suitable for workshop submission
- [ ] Gradio demo anyone can try

---

## Project 2: TransitNet - Exoplanet Detection + LLM Explainer

### The Pitch
"I trained a neural network on real NASA TESS data to detect exoplanet transits with 94% accuracy, then built an LLM-powered system that explains each detection to astronomers in natural language."

### Why It's Unique
- **Real ML training**, not just API calls (CNN/Transformer on time series)
- **Multimodal**: combines classical ML with LLM explainability
- Uses **real NASA data** (impressive in interviews)
- Explainability is a hot research area
- Connects to IBM Quantum's interest in physics applications

### Technical Approach

```
Phase 1: Classical ML (Transit Detection)

Data:
- NASA TESS light curves (free via Lightkurve)
- Labeled transiting exoplanet candidates
- Synthetic data augmentation

Model Options:
- 1D CNN (proven for light curves)
- Transformer for time series
- Or use existing model and focus on explainability

Metrics:
- Precision/Recall on known transits
- Compare to existing methods (BLS algorithm)

Phase 2: LLM Explainability

For each detection, generate:
- Natural language explanation of why it's a transit
- Confidence reasoning
- Suggested follow-up observations

Fine-tune small model (Llama 3.2 3B) on:
- Astronomy paper abstracts
- Exoplanet detection explanations
- Domain-specific reasoning chains

Architecture:
Transit Detector (CNN) -> Features -> LLM -> Natural Language Explanation
```

### Deliverables
- Trained transit detection model
- Fine-tuned explainer LLM
- Interactive demo: upload light curve, get detection + explanation
- Dataset contribution (curated TESS examples)
- Blog post or technical report

### GPU Budget: ~$30-50

### Why This Matters for Internships
- Shows **both traditional ML AND LLM skills**
- Demonstrates **end-to-end ML pipeline** (data -> model -> deployment)
- Real scientific application (not toy problem)
- Explainability is hot in ML safety/alignment

---

## Project 3: HPCMind - Intelligent HPC Observability

### The Pitch
"I fine-tuned an LLM specifically for HPC log analysis and incident diagnosis, achieving 85% accuracy on root cause identification versus 60% for GPT-4 - deployed as an MCP server integrated with Prometheus/Grafana."

### Why It's Unique
- **Builds directly on your Pulse project** (leverage existing work)
- **Novel niche** - no one has done this well
- Combines your **systems background with ML depth**
- Perfect for **NVIDIA** (their LLM inference role mentions this)
- Production-ready (not just research)

### Technical Approach

```
Data Collection:
- SLURM job logs (simulate or collect from HPC clusters)
- GPU utilization patterns
- Out-of-memory errors, deadlocks, communication failures
- Prometheus/node_exporter metrics
- Synthetic incident data (you create scenarios)

Dataset Structure:
{
  "logs": "[GPU memory: 39GB/40GB] [NCCL timeout] ...",
  "metrics": {"gpu_util": 0.98, "memory": 0.97, ...},
  "root_cause": "GPU OOM during gradient accumulation",
  "remediation": "Reduce batch size or enable gradient checkpointing"
}

Fine-tuning:
- Base: Qwen2.5-7B or Llama 3.2 3B
- Method: QLoRA + reasoning examples
- Focus: root cause analysis, not just summarization

Integration:
- MCP server for Claude Code
- Grafana plugin or alert webhook
- Prometheus integration

Evaluation:
- Root cause identification accuracy
- Time to diagnosis vs manual analysis
- Comparison to GPT-4/Claude on HPC-specific tasks
```

### Deliverables
- Fine-tuned HPC ops model
- MCP server (Integrate with Claude Code)
- Grafana plugin or webhook handler
- Benchmark dataset for HPC observability
- Demo video showing real incident diagnosis

### GPU Budget: ~$30-50

### Why This Matters for Internships
- **NVIDIA** explicitly wants "LLM Inference Performance Analysis"
- Shows **production engineering** skills (not just notebooks)
- **Novel contribution** (publishable, differentiating)
- Connects to your **existing Pulse project** (coherent narrative)

---

## 12-Week Execution Timeline

### Phase 1: Foundation (Weeks 1-4)

**Week 1: Environment + Learning**
- [ ] Set up Brev with A100
- [ ] Complete DeepLearning.AI GRPO course
- [ ] Clone Unsloth, run "train in 5 minutes" tutorial
- [ ] Read AstroVisBench paper
- [ ] Download TESS data samples via Lightkurve

**Week 2: AstroCode Data Curation**
- [ ] Scrape AstroPy code examples
- [ ] Collect astronomy Jupyter notebooks from papers
- [ ] Create instruction-output pairs for fine-tuning
- [ ] Set up AstroVisBench evaluation locally
- [ ] Start Project 2 data download (TESS light curves)

**Week 3: AstroCode Fine-Tuning**
- [ ] Fine-tune Qwen2.5-Coder-7B on astronomy code
- [ ] Track with MLflow/W&B
- [ ] Run AstroVisBench evaluation
- [ ] Iterate on data quality
- [ ] Begin TransitNet: explore TESS data

**Week 4: AstroCode Completion + TransitNet Start**
- [ ] Final AstroCode training runs
- [ ] Upload to HuggingFace
- [ ] Write model card
- [ ] Build Gradio demo
- [ ] TransitNet: train baseline CNN on TESS data

### Phase 2: TransitNet + Refinement (Weeks 5-8)

**Week 5: TransitNet Core**
- [ ] Train transit detection model (CNN or Transformer)
- [ ] Evaluate on held-out TESS data
- [ ] Compare to BLS baseline
- [ ] Document detection performance
- [ ] Start HPCMind data collection

**Week 6: TransitNet Explainer**
- [ ] Collect astronomy explanations for fine-tuning
- [ ] Fine-tune Llama 3.2 3B as explainer
- [ ] Connect detector -> explainer pipeline
- [ ] Build evaluation metrics for explanation quality

**Week 7: TransitNet Deployment + HPCMind Start**
- [ ] Build TransitNet demo (upload light curve, get result)
- [ ] Deploy on HuggingFace Spaces
- [ ] Write technical documentation
- [ ] HPCMind: curate HPC log dataset
- [ ] HPCMind: define evaluation metrics

**Week 8: HPCMind Fine-Tuning**
- [ ] Fine-tune on HPC log analysis
- [ ] Evaluate root cause accuracy
- [ ] Compare to GPT-4 on HPC tasks
- [ ] Integrate with Prometheus (webhook)

### Phase 3: HPCMind + Polish (Weeks 9-10)

**Week 9: HPCMind Production**
- [ ] Build MCP server for Claude Code integration
- [ ] Create Grafana plugin or alerting integration
- [ ] Record demo video
- [ ] Benchmark and document results

**Week 10: Documentation + Technical Reports**
- [ ] Write AstroCode technical report
- [ ] Write TransitNet blog post
- [ ] Write HPCMind documentation
- [ ] Update all GitHub READMEs
- [ ] Create portfolio page showcasing all three

### Phase 4: Applications (Weeks 11-12)

**Week 11: Resume + Portfolio**
- [ ] Update resume with new projects
- [ ] Record demo videos for each project
- [ ] Prepare talking points for interviews
- [ ] Get resume reviewed
- [ ] Start applications

**Week 12: Applications + Networking**
- [ ] Apply to 20+ companies (prioritized list)
- [ ] Reach out to researchers (AstroMLab team, etc.)
- [ ] Post on LinkedIn about projects
- [ ] Continue iterating based on feedback

---

## Budget Breakdown

| Project | GPU Hours | Cost Estimate |
|---------|-----------|---------------|
| AstroCode fine-tuning | 8h A100 | $25-35 |
| AstroCode experiments | 4h A100 | $12-15 |
| TransitNet detection model | 4h A100 | $12-15 |
| TransitNet explainer | 4h A100 | $12-15 |
| HPCMind fine-tuning | 6h A100 | $18-25 |
| HPCMind experiments | 4h A100 | $12-15 |
| Buffer for iteration | 8h A100 | $25-30 |
| **Total** | **38h** | **$115-150** |

This fits comfortably in your $75-175 budget.

---

## Resume Before/After

### BEFORE (Current)
```
ACADEMIC PROJECTS

Pulse - HPC Cluster Observability Platform
- Built an HPC observability platform with simulated GPU/node telemetry
- Added an LLM-based ops assistant for basic incident investigation

Deep Research Agent - Agentic Research Assistant
- Built an autonomous research pipeline that plans, searches, iteratively refines
- Integrated web retrieval + academic paper search
```

### AFTER (With New Projects)
```
RESEARCH & ML PROJECTS

AstroCode - Scientific Computing Code Generator
- Fine-tuned 7B model to generate astronomy/physics code, outperforming
  GPT-4 on AstroVisBench by X% using QLoRA + GRPO training
- Released open-source model (Y downloads) with interactive demo
- Tech: PyTorch, Transformers, Unsloth, vLLM, HuggingFace

TransitNet - Exoplanet Detection with LLM Explainability
- Trained CNN on 50K+ NASA TESS light curves achieving 94% transit
  detection accuracy, surpassing BLS baseline by 12%
- Built fine-tuned LLM explainer generating natural language reasoning
  for each detection, deployed as interactive astronomy tool
- Tech: PyTorch, Lightkurve, NASA TESS API, Transformers

HPCMind - Intelligent HPC Observability System
- Fine-tuned LLM for HPC log analysis achieving 85% root cause accuracy
  vs 60% for GPT-4 on custom benchmark
- Deployed as MCP server integrated with Prometheus/Grafana for real-time
  incident diagnosis on GPU clusters
- Tech: PyTorch, Prometheus, Grafana, MCP Protocol, vLLM
```

### Key Improvements
1. **Quantified results** (X% accuracy, Y downloads)
2. **Shows model training** (not just API usage)
3. **Benchmarked against frontier models** (beat GPT-4)
4. **Production deployment** (MCP servers, HuggingFace)
5. **Real data** (NASA TESS, HPC clusters)
6. **Coherent narrative** (scientific computing specialist)

---

## Updated Skills Section

### BEFORE
```
AI/ML: LangGraph, RAG, embeddings (pgvector), vLLM, Ollama, scikit-learn
```

### AFTER
```
AI/ML: Model Fine-tuning (QLoRA, GRPO, LoRA), vLLM, PyTorch, Transformers,
       HuggingFace, MLflow, Unsloth, RAG, Embeddings, LangGraph
ML Ops: Model Deployment, Quantization (AWQ/GPTQ), Inference Optimization,
        MCP Servers, Model Evaluation & Benchmarking
```

---

## Interview Talking Points

### For IBM Quantum
"I'm deeply interested in the intersection of physics and AI. I built TransitNet to detect exoplanets using NASA data, and AstroCode to generate scientific computing code. I'd love to apply these skills to quantum computing applications."

### For NVIDIA
"I built HPCMind, a fine-tuned LLM for HPC observability that achieves 85% root cause accuracy on GPU cluster incidents. I integrated it with Prometheus and Grafana for real-time diagnosis. I'm excited about the LLM Inference Performance role because it combines my systems background with ML depth."

### For Google/Research Labs
"I created the first fine-tuned model specifically for astronomy code generation, benchmarked on AstroVisBench. The model beats GPT-4 on domain-specific tasks while being 10x smaller. I'm interested in how specialized small models can advance scientific computing."

### For Scale AI
"I've built three end-to-end ML projects with rigorous evaluation - AstroCode benchmarked on AstroVisBench, TransitNet evaluated on TESS data, and HPCMind with custom HPC diagnostics benchmarks. I understand the importance of systematic evaluation in ML."

---

## Success Metrics

### After 3 Months You Have:

1. **3 HuggingFace models** with downloads and stars
2. **3 interactive demos** anyone can try
3. **3 technical write-ups** (blog or arxiv-style)
4. **Quantified benchmarks** showing beats-larger-models
5. **Resume that stands out** with real ML depth
6. **20+ applications submitted** to strategic targets
7. **Portfolio that tells a story**: "LLMs for Scientific Computing"

### The Differentiator

When a recruiter asks "What makes you different?":

"I don't just use LLMs - I train them. I've fine-tuned three specialized models that outperform GPT-4 in their domains: astronomy code generation, exoplanet detection explanation, and HPC observability. Each one is deployed, benchmarked, and open-sourced. I understand both the model training side and the production engineering side."

---

## Next Steps (This Week)

### Today
- [ ] Set up Brev account
- [ ] Install Unsloth/Axolotl locally for testing
- [ ] Read AstroVisBench paper (https://arxiv.org/html/2505.20538)

### This Week
- [ ] Complete DeepLearning.AI GRPO course
- [ ] Download TESS data samples
- [ ] Set up MLflow for experiment tracking
- [ ] Create GitHub repos for all 3 projects

### Next Week
- [ ] Start AstroCode data curation
- [ ] First fine-tuning experiments

Let's execute.
