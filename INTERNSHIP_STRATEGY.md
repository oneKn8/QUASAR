# Internship Strategy: Small Model Fine-Tuning to Beat 32B+ Models

## Executive Summary

**Goal**: Land ML/AI internship in USA within 3 months
**Budget**: $75-175 in Brev GPU credits
**Current Gap**: Strong systems engineering, weak ML depth (no model training experience)

---

## Part 1: Research Findings - Small Models That Beat Large Ones

### The Key Insight

**"A fine-tuned 3B model beat our 70B baseline"** - This is the paradigm shift. Task-specific fine-tuning of small models consistently outperforms larger general-purpose models.

### Best Small Models for Fine-Tuning (2026)

| Model | Parameters | Why It's Good | Best For |
|-------|------------|---------------|----------|
| **Qwen3-8B** | 8B | Matches Qwen2.5-14B on 15 benchmarks, excellent reasoning | General tasks, math, code |
| **Qwen2.5-3B** | 3B | Matches Qwen2-7B performance, knowledge-dense | Resource-constrained tasks |
| **Phi-4** | 14B | 91.8% on AMC math, beats GPT-4o on STEM | Math, reasoning, STEM |
| **Llama 3.2 1B/3B** | 1-3B | Proven to match GPT-4o after fine-tuning | Edge deployment, specific tasks |
| **CodeLlama-34B** | 34B | Beats GPT-4 on HumanEval after fine-tuning | Code generation |

### Tasks Where Fine-Tuned Small Models Excel

1. **SQL Generation**: Fine-tuned GPT-3.5 beats GPT-4 at 1/3 cost (83.6% accuracy)
2. **Domain-Specific Tasks**: Legal contract review, medical text classification
3. **Sentiment Analysis**: Fine-tuned Llama2 beats FinBERT on finance
4. **Narrow Code Generation**: Domain-specific code (not general programming)
5. **Text Classification**: News headlines, hawkish-dovish classification

### Why Small Models Win on Specific Tasks

- Large models are optimized for breadth, not depth
- Fine-tuning on 100-1000 high-quality examples beats zero-shot on massive models
- Small models can be iterated faster, deployed cheaper
- GRPO enables RL training with <$100 compute

---

## Part 2: Fine-Tuning Techniques You Must Know

### GRPO (Group Relative Policy Optimization)

**Why it matters**: Cuts RL compute in half vs PPO. Works with <100 training examples. Enables reasoning improvements on small models.

**Key insight**: "We can all train reasoning models from our garages by spending less than $100 on cloud GPU services"

**How it works**:
- Samples multiple responses per question
- Calculates relative advantage without separate critic model
- Significantly reduced memory footprint

### PEFT Methods

| Method | Memory Reduction | When to Use |
|--------|------------------|-------------|
| LoRA | 10-100x fewer params | General fine-tuning |
| QLoRA | 4-bit base + LoRA | GPU memory constrained |
| HQQ | No calibration data needed | Quick quantization |

### Model Merging (Advanced)

- Merging multiple fine-tuned models creates emergent capabilities
- "Not merely aggregation, but transformative"
- Works better with larger models (7B+)
- Sweet spots exist in parameter space between specialized models

---

## Part 3: Your Gap Analysis

### What You Have (Strong)
- Systems engineering (observability, Docker, CI/CD)
- Agent orchestration (LangGraph, multi-agent)
- Production experience (SimpliSolve internship)
- Full-stack skills (FastAPI, React, PostgreSQL)

### What You're Missing (Critical)
1. **No model training/fine-tuning experience**
2. **No evaluation/benchmarking work**
3. **No understanding of model internals**
4. **No publications or research contributions**
5. **Projects are "LLM as API" tier**

### What Top Companies Want
- Publications (NeurIPS, ICML, ICLR) - preferred not required for undergrad
- Fine-tuning, RL experience
- Evaluation frameworks
- Agent safety research
- Multi-agent control

---

## Part 4: The Differentiated Project Strategy

### NOT These (Everyone Does Them):
- RAG chatbots
- Document Q&A systems
- AI assistants using Claude/GPT APIs
- Generic coding assistants

### DO This: "Small Model Specialist" - A Complete Fine-Tuning + Evaluation Pipeline

#### Project Name: **FORGE** (Fine-tuned Open-source Reasoning & Generation Evaluator)

#### What Makes It Unique:
1. **You train the models** (not just call APIs)
2. **You evaluate systematically** (not just vibes)
3. **You prove small beats big** (publishable claim)
4. **You deploy efficiently** (edge-ready)

#### The Architecture:

```
FORGE Pipeline:

1. DATA CURATION
   - Identify narrow task (SQL, legal, medical)
   - Curate 500-2000 high-quality examples
   - Create train/val/test splits
   - Document data provenance

2. FINE-TUNING
   - Base: Qwen3-8B or Llama 3.2 3B
   - Method: QLoRA + GRPO for reasoning
   - Track with MLflow/W&B
   - Multiple experimental runs

3. EVALUATION
   - Compare vs GPT-4, Claude, 70B models
   - Multiple metrics (accuracy, latency, cost)
   - Statistical significance tests
   - Create reproducible benchmark

4. DEPLOYMENT
   - Quantize to 4-bit (AWQ/GPTQ)
   - Deploy with vLLM or SGLang
   - Measure inference throughput
   - Edge deployment demo

5. DOCUMENTATION
   - Technical report (arxiv-style)
   - Reproducible code
   - Model on HuggingFace Hub
   - Interactive demo
```

---

## Part 5: Specific Project Recommendations

### Option A: SQL Generation Specialist (Recommended - Highest Impact)

**Why**: Proven domain where small fine-tuned models beat GPT-4. Clear benchmarks exist (Spider, BIRD-SQL).

**The Project**:
1. Fine-tune Qwen3-8B on Spider + BIRD datasets
2. Add GRPO training for query reasoning
3. Beat GPT-4o on BIRD-SQL benchmark (71.83% is current SOTA)
4. Deploy as an MCP server for Claude Code integration

**Deliverables**:
- Model on HuggingFace: `santo/sql-forge-8b`
- Technical report with benchmark results
- Interactive demo (Gradio/Streamlit)
- MCP server for IDE integration

**GPU Budget**: ~$50-80 (4-8 hours on A100)

---

### Option B: Code Review Specialist

**Why**: Combines your systems background with ML. Clear evaluation metrics exist.

**The Project**:
1. Fine-tune on code review datasets (GitHub PRs with human reviews)
2. Train to identify bugs, security issues, style problems
3. Evaluate vs GPT-4/Claude on real codebases
4. Deploy as GitHub Action

**Deliverables**:
- Fine-tuned model with specialized code review capability
- Benchmark against frontier models on code review accuracy
- GitHub Action for automated PR review
- Technical report

---

### Option C: Observability Specialist (Leverages Your Background)

**Why**: Unique niche. No one is doing this. Combines your HPC/observability experience.

**The Project**:
1. Fine-tune on log analysis, metric interpretation
2. Train to diagnose system issues from telemetry data
3. Integrate with your Pulse project
4. Benchmark on real incident datasets

**Deliverables**:
- Model specialized for ops/SRE tasks
- Benchmark dataset for LLM-based observability
- Integration with existing observability tools
- Novel contribution (no existing work in this space)

---

## Part 6: 3-Month Roadmap

### Month 1: Foundation (Weeks 1-4)

**Week 1: Setup & Learning**
- Set up Brev environment with A100
- Complete DeepLearning.AI GRPO course
- Read DeepSeekMath paper (GRPO origin)
- Clone Unsloth/Axolotl repos, run tutorials

**Week 2: Data Curation**
- Choose project (recommend Option A: SQL)
- Download Spider + BIRD-SQL datasets
- Analyze data distribution
- Create 80/10/10 train/val/test split
- Document data preprocessing

**Week 3: First Fine-Tuning Run**
- Fine-tune Qwen3-8B with QLoRA on SQL task
- Track experiments with MLflow
- Evaluate on validation set
- Iterate on hyperparameters

**Week 4: GRPO Training**
- Implement reward function for SQL correctness
- Run GRPO training loop
- Compare SFT vs SFT+GRPO
- Document findings

**Deliverable**: Working fine-tuned model, initial benchmark results

---

### Month 2: Evaluation & Iteration (Weeks 5-8)

**Week 5: Systematic Evaluation**
- Run evaluation on BIRD-SQL test set
- Compare vs GPT-4o, Claude 3.5, Llama 70B
- Measure: accuracy, latency, cost per query
- Statistical significance tests (bootstrap)

**Week 6: Model Optimization**
- Quantize to 4-bit with AWQ
- Measure accuracy vs efficiency tradeoff
- Deploy with vLLM
- Benchmark throughput (tokens/sec)

**Week 7: Advanced Techniques**
- Try model merging (merge SQL + reasoning specialists)
- Experiment with different base models
- Document what works/doesn't

**Week 8: Technical Report**
- Write arxiv-style technical report
- Create visualizations (learning curves, benchmark charts)
- Document methodology rigorously

**Deliverable**: Technical report, optimized model, benchmark results showing beats-larger-models

---

### Month 3: Polish & Deploy (Weeks 9-12)

**Week 9: HuggingFace Release**
- Upload model to HuggingFace Hub
- Write comprehensive model card
- Include training code, configs
- Create usage examples

**Week 10: Demo & Integration**
- Build Gradio demo (natural language to SQL)
- Create MCP server for Claude Code integration
- Deploy demo on HuggingFace Spaces

**Week 11: Portfolio & Resume Update**
- Update resume with project
- Write blog post explaining approach
- Create GitHub README with clear results
- Record demo video

**Week 12: Applications & Networking**
- Apply to target companies (see list below)
- Reach out to researchers in area
- Post on LinkedIn/Twitter about results
- Prepare for technical interviews

**Deliverable**: Complete portfolio piece, applications submitted

---

## Part 7: Target Companies & Roles

### Tier 1: Research-Heavy (Need Strong Results)
- **Scale AI** - ML Research Intern
- **Anthropic** - Research Intern (long shot but worth trying)
- **OpenAI** - Research Intern
- **DeepMind** - Research Intern

### Tier 2: Applied ML (Best Fit for Your Profile)
- **NVIDIA** - LLM Inference Performance (posting exists)
- **Adobe** - AI/ML Intern
- **Qualcomm** - ML/AI Engineering Intern
- **ByteDance** - Seed-LLM-Model Intern
- **d-Matrix** - ML Intern (KV-Cache optimization)

### Tier 3: Startups (Fastest Path to Interesting Work)
- **Predibase** - They literally teach GRPO
- **ThirdLayer** - YC AI startup
- **Oxen.ai** - Model training focused
- **Modal** - ML infrastructure
- **Anyscale/Ray** - Distributed ML

### Tier 4: Big Tech Applied
- **Meta** - PyTorch/FAIR teams
- **Google** - Cloud AI, Gemini teams
- **Microsoft** - Phi team, Azure AI
- **Amazon** - Bedrock, SageMaker

---

## Part 8: Budget Breakdown

### Brev Credits ($75-175)

| Task | GPU | Hours | Cost Estimate |
|------|-----|-------|---------------|
| Initial experiments | A100 40GB | 5h | $10-15 |
| QLoRA fine-tuning | A100 80GB | 8h | $20-25 |
| GRPO training | A100 80GB | 10h | $25-35 |
| Evaluation runs | A100 40GB | 5h | $10-15 |
| Model merging experiments | A100 80GB | 5h | $15-20 |
| Final training runs | A100 80GB | 10h | $30-40 |
| **Total** | | ~43h | **$110-150** |

This fits your budget with buffer for iteration.

---

## Part 9: Key Resources

### Courses (Free)
- [DeepLearning.AI GRPO Course](https://www.deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo/)
- [Hugging Face PEFT Course](https://huggingface.co/learn)

### Libraries
- **Unsloth**: 2x faster fine-tuning, QLoRA optimization
- **Axolotl**: Config-based fine-tuning
- **TRL**: GRPO/DPO/PPO training
- **LM-Eval-Harness**: Standardized benchmarks
- **vLLM**: Fast inference serving

### Papers to Read
1. DeepSeekMath (GRPO introduction)
2. QLoRA paper
3. LoRA original paper
4. Model Merging surveys

### Benchmarks
- **Spider**: Text-to-SQL benchmark
- **BIRD-SQL**: More challenging SQL benchmark
- **HumanEval**: Code generation
- **MBPP**: Python programming

---

## Part 10: What Success Looks Like

### In 3 Months You Should Have:

1. **A fine-tuned model** that demonstrably beats larger models on a specific task
2. **Benchmark results** with statistical rigor
3. **Technical report** suitable for blog or arxiv
4. **HuggingFace release** with proper documentation
5. **Interactive demo** anyone can try
6. **Updated resume** highlighting ML depth
7. **Applications submitted** to 20+ companies

### Interview Talking Points:

"I fine-tuned a 8B parameter model to outperform GPT-4 on SQL generation by X%. I used QLoRA for memory efficiency and GRPO for reasoning improvements. The model achieves Y% accuracy on BIRD-SQL benchmark while running at Z tokens/second on a single A100. I quantized it to 4-bit for edge deployment with minimal accuracy loss. The technical report details my methodology and all code is reproducible."

This is infinitely more impressive than "I built a RAG chatbot."

---

## Next Steps

1. **Today**: Set up Brev account, install Unsloth
2. **This week**: Complete GRPO course, read DeepSeekMath paper
3. **Next week**: Start data curation for chosen project
4. **Week 3**: First fine-tuning run

Let's execute.
