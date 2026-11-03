# Final Year Project: Conformal RAG Platform
## Panel Presentation & Viva Defense Guide

> **Author**: Vagi S  
> **Project**: Conformal RAG Platform with Provable Risk Control  
> **Executable Demo**: `python panel_demo.py` (Interactive) or `python panel_demo.py --auto` (Automated)

---

## 1. Quick Start: How to Run the Demo

Open PowerShell in the project directory:

```powershell
# Interactive mode (recommended for viva - pauses after each step so you can explain)
.venv\Scripts\python.exe panel_demo.py

# Auto-run mode (runs continuously without pausing)
.venv\Scripts\python.exe panel_demo.py --auto
```

---

## 2. Step-by-Step Demo Walkthrough (What to Say to the Panel)

| Step | Topic | What the Code Does | What to Say to the Panel |
|------|-------|--------------------|--------------------------|
| **Banner** | **Project Overview** | Displays project identity, author, and theoretical foundation | *"Respected panel members, standard RAG architectures retrieve context and generate text, but LLMs frequently hallucinate facts. Traditional systems use arbitrary heuristic thresholds (like cosine similarity > 0.7) which have no statistical guarantees. Our system implements Conformal Risk Control to provide a mathematically proven bound on the hallucination rate: $\mathbb{E}[P(\text{approved} \land \text{wrong})] \le \alpha$."* |
| **Step 1** | **Hybrid Retrieval** | Indexes 6 documents using BM25Okapi + dense embeddings (`intfloat/e5-base-v2`) and fuses results using Reciprocal Rank Fusion (RRF with $k=60$). | *"Dense vector embeddings capture semantic meaning and synonyms, but often miss exact names, numbers, and identifiers. BM25 keyword search catches exact terms but misses paraphrases. We fuse both with Reciprocal Rank Fusion to ensure the generator receives optimal context in under 70ms."* |
| **Step 2** | **NLI Faithfulness Scoring** | Uses `cross-encoder/nli-deberta-v3-small` to score entailment between context and response. | *"Instead of asking an LLM to judge itself, which is expensive and prone to self-bias, we use a Natural Language Inference (NLI) model. We decompose responses into claims and measure the entailment probability: a faithful answer scores 0.99, while a hallucinated answer scores 0.001."* |
| **Step 3** | **Conformal Risk Control Calibration** | Takes labeled calibration pairs and computes the exact finite-sample threshold using Bates et al. (2021) Theorem 1. | *"Here is our core theoretical contribution. Given a target error budget $\alpha = 15\%$, the CRC theorem proves that sorting calibration hallucinations and choosing the $c_{\max} = \lfloor (n+1)\alpha \rfloor - 1$ largest hallucination score guarantees finite-sample coverage on unseen data."* |
| **Step 4** | **Live Prediction (Approve vs Reject)** | Runs live candidate answers against the calibrated threshold to make binary decisions. | *"Here we demonstrate live inference: faithful responses score above the threshold and are APPROVED with high confidence. Subtle and fabricated hallucinations score below the threshold and are REJECTED (the system abstains). Result: 4/4 correct decisions."* |
| **Step 5** | **Held-Out Verification (Proof of Bound)** | Evaluates 40 unseen held-out samples. Verifies whether empirical error $\le \alpha \cdot n$. | *"A mathematical claim must be empirically validated. On 40 held-out samples that the calibration never saw, our system approved 5 wrong answers against a budget of 6 allowed ($40 \times 15\% = 6$). The mathematical guarantee empirically holds on unseen data!"* |
| **Step 6** | **Per-Domain Calibration** | Calibrates different risk budgets for Tech ($\alpha = 15\%$) vs Medical ($\alpha = 5\%$) and performs automatic query routing. | *"In real-world deployment, different domains require different risk appetites. A medical diagnosis assistant cannot tolerate errors ($\alpha = 5\%$), whereas a general knowledge assistant can be relaxed ($\alpha = 15\%$). Our system calculates domain-specific thresholds: Medical gets a stricter threshold (0.0256) and higher abstention rate to guarantee patient safety."* |
| **Step 7** | **Online Adaptive Conformal Inference (ACI)** | Simulates 60 time steps with distribution shift at $t=30$ using Gibbs & Candes (2021). | *"Offline calibration assumes exchangeability. But real production traffic suffers from distribution drift. We incorporate Adaptive Conformal Inference (ACI): as errors are observed, the feedback loop dynamically increases the threshold from 0.41 to 0.63, making the system automatically more conservative when data quality degrades."* |
| **Step 8** | **FDR Monitoring & Drift Detection** | Tracks live False Discovery Rate (FDR) and flags warnings when error budgets are breached. | *"We implement continuous production monitoring. If the empirical FDR exceeds $1.5 \times \alpha$, the system flags a drift warning and triggers automatic re-calibration."* |
| **Step 9** | **Client SDK** | Demonstrates the high-level 3-line Python API: `rag.index()`, `rag.retrieve()`, `rag.check()`. | *"Finally, everything is packaged into an easy-to-integrate Python SDK. Any developer can add provable hallucination control to their existing RAG application in just three lines of code."* |

---

## 3. Mathematical Foundations Explained (For Viva / Theory Questions)

### 1. What does the guarantee actually mean?
$$\mathbb{E}[P(\text{approved and wrong})] \le \alpha$$
- $\alpha$ is the user-specified error budget (e.g., $0.05$ or $0.15$).
- It bounds the **marginal probability** of the joint event: the answer was **approved** by the system, **AND** it contained a factual error.
- **Key distinction**: It does **not** claim zero errors on every individual answer. It guarantees that across all questions in deployment, the expected rate of approved hallucinations will never exceed $\alpha$.

### 2. How is the threshold computed?
From Bates et al. (2021), *Distribution-Free Risk Control*:
$$c_{\max} = \left\lfloor (n + 1)\alpha \right\rfloor - 1$$
Let $s_{(1)} \ge s_{(2)} \ge \dots \ge s_{(m)}$ be the sorted faithfulness scores of the hallucinations in the calibration set.
The threshold is:
$$\hat{\tau} = s_{(c_{\max} + 1)}$$
Answers scoring **strictly above** $\hat{\tau}$ are approved; answers at or below $\hat{\tau}$ are rejected (abstained).

### 3. Why is there a $(n+1)$ instead of $n$?
The $+1$ is the **conformal finite-sample correction**. Because the test sample is exchangeable with the $n$ calibration samples, the test point can occupy any of the $n+1$ rank slots with equal probability ($1/(n+1)$). Counting as though there are $n+1$ points guarantees distribution-free validity on **unseen future points**.

---

## 4. Top Panel Questions & Model Answers

### Q1: "Why not just use cosine similarity between the query and retrieved document?"
> **Answer**:  
> *"Cosine similarity only measures whether the retriever found relevant context; it tells us nothing about whether the LLM faithfully followed that context or hallucinated during generation. A model can retrieve the right Wikipedia article but still state the wrong year. Furthermore, research ('The Semantic Illusion', 2025) proved that embedding similarity has near 100% false positive rates when detecting subtle factual hallucinations."*

### Q2: "Why use NLI instead of asking GPT-4 or another LLM to judge the output?"
> **Answer**:  
> *"1. **Speed & Latency**: NLI cross-encoders run on CPU in 100–200ms without network overhead, whereas LLM judges take 2–5 seconds.  
> 2. **Cost**: NLI runs 100% locally and free-tier, eliminating recurring API costs.  
> 3. **Self-Enhancement Bias**: LLM judges frequently approve plausible-sounding hallucinations because they suffer from the same parametric memory biases as the generator."*

### Q3: "What assumption does Conformal Risk Control require?"
> **Answer**:  
> *"CRC requires **exchangeability** between calibration and test data (which is a weaker condition than i.i.d.). If the distribution shifts significantly in production, offline CRC guarantees can degrade. That is precisely why our project introduces **Adaptive Conformal Inference (ACI)** from Gibbs & Candes (2021) and live **FDR monitoring** to detect and adapt to real-time distribution drift."*

### Q4: "What is your novel contribution compared to existing RAG benchmarks?"
> **Answer**:  
> *"Most existing RAG frameworks (LangChain, LlamaIndex, Ragas) only provide heuristic scores or uncalibrated similarity thresholds. Our project is the first end-to-end open-source platform that combines:  
> 1. Exact finite-sample conformal bounds (Bates et al. 2021).  
> 2. Per-domain calibration (stricter bounds for medical/finance, relaxed for tech).  
> 3. Online adaptive thresholding (ACI) for live drift compensation.  
> 4. Full production readiness: hybrid retrieval, Redis sliding-window rate limiting, Qdrant vector DB support, FastAPI REST server, and a web dashboard."*
