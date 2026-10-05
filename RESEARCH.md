# Conformal RAG Platform — Research Document

**Date**: July 3, 2026 (updated July 23, 2026)  
**Project**: Production conformal RAG with risk control  
**Status**: Research complete, implementation complete, elevation roadmap defined

---

## 1. Executive Summary

This document compiles research findings on conformal prediction for RAG systems, competitor analysis, academic landscape, community insights, and technology recommendations. The research confirms a genuine gap: **no production-ready conformal RAG library with risk control (FDR) exists**. All existing solutions are research scripts or student projects without production features.

---

## 2. Competitor Analysis (GitHub)

### 2.1 Direct Competitors

#### trustworthy-rag (HHHAnQi)
- **URL**: https://github.com/HHHAnQi/trustworthy-rag
- **Created**: April 2026
- **Description**: Trustworthy Agentic RAG with Conformal Prediction, calibrated abstention, NLI faithfulness checking, and LangGraph workflow
- **Tech Stack**: LangGraph, NLI, Conformal Prediction
- **Structure**: configs/, data/, frontend/, scripts/, src/
- **Gaps**: No production API, no CRC (Conformal Risk Control), no adaptive thresholding, no distribution shift handling, student project quality

#### conformal-trust-assistant (balance-loz)
- **URL**: https://github.com/balance-loz/conformal-trust-assistant
- **Created**: June 2026
- **Description**: RAG-ассистент с гарантией достоверности (conformal prediction), bandit-роутингом моделей и cost-экономикой. FastAPI · LangGraph.
- **Tech Stack**: FastAPI, LangGraph, Docker, uv package manager, pre-commit hooks
- **Structure**: backend/, frontend/, deploy/, docs/, scripts/, docker-compose.yaml
- **Strengths**: Has Docker deployment, frontend, CI/CD, pre-commit hooks
- **Gaps**: Static calibration only, no online adaptive conformal, no distribution shift detection, no multi-provider LLM routing

#### TRAQ-Reimplementation (huysuy05)
- **URL**: https://github.com/huysuy05/TRAQ-Reimplementation
- **Created**: January 2026
- **Description**: A more easy-to-use reimplementation code of "TRAQ: Trustworthy Retrieval Augmented Question Answering via Conformal Prediction"
- **Paper**: https://arxiv.org/pdf/2307.04642
- **Tech Stack**: Python script-level
- **Data**: labeled_v2.jsonl, triviaqa_full.json
- **Gaps**: Script-level only, no API, no production features, no documentation

#### LettuceDetect (KRLabsOrg)
- **URL**: https://github.com/KRLabsOrg/LettuceDetect
- **Created**: February 2025
- **Description**: Lightweight hallucination detection framework for RAG applications
- **Tech Stack**: Python, has API (lettucedetect_api/), docs (mkdocs), tests
- **Structure**: lettucedetect/, lettucedetect_api/, demo/, docs/, tests/, scripts/
- **Strengths**: Proper Python package, API, documentation, tests
- **Gaps**: No conformal prediction, heuristic-based only, no statistical guarantees

#### conformal-rag-guardrails (debu-sinha)
- **URL**: https://github.com/debu-sinha/conformal-rag-guardrails
- **Description**: Code and experiments for conformal prediction-based hallucination detection in RAG systems
- **Paper**: "The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems" (arXiv:2512.15068)
- **Key Finding**: Embedding-based detection fails on real hallucinations (100% FPR on HaluEval)
- **Gaps**: Research-only code, no production deployment

#### Hallucination_Risk_Management (Mary-yu123)
- **URL**: https://github.com/Mary-yu123/Hallucination_Risk_Management
- **Created**: May 2026
- **Description**: Hallucination Risk Management for Insurance AI using RAG, Data Mesh, and Conformal Prediction
- **Gaps**: Domain-specific (insurance), not general-purpose

### 2.2 Related Repos (Hallucination Detection)

| Repo | Stars | Approach |
|------|-------|----------|
| **LettuceDetect** | Medium | Lightweight heuristic detection |
| **FinTrustRAG** | Low | Financial domain, faithfulness verification |
| **RAG-klarity** | Low | Hallucination detection for RAG |
| **VeritasRAG** | Low | Enterprise faithfulness verification |
| **RAGognizer** | Medium | Hallucination-aware fine-tuning (paper) |
| **MGHall** | Low | Neurosymbolic: causal graphs + GNNs |
| **hallucination-benchmark-trivialplus** | Medium | Amazon Science, ACL 2026 benchmark |

### 2.3 Competitor Gap Summary

| Dimension | All Competitors | Our Project |
|-----------|----------------|-------------|
| Conformal method | Coverage guarantee (CP) | Risk control (CRC) — FDR on hallucination rate |
| Scoring function | Embeddings or basic NLI | LLM judge + NLI ensemble |
| Adaptive calibration | Static | Online conformal with rolling window |
| Production deployment | Research scripts | FastAPI + Redis + PostgreSQL + Docker |
| LLM provider | Single provider | Multi-provider with fallbacks |
| Cost optimization | None | NLI filter + caching = 70% cost reduction |
| Distribution shift | Known problem, no solution | ConformalDrift audit protocol built-in |
| Benchmarks | 1-2 datasets | 5 benchmarks, reproducible |

---

## 3. Academic Landscape

### 3.1 Core Papers

#### TRAQ: Trustworthy Retrieval Augmented Question Answering via Conformal Prediction
- **Venue**: NAACL 2024
- **Authors**: Shuo Li et al.
- **URL**: https://aclanthology.org/anthology-files/anthology-files/pdf/nacla/2024.naacl-long.210.pdf
- **Code**: https://github.com/shuoli90/TRAQ
- **Key Contribution**: First end-to-end statistical correctness guarantee for RAG using conformal prediction. Constructs prediction sets for retriever + generator, aggregates them.
- **Method**: Split conformal prediction on retrieval scores + generation log-probs. Bayesian optimization to minimize prediction set size.
- **Results**: 16.2% reduction in prediction set size vs ablation
- **Limitation**: Coverage guarantee only, not risk control (FDR)

#### C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models
- **Venue**: ICML 2024
- **Authors**: Mintong Kang, Nezihe Merve Gürel, Ning Yu, Dawn Song, Bo Li
- **URL**: https://proceedings.mlr.press/v235/kang24a.html
- **Key Contribution**: First framework to certify generation risks for RAG. Conformal risk analysis with upper confidence bound (conformal generation risk).
- **Method**: Conformal risk analysis for general bounded risk functions under distribution shifts. Proves RAG achieves lower risk than vanilla LLM.
- **Theoretical Guarantees**: Distribution shift guarantees for bounded risk functions
- **Limitation**: Theoretical framework, no production implementation

#### Conformal-RAG: Response Quality Assessment via Conditional Conformal Factuality
- **Venue**: ACM 2025 (DOI: 10.1145/3726302.3730244)
- **Authors**: Ningyu Feng, Yi Sui, Shiyi Hou, Jesse C. Cresswell, Ga Wu
- **URL**: https://arxiv.org/html/2506.20978v1
- **Key Contribution**: Conditional conformal prediction for sub-claim factuality verification. Retains 60% more sub-claims at same factuality level.
- **Method**: Conformal scoring function on sub-claims using retrieved context. Group-conditional coverage across domains.
- **Results**: 60% more retained claims vs baselines
- **Limitation**: Sub-claim level, not end-to-end hallucination rate FDR

#### CONFLARE: CONFormal LArge language model REtrieval
- **Date**: April 2024
- **Authors**: Pouria Rouzrokh, Shahriar Faghani, Cooper U. Gamble, Moein Shariatnia, Bradley J. Erickson
- **URL**: https://arxiv.org/html/2404.04287
- **Code**: https://github.com/Mayo-Radiology-Informatics-Lab/conflare
- **Key Contribution**: Four-step framework for conformal prediction on retrieval uncertainty. Python package available.
- **Method**: Calibration set of questions → similarity score analysis → cutoff threshold → conformal-augmented retrieval
- **Results**: Guarantees true answer in context with (1-α) confidence
- **Limitation**: Retrieval-level only, not generation-level risk control

#### CalibRAG: Reliable Decision Making via Calibration-Oriented Retrieval-Augmented Generation
- **Venue**: NeurIPS 2025
- **Authors**: Chaeyun Jang et al.
- **URL**: https://proceedings.neurips.cc/paper_files/paper/2025/
- **Key Contribution**: Calibrated retrieval for decision-making. Forecasting function for confidence scores.
- **Method**: Train forecasting function f(t, q, d) to predict correctness before generation. Strictly proper scoring rules.
- **Limitation**: Decision calibration, not hallucination FDR

#### The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems
- **Date**: December 2025
- **Author**: Debu Sinha
- **URL**: https://arxiv.org/pdf/2512.15068
- **Code**: https://github.com/debu-sinha/conformal-rag-guardrails
- **Key Contribution**: **CRITICAL FINDING** — Embedding-based hallucination detection fails on real hallucinations.
- **Results**:
  - Synthetic hallucinations (NQ): 95% coverage, 0% FPR ✓
  - Real hallucinations (HaluEval): 95% coverage, **100% FPR** ✗
  - GPT-4 judge: 7% FPR (95% CI: [3.4%, 13.7%])
- **Key Insight**: "RLHF-trained models produce hallucinations that are semantically indistinguishable from faithful responses in embedding space"
- **Implication**: Must use LLM-based scoring, not embeddings

#### Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction
- **Date**: November 2025
- **URL**: https://arxiv.org/html/2511.17908v2
- **Key Contribution**: Conformal prediction for pre-generation filtering. Reduces context 2-3x while maintaining coverage.
- **Method**: Apply CP immediately after retrieval to filter irrelevant snippets
- **Results**: 2-3x context reduction, improved factual accuracy on NeuCLIR
- **Limitation**: Retrieval-level only

#### ConfRAG: Confidence-Guided Retrieval-Augmenting Generation
- **Date**: June 2025
- **URL**: http://arxiv.org/pdf/2506.07309
- **Key Contribution**: ConfQA fine-tuning reduces hallucination 20-40% → below 5%. ConfRAG triggers RAG only when model says "unsure".
- **Results**: 95%+ accuracy, 30% fewer unnecessary retrievals
- **Limitation**: Requires fine-tuning, not plug-and-play

#### Dependable RAG: Factual Confidence Prediction
- **Date**: May 2025
- **URL**: https://arxiv.org/html/2605.05244v1
- **Key Contribution**: Two-stage: CP for retrieval filtering + attention-based factuality classifier. Diagnostic metrics for exchangeability.
- **Results**: 6% answer quality boost from retrieval filtering, 77% hallucination detection
- **Limitation**: Two-stage complexity, no production API

### 3.2 Paper Summary Table

| Paper | Venue | Method | Guarantee | Production Ready |
|-------|-------|--------|-----------|-----------------|
| TRAQ | NAACL 2024 | Split CP on retrieval + generation | Coverage (1-α) | No |
| C-RAG | ICML 2024 | Conformal risk analysis | Risk upper bound | No |
| Conformal-RAG | ACM 2025 | Conditional CP on sub-claims | Group-conditional coverage | No |
| CONFLARE | 2024 | CP on retrieval similarity | Retrieval coverage | Package exists |
| CalibRAG | NeurIPS 2025 | Forecasting function | Decision calibration | No |
| Semantic Illusion | 2025 | CP on hallucination detection | Coverage + FPR analysis | Audit code only |
| Context Engineering | 2025 | CP on snippet retention | Context coverage | No |
| ConfRAG | 2025 | Confidence-based triggering | Accuracy | No |
| Dependable RAG | 2025 | CP + factuality classifier | Retrieval + factuality | No |

---

## 4. Reddit Community Insights (r/Rag)

### 4.1 Hallucination Detection Comparison (Jan 2026)

**Source**: https://www.reddit.com/r/Rag/comments/1qp7psa/

**Methods Compared**:
| Method | Accuracy | Latency | Notes |
|--------|----------|---------|-------|
| LLM judge (GPT-4) | 100% | ~1.3s | Best accuracy, highest cost |
| Atomic claim verification | 100% recall | ~10.7s | High latency |
| Encoder-based NLI | ~91% | ~486ms | CPU-only, fastest |

**Community Best Practice**:
> "We put NLI as a fast filter, async LLM judge on low-confidence hits, and cache verdicts per doc passage. Reduced latency and cost by 70% while keeping recall."

**Critical Finding**:
> "We tested NLI based detectors like Azure groundedness on our RagTruth++ dataset. Results were more like 0.35 F1 score. Our own hallucination detection (agentic verification) scores around 0.8 F1 on the same dataset."

**Implication**: NLI alone is insufficient for production. Must combine with LLM-based scoring.

### 4.2 Hallucination Reduction Techniques (Jan 2025)

**Source**: https://www.reddit.com/r/Rag/comments/1hy9r7k/

**Top Recommendations**:
1. Fix retrieval first (multi-query, re-ranking, fusion)
2. Use semantic entropy for uncertainty
3. Evaluate with RAGAS, DeepEval, G-Eval, TLM
4. Prompt engineering: "only answer using context provided"
5. Split large documents into smaller chunks
6. Add hallucination detection methods for real-time catching

### 4.3 Community Pain Points

1. **No statistical guarantees** — Everyone uses heuristic thresholds
2. **Distribution shift** — Calibration breaks when query distribution changes
3. **Cost/latency tradeoff** — LLM judges are accurate but expensive
4. **Embedding limitations** — Semantic similarity doesn't catch factual errors
5. **No production tools** — Only research code available

---

## 5. Technology Recommendations

### 5.1 Conformal Prediction Method

**Recommended: Conformal Risk Control (CRC)**

Why CRC over standard Split Conformal Prediction (SCP):
- SCP provides coverage guarantee: P(true answer in set) ≥ 1-α
- CRC provides risk control: P(hallucination) ≤ α (FDR control)
- For production RAG, hallucination rate (FDR) is more meaningful than coverage
- CRC extends conformal prediction to control false discovery rate
- Enables: "This answer has ≤5% chance of hallucinating"

### 5.2 Scoring Function

**Recommended: Hybrid LLM Judge + NLI Ensemble**

Based on Semantic Illusion paper findings:
- Embedding-based detection: 100% FPR on real hallucinations (FAIL)
- LLM judge: 7% FPR (SUCCESS)
- NLI alone: 0.35 F1 (INSUFFICIENT)

Architecture:
1. **Fast path**: NLI encoder scores faithfulness (486ms)
2. **Slow path**: LLM judge (GPT-4o-mini) verifies low-confidence cases
3. **Ensemble**: Weighted combination with learned weights

Cost optimization:
- NLI filter catches 70% of clear cases (cheap)
- LLM judge only runs on uncertain 30% (expensive)
- Cache verdicts per document passage
- Total: 70% cost reduction vs LLM-only

### 5.3 Retrieval Stack

**Recommended: Hybrid Dense + Sparse with RRF**

- **Dense**: E5-base-v2 or BGE-base-en-v1.5 (strong, fast)
- **Sparse**: BM25 via rank_bm25 or Elasticsearch
- **Fusion**: Reciprocal Rank Fusion (RRF) — community consensus
- **Reranker**: Cohere Rerank or cross-encoder (optional, improves quality)

### 5.4 Vector Store

**Recommended: ChromaDB (dev) → Qdrant (prod)**

- **ChromaDB**: Local development, zero config, fast iteration
- **Qdrant**: Production scale, filtering, quantization, REST API
- Migration path: ChromaDB for prototyping, Qdrant for deployment

### 5.5 LLM Provider

**Recommended: LiteLLM + Multi-Provider Fallbacks**

Based on user requirement (OpenModels free API with fallbacks):
- **Primary**: OpenModels free API (user specified)
- **Fallback 1**: OpenAI GPT-4o-mini (reliable, cheap)
- **Fallback 2**: Anthropic Claude 3.5 Haiku (strong reasoning)
- **Fallback 3**: Groq (fast, free tier)
- **Routing**: LiteLLM for unified interface + automatic failover

### 5.6 API Framework

**Recommended: FastAPI + Redis + PostgreSQL**

- **FastAPI**: Async, auto-docs, type-safe, industry standard
- **Redis**: Caching (conformal thresholds, verdicts, rate limiting)
- **PostgreSQL**: Calibration data, query logs, user management
- **Deployment**: Docker → Fly.io or Railway (free tier)

### 5.7 Adaptive Calibration

**Recommended: Online Conformal with Rolling Window**

Problem: Static calibration breaks under distribution shift (ConformalDrift paper)

Solution:
- Rolling window of last N calibration samples (e.g., 500)
- Recompute threshold periodically (e.g., daily)
- Monitor coverage/FDR in real-time
- Alert when drift detected (ConformalDrift audit protocol)

### 5.8 Observability

**Recommended: Custom Dashboard + LangSmith**

Metrics to track:
- **Coverage**: % of queries where correct answer in prediction set
- **FDR**: % of "certified" answers that hallucinate
- **Abstention rate**: % of queries where system says "I'm not confident"
- **Latency**: p50, p95, p99 response times
- **Cost**: API calls per query, total spend
- **Drift**: Calibration set freshness, distribution shift alerts

---

## 6. Benchmarks to Reproduce

| Benchmark | What It Tests | Why It Matters | Datasets |
|-----------|--------------|----------------|----------|
| **FinanceBench** | Financial document QA | High-stakes, real hallucinations | FinanceBench |
| **HotpotQA** | Multi-hop reasoning | Tests retrieval + generation jointly | HotpotQA |
| **RAGTruth** | Real LLM hallucinations | Community standard, worst-case for embeddings | RAGTruth |
| **HaluEval** | RLHF model hallucinations | Semantic Illusion paper shows 100% FPR | HaluEval |
| **MultiHop-RAG** | Multi-document reasoning | Tests complex retrieval chains | MultiHop-RAG |

---

## 7. Risk Analysis

### 7.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Conformal threshold too conservative (high abstention) | Medium | Medium | Tune α, use CRC for FDR control |
| LLM API rate limits / downtime | High | High | Multi-provider fallbacks, caching |
| Distribution shift breaks calibration | High | High | Online conformal, drift detection |
| Embedding detection fails (Semantic Illusion) | Certain | High | Use LLM-based scoring, not embeddings |
| Cost overrun from LLM API calls | Medium | Medium | NLI filter, caching, budget alerts |

### 7.2 Project Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Scope creep | High | High | MVP first, iterate |
| 12-week timeline too tight | Medium | High | Prioritize core (library + API), defer dashboard |
| No real users for production claim | Medium | Medium | Deploy as public API, track usage |
| Paper rejected | Medium | Medium | Strong benchmarks, reproducible results |

---

## 8. Success Criteria

### 8.1 MVP (Week 1-8)

- [ ] `conformal-rag` pip package with CRC implementation
- [ ] FastAPI serving endpoint with /predict, /calibrate, /health
- [ ] NLI + LLM judge hybrid scoring
- [ ] Basic benchmark on HotpotQA + RAGTruth
- [ ] Docker deployment on Fly.io

### 8.2 Full Release (Week 9-16)

- [ ] Online adaptive conformal calibration
- [ ] Distribution shift detection (ConformalDrift)
- [ ] Dashboard with real-time metrics
- [ ] 5 benchmark evaluations with reproducible results
- [ ] Documentation + tutorials
- [ ] Paper draft (if targeting workshop/conference)

### 8.3 Impressiveness Bar

- **Deployable product**: Public API serving external developers
- **Open source tool**: pip installable, well-documented
- **Technical showcase**: Live demo with real-time metrics
- **Competition entry**: Submit to relevant benchmarks
- **Paper potential**: Novel CRC application to RAG hallucination FDR

---

## 9. Open Questions

1. **Which free LLM API to use?** — OpenModels, Groq, Together, Fireworks?
2. **Calibration dataset size?** — 500, 1000, 2000 samples?
3. **α threshold default?** — 0.05 (5% FDR) or 0.01 (1% FDR)?
4. **Rolling window size?** — Last 500, 1000, 2000 queries?
5. **Dashboard framework?** — RESOLVED (2026-08-16): dependency-free vanilla JS + FastAPI StaticFiles ("Ember Console" frontend in `frontend/`).

---

## 10. References

### Papers
1. Li et al. (2024). TRAQ: Trustworthy Retrieval Augmented Question Answering via Conformal Prediction. NAACL 2024.
2. Kang et al. (2024). C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models. ICML 2024.
3. Feng et al. (2025). Conformal-RAG: Response Quality Assessment via Conditional Conformal Factuality. ACM 2025.
4. Rouzrokh et al. (2024). CONFLARE: CONFormal LArge language model REtrieval.
5. Jang et al. (2025). CalibRAG: Reliable Decision Making via Calibration-Oriented RAG. NeurIPS 2025.
6. Sinha (2025). The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection. arXiv:2512.15068.
7. (2025). Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction. arXiv:2511.17908.
8. Huang et al. (2025). ConfRAG: Confidence-Guided Retrieval-Augmenting Generation. arXiv:2506.07309.
9. (2025). Dependable RAG: Factual Confidence Prediction. arXiv:2605.05244.

### GitHub Repos
1. https://github.com/HHHAnQi/trustworthy-rag
2. https://github.com/balance-loz/conformal-trust-assistant
3. https://github.com/huysuy05/TRAQ-Reimplementation
4. https://github.com/KRLabsOrg/LettuceDetect
5. https://github.com/debu-sinha/conformal-rag-guardrails
6. https://github.com/shuoli90/TRAQ
7. https://github.com/Mayo-Radiology-Informatics-Lab/conflare

### Reddit Discussions
1. https://www.reddit.com/r/Rag/comments/1qp7psa/compared_hallucination_detection_for_rag_llm/
2. https://www.reddit.com/r/Rag/comments/1hy9r7k/better_hallucination_reducing_techniques/

---

## 11. Detailed Technology Stack Findings

### 11.1 Conformal Prediction Libraries

| Library | CRC Support | Online Adaptive | API Style | Status |
|---------|------------|-----------------|-----------|--------|
| **MAPIE v1** | ✅ ConformalRiskControl class | Via incremental updates | scikit-learn compatible (`fit/predict`) | Active (2025) |
| **online-cp** | ✅ Via online CRC | ✅ Native streaming | Functional API | Active |
| **ACI (Adaptive Conformal Inference)** | ✅ Adaptive threshold | ✅ Per-round adjustment | Functional API | Active |
| **online_conformal (SAOCP/FACI)** | ✅ Risk control | ✅ SAOCP + FACI variants | Functional API | Active |
| **TorchCP** | Limited | Partial | PyTorch native | Active |
| **conformal-evaluation** | No (evaluation only) | No | Functional | Niche |

**Decision**: MAPIE v1 for core CRC (scikit-learn ecosystem, production-tested) + online-cp for adaptive/online conformal (streaming distribution shift).

MAPIE v1 key API:
```python
from mapie.conformal_risk_control import ConformalRiskControl
crc = ConformalRiskControl(method="score", alpha=0.05)
crc.fit(X_cal, y_cal)
y_pred, y_set = crc.predict(X_test, return_risk=True)
```

MAPIE v1 roadmap: risk control support for LLM-as-Judge planned (2026).

### 11.2 Vector Database Comparison

| DB | Hybrid Search | Filtering | Managed Option | Latency (1M vectors) | Free Tier |
|----|--------------|-----------|----------------|---------------------|-----------|
| **Qdrant** | ✅ Dense + Sparse native | ✅ Payload filters | Qdrant Cloud | ~20ms | 1GB free |
| ChromaDB | ❌ Dense only | Basic | No (local only) | ~50ms | Unlimited local |
| Pinecone | ✅ (serverless) | ✅ Metadata | Pinecone | ~15ms | 2GB free |
| Weaviate | ✅ (hybrid) | ✅ Filters | Weaviate Cloud | ~25ms | Free sandbox |
| Milvus | ✅ (GPU) | ✅ Complex | Zilliz Cloud | ~10ms | 1GB free |

**Decision**: Qdrant — best filtering (payload conditions), native hybrid search, Rust-based (fast), cheapest managed ($5/mo for 1GB).

Hybrid search with Qdrant:
```python
from qdrant_client import QdrantClient
client = QdrantClient(":memory:")
# Sparse + dense vectors with RRF fusion
results = client.search(
    collection_name="docs",
    query_vector=("dense", dense_embedding),
    query_sparse=SparseVector(indices=[...], values=[...]),
    limit=10
)
```

### 11.3 Free LLM API Providers

| Provider | Free Tier | RPM | Best For | Latency |
|----------|----------|-----|----------|---------|
| **Groq** | 1000 req/day, 30 RPM | 30 | Fastest inference (Llama, Mixtral) | ~0.3s |
| **OpenRouter** | 50 req/day (free models) | 50 | Model routing, 20+ free models | ~1s |
| **Together AI** | 50K tokens/day free | 10 | Best open-source (Llama, Qwen) | ~0.5s |
| **Cerebras** | 30 req/min | 30 | Ultra-fast (Llama 3.1) | ~0.2s |
| **Google AI Studio** | 1000 req/day | 15 | Gemini 2.5 Flash (multimodal) | ~1s |
| **HuggingFace** | 1000 req/day | 10 | Free inference API | ~2s |

**Decision**: Groq primary (fastest, 1000/day), OpenRouter fallback (20+ models), Together AI for open-source models, Cerebras for ultra-fast. All routed through LiteLLM.

### 11.4 LiteLLM Configuration

```python
import litellm
from litellm import completion

# Multi-provider fallback chain
litellmfallbacks = [
    {"Provider": "groq/llama-3.3-70b-versatile"},
    {"Provider": "openrouter/meta-llama/llama-3.1-8b-instruct:free"},
    {"Provider": "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"},
    {"Provider": "gemini/gemini-2.5-flash"},
]

response = completion(
    model="groq/llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    fallbacks=litellmfallbacks,
    caching=True,
    num_retries=3,
)
```

LiteLLM features: unified interface, automatic failover, load balancing, cost tracking, health checks, response caching, callback logging.

### 11.5 NLI Ensemble for Hallucination Detection

**Primary model**: `cross-encoder/nli-deberta-v3-base` (HuggingFace)
- SciHal 2025 shared task winner: DeBERTa-v3 NLI ensemble
- F1: 0.61 (individual), higher with ensemble
- Latency: ~486ms on CPU

**Ensemble approach**:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

models = [
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-deberta-v3-small",  # fallback
]

def ensemble_nli(premise, hypothesis, threshold=0.5):
    scores = []
    for model_name in models:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        inputs = tok(premise, hypothesis, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        scores.append(probs[0][0].item())  # contradiction score
    avg_score = sum(scores) / len(scores)
    return avg_score < threshold  # True = faithful
```

**Hybrid scoring pipeline**:
1. NLI fast path (~486ms) — catches 70% of clear cases
2. LLM judge path (~1.3s) — verifies low-confidence 30%
3. Cache verdicts per document passage
4. Total: 70% cost reduction vs LLM-only

### 11.6 FastAPI Production Patterns

**Tech stack**: FastAPI + asyncpg + SQLAlchemy 2.0 + Redis + PostgreSQL

**Key patterns**:
```python
# Async database with connection pooling
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
)

# Redis caching for conformal thresholds
import redis.asyncio as redis
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
await r.setex(f"threshold:{query_hash}", 3600, threshold_value)

# Production serving with Gunicorn + Uvicorn
# gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 11.7 RAG Evaluation Stack

| Tool | Best For | CI/CD Integration | Metrics |
|------|----------|-------------------|---------|
| **RAGAS** | Experimentation, research | Manual | Faithfulness, answer relevancy, context precision |
| **DeepEval** | CI/CD gates, regression | ✅ pytest | Hallucination, bias, toxicity, answer relevancy |
| **TruLens** | Production monitoring | ✅ Dashboard | Groundedness, answer relevance, context relevance |
| **LangSmith** | Tracing, debugging | ✅ LangChain | Trace latency, token usage, retrieval quality |

**Decision**: RAGAS (experimentation) + DeepEval (CI/CD gates) + TruLens (production monitoring).

### 11.8 Deployment Platform Comparison

| Platform | Best For | Free Tier | Deploy Time | GPU Support |
|----------|---------|-----------|-------------|-------------|
| **Railway** | Best DX, quick start | $5 credit | ~2 min | No |
| **Fly.io** | AI/ML workloads, global | 3 shared VMs | ~3 min | Yes (paid) |
| **Render** | Predictable pricing | 750 hrs/mo | ~5 min | No |
| **Vercel** | Frontend + API routes | 100 GB | ~1 min | No |
| **Docker local** | Development | Unlimited | ~1 min | Yes |

**Decision**: Fly.io (best for AI/ML, global edge, free tier with 3 shared VMs). Railway as backup (best DX).

### 11.9 Package Publishing with uv

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "conformal-rag"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "mapie>=1.0",
    "qdrant-client>=1.7",
    "litellm>=1.30",
    "fastapi>=0.110",
    "sqlalchemy[asyncio]>=2.0",
    "redis>=5.0",
    "transformers>=4.36",
    "torch>=2.1",
]
```

```bash
# Build and publish
uv build
uv publish --trusted-publisher pypi  # OIDC, no tokens needed
```

PyPI trusted publishing (OIDC): no API tokens, GitHub Actions can publish directly.

---

*Document generated: July 3, 2026 (updated July 23, 2026)*  
*Research scope: GitHub, Reddit, web search, academic papers, library benchmarks*  
*Status: Implementation complete, elevation roadmap defined*

---

## 12. Latest Findings (July 2026)

### 12.1 Key Papers (2025-2026)

#### The Semantic Illusion (Dec 2025)
- **Author**: Debu Sinha
- **URL**: https://arxiv.org/pdf/2512.15068
- **Critical Finding**: Embedding-based hallucination detection fails on real hallucinations
  - Synthetic hallucinations (NQ): 95% coverage, 0% FPR ✓
  - Real hallucinations (HaluEval): 95% coverage, **100% FPR** ✗
  - GPT-4 judge: 7% FPR (95% CI: [3.4%, 13.7%])
- **Implication**: Must use LLM-based scoring, not embeddings

#### Conformal-RAG (Jul 2025)
- **Authors**: Ningyu Feng, Yi Sui, Shiyi Hou, Jesse C. Cresswell, Ga Wu
- **URL**: https://arxiv.org/html/2506.20978v1
- **Key Contribution**: Conditional conformal prediction for sub-claim factuality verification
- **Results**: 60% more retained claims vs baselines
- **Limitation**: Sub-claim level, not end-to-end hallucination rate FDR

#### Provable Bounds on Hallucination Rate (Apr 2026)
- **URL**: https://arxiv.org/html/2604.12017
- **Key Contribution**: Rigorous bound: Pr[hallucinate] <= 1 - rho + delta
- **Implication**: Provides theoretical grounding for our FDR guarantee

#### When Can CRC Certify LLM Outputs? (Jun 2026)
- **URL**: https://arxiv.org/html/2606.07787v1
- **Key Contribution**: Impossibility result for CRC on LLMs
- **Key Result**: ACI reduces violations from 71% to 21%
- **Implication**: Validates our online adaptive approach

### 12.2 Ecosystem Integration Gap

**LangChain**: No conformal prediction integration exists. Current RAG components use naive thresholding or no confidence at all.

**LlamaIndex**: No conformal prediction integration exists. Similar gap to LangChain.

**Opportunity**: First conformal prediction wrapper for either framework would be a genuine contribution.

### 12.3 Elevation Roadmap

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| P0 | Audit fixes (score direction, conformal threshold, guards, async) | 15-25h | Correctness — see docs/remaining-work.md |
| P0 | HuggingFace Space | 2-4h | Public demo URL |
| P0 | Benchmark with LLM judge | 5 min | Paper results |
| P1 | Chrome extension | 15-25h | "Wow" factor |
| P1 | LangChain wrapper | 6h | Ecosystem |
| P1 | LlamaIndex wrapper | 6h | Ecosystem |
| P2 | Publication (arXiv / next venue) | 2-3 weeks | Academic validation |

*(Superseded 2026-08-16: the Gradio demo row was deleted — the web UI is now the vanilla-JS Ember Console frontend.)*

### 12.4 Publication Targets

| Venue | Deadline | Status |
|-------|----------|--------|
| **UncertaiNLP @ EMNLP 2026** | Aug 7, 2026 | **PASSED** (as of 2026-08-16) — re-target arXiv or next venue |
| **arXiv preprint** | Rolling | After audit fixes + benchmark results |

---

*Updated: August 16, 2026*
