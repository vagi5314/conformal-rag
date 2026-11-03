# Conformal RAG Platform — Complete Project Report

**Project:** `conformal-rag` — Production conformal RAG with FDR-guaranteed hallucination bounds
**Author:** Vagi S — SNS College of Engineering, Chennai (`vagi5314@gmail.com`)
**Version:** 0.1.0 · Apache 2.0 · Python ≥3.10,<3.13
**Branch:** `dev` · Repository: `https://github.com/vagi5314/conformal-rag`
**Document purpose:** A single self-contained reference for writing an academic/technical report — what the project is, how it works, what it uses, its results, strengths, weaknesses, and the exact important code.

> **How to read this document.** Section 1–3 give the report-worthy narrative. Section 4–8 are the technical body (architecture, stack, module-by-module code map, data flow, API). Section 9–14 cover the UI, benchmarks, tests, deployment, and configuration. Section 15–18 are the critical evaluation (advantages, disadvantages, honest status, known defects). Section 19–21 are the file index, roadmap, and references. Every claim is tied to a file and line where possible.

---

## 1. What This Project Is

`conformal-rag` is a **Python library + REST API + web console** that wraps any Retrieval-Augmented Generation (RAG) pipeline with **Conformal Risk Control (CRC, Bates et al. 2021)**. It does not generate answers. It takes a query, the retrieved context, and an already-generated LLM response, then decides:

- **Approve** the answer as trustworthy, or
- **Abstain** because the answer may be a hallucination.

Once calibrated on labeled data, it provides the finite-sample statistical guarantee:

$$\mathbb{E}\big[P(\text{answer approved} \;\wedge\; \text{answer wrong})\big] \le \alpha$$

where α is a user-chosen error budget (e.g. 0.05 = 5%). In plain terms: *of all the answers the system approves, the expected rate that turn out wrong is at most α — and that is a mathematical promise, not a heuristic score.*

The project is a final-year engineering project that has been taken well past a prototype: it has a real conformal implementation, a hybrid NLI + LLM scoring cascade, hybrid dense+sparse retrieval, per-domain calibration, online adaptation, a FastAPI server with auth/rate-limiting/persistence, a dependency-free web UI, five benchmark plugins, a ~263-test suite, Docker/Compose/Fly.io deployment, and a paper draft.

### 1.1 The one-sentence pitch

> Standard RAG systems use arbitrary confidence thresholds ("cosine similarity > 0.7") that carry no statistical guarantee. This project replaces the arbitrary threshold with a calibrated one that provably bounds the rate of approved hallucinations.

### 1.2 What is genuinely novel / differentiating

| Dimension | Typical RAG tooling | This project |
|---|---|---|
| Guarantee | Heuristic score / no guarantee | `E[P(approved ∧ wrong)] ≤ α` (finite-sample, distribution-free) |
| Scoring | Embedding similarity or single NLI | NLI fast path + LLM judge for uncertain cases |
| Calibration | Global, static | Per-domain (finance α=0.01, medical α=0.001, etc.) |
| Drift | Not handled | Quantile-level online ACI + FDR monitor + drift flag |
| Retrieval | Dense only | Dense (e5) + sparse (BM25) fused with RRF |
| Deployment | Script | FastAPI + Docker + Redis + Qdrant + auth + persistence |
| Evaluation | 1 dataset | 5 benchmark plugins + latency/cost profiling |

The research document (`RESEARCH.md`) establishes that no production-ready conformal RAG library with FDR control existed at project start — competitors are research scripts or student projects.

---

## 2. The Problem It Solves

RAG reduces hallucination by feeding retrieved documents to an LLM, but it does not eliminate it. An LLM can retrieve the correct Wikipedia article and still state the wrong year. Existing defences fall into three weak categories:

1. **Embedding similarity** between response and context. The paper *The Semantic Illusion* (Sinha, 2025, arXiv:2512.15068) showed embedding-based hallucination detection has a **100% false-positive rate on real hallucinations** (HaluEval), because RLHF-trained models produce hallucinations that are semantically indistinguishable from faithful text.
2. **Single NLI model.** Cheap and fast (~0.5s CPU) but weak on complex reasoning; community reports ~0.35 F1 on RAGTruth.
3. **LLM-as-judge.** Accurate (~7% FPR) but expensive, slow, and shares the generator's parametric biases.

None of these produces a *statistical guarantee*. They output a number; nobody can say what that number means across a deployment. This project supplies the missing layer: a calibrated decision rule with a provable error bound, plus a monitoring loop that tells you when the bound is at risk.

---

## 3. Core Theory: Conformal Risk Control

### 3.1 Why CRC and not plain conformal prediction

Classical split conformal prediction gives a **coverage** guarantee: the true answer is in the prediction set with probability ≥ 1−α. For hallucination control, the meaningful quantity is the **false discovery rate** — of the answers you *accept*, how many are wrong. CRC (Bates et al., 2021, *Distribution-Free Risk Control*) controls a monotone risk directly, which maps cleanly onto "approved ∧ wrong".

### 3.2 The loss and the threshold

Let `s(q,c,r) ∈ [0,1]` be a **faithfulness** score: **higher = more supported by context**. The loss for a calibration point is

$$\mathcal{L}_i(\tau) = \mathbb{1}[y_i = 1]\cdot\mathbb{1}[s_i > \tau]$$

where `y=1` means hallucination. This loss is monotone non-increasing in τ and bounded by B=1 — exactly the condition Bates et al. Theorem 1 requires.

With `n` calibration samples and `m` hallucinations:

$$c_{\max} = \lfloor (n+1)\alpha \rfloor - 1$$

The threshold τ̂ is the **`(c_max + 1)`-th largest hallucination score**. Then at most `c_max` hallucinations can be approved, so

$$P(\text{approved} \wedge y=1) \le \frac{c_{\max}+1}{n+1} = \frac{\lfloor(n+1)\alpha\rfloor}{n+1} \le \alpha.$$

**Decision rule:** approve iff `s > τ̂`; abstain otherwise.

**Why the inequality is strict:** τ̂ is itself a hallucination's score. Approving ties would admit `c_max + 1` hallucinations and break the count. **Ties abstaining is load-bearing**, not a rounding convention.

**Why `(n+1)` not `n`:** the conformal finite-sample correction. The test point is exchangeable with the `n` calibration points and can occupy any of `n+1` rank slots with equal probability.

### 3.3 Degenerate cases (handled explicitly in code)

| Condition | Behaviour | Meaning |
|---|---|---|
| `c_max ≥ m` | τ̂ = 0.0 | Approving everything already meets the bound |
| `c_max < 0` | τ̂ = `nextafter(1.0, 2.0)` | Not enough data to certify α → abstain on everything + warning |
| `n` too small | same as above | Minimum `n ≥ ⌈1/α⌉ − 1` (19 at α=0.05, 99 at α=0.01) |

### 3.4 The guarantee, stated honestly

The project is careful — unusually careful — to state only what is true:

- ✅ Claimed: **marginal** bound `E[P(approved ∧ wrong)] ≤ α`.
- ❌ Not claimed: the **conditional** bound `P(wrong | approved) ≤ α`. The 2026 paper *When Can CRC Certify LLM Outputs?* proves that conditional claim is unachievable in general.
- ❌ Not claimed: coverage. A valid deployment may abstain on most inputs; that is allowed.
- ⚠️ Uncalibrated engines use a plain 0.5 threshold, report `calibrated: false`, and carry **no** guarantee.

### 3.5 Online Adaptive Conformal Inference (ACI)

Offline calibration assumes exchangeability. Real traffic drifts. Following Gibbs & Candès (2021), the project maintains a running quantile level λ_t over a sliding window of recent scores:

$$\lambda_{t+1} = \text{clip}(\lambda_t + \eta(\epsilon_t - \alpha), 0, 1), \qquad \hat\tau_t = Q_{\lambda_t}(W_t)$$

where `ε_t = 1[approved ∧ wrong]`. Too many approved mistakes → raise λ → raise threshold → abstain more. This is explicitly documented as a **heuristic**: it trades away the finite-sample certificate for adaptivity, and its convergence is asymptotic. Code: `src/conformal_rag/online.py:25-78`.

---

## 4. System Architecture

```
Client (browser / Python SDK / curl)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ FastAPI app  (src/conformal_rag/api.py:244 create_app)        │
│  middleware: API-key auth · rate limiting · CORS              │
│  routes: /predict /calibrate /retrieve /index /metrics ...    │
└───────────┬──────────────────────────────────────────────────┘
            │
   ┌────────┴─────────┐
   ▼                  ▼
┌──────────────┐  ┌───────────────────────────────────────────┐
│ Hybrid       │  │ ConformalRAGEngine (engine.py:44)         │
│ Retriever    │  │  ├─ ScoringPipeline (scoring.py:226)      │
│ (retrieval.  │  │  │   ├─ NLIScorer (DeBERTa-v3)           │
│  py:66)      │  │  │   └─ LLMJudge (LiteLLM fallback chain)│
│ dense+sparse │  │  ├─ CRC threshold (engine.py:140)         │
│ + RRF        │  │  ├─ FDRMonitor (monitoring.py:19)         │
└──────┬───────┘  │  └─ OnlineConformal ACI (online.py:25)    │
       │          └───────────────────────────────────────────┘
       ▼                        │
┌──────────────┐          ┌─────┴──────────────┐
│ in-memory    │          │ DomainManager      │
│ or Qdrant    │          │ (domain.py:25)     │
│ (v1.18.0)    │          │ per-domain engines │
└──────────────┘          │ + embedding router │
                          └────────────────────┘
        Optional infra: Redis (rate limit) · PostgreSQL (reserved)
```

**Design principle:** every production feature is **opt-in via environment variables**. With none set, the app runs zero-config locally exactly as a simple demo (no auth, in-memory state, in-memory index).

---

## 5. Technology Stack

### 5.1 Runtime and framework

| Layer | Technology | Why |
|---|---|---|
| Language | Python ≥3.10,<3.13 | Ecosystem for ML/NLP |
| API | FastAPI + Uvicorn | Async, OpenAPI docs, Pydantic validation |
| Schemas | Pydantic v2 | Typed request/response models |
| Packaging | hatchling + `uv` | Fast installs, modern build backend |
| Lint/test | ruff, pytest, pytest-asyncio | CI-gated |

### 5.2 ML / NLP

| Component | Model / Library | Role |
|---|---|---|
| NLI scorer | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (default), `cross-encoder/nli-deberta-v3-small` (fast preset) | Claim-level entailment / faithfulness |
| Dense embedder | `intfloat/e5-base-v2` (768-dim) | Semantic retrieval |
| Sparse retrieval | `rank_bm25` (`BM25Okapi`) | Exact-term/keyword retrieval |
| Fusion | Reciprocal Rank Fusion (k=60) | Merge dense + sparse rankings |
| LLM judge | LiteLLM → Groq → OpenRouter → Together → Gemini | Faithfulness verification on uncertain cases |
| Deep learning | PyTorch + Transformers | Model inference |
| Vector DB | Qdrant v1.18.0 (`qdrant-client`) | Production dense + sparse store |

### 5.3 Infrastructure

| Component | Technology | Role |
|---|---|---|
| Cache / rate limit | Redis (`redis` client) | Atomic Lua sliding-window limiter |
| Database | PostgreSQL 16 + SQLAlchemy 2.0 + asyncpg | Configured/reserved; not yet used by the core path |
| Containers | Docker multi-stage, docker-compose | Local stack + deployment |
| Hosting | Fly.io (`fly.toml`), HuggingFace Spaces (planned) | Deployment targets |
| CI | GitHub Actions (matrix 3.10/3.11/3.12, Qdrant+Redis services) | Lint + tests |
| Publish | `pypi.yml` | PyPI trusted publishing |

### 5.4 Frontend

| Component | Technology | Role |
|---|---|---|
| UI | Vanilla HTML/CSS/JS, no build step | Served by FastAPI at `/` |
| Fonts | Bricolage Grotesque + Space Mono | "Ember Console" visual identity |
| Charts | Hand-written SVG (no chart library) | Support dial, FDR/coverage history |

Dependencies deliberately **removed**: MAPIE and `online-cp` (never imported; CRC is hand-rolled), Gradio (replaced by the vanilla JS UI).

---

## 6. Module-by-Module Breakdown (Important Code)

All paths relative to `src/conformal_rag/`.

### 6.1 `engine.py` — the core (290 lines)

| Symbol | Lines | What it does |
|---|---|---|
| `ConformalRAGEngine.__init__` | 51–68 | Clamps α, builds `ScoringPipeline`, wires `FDRMonitor` + optional `OnlineConformal` |
| `calibrate()` | 74–138 | Validates labels, scores batch, computes CRC threshold + conformal CI, returns `CalibrationStats` |
| `_crc_threshold()` | 140–161 | **The mathematical heart.** Bates et al. Theorem 1 construction (see §3.2) |
| `_conformal_lower_bound()` | 163–176 | One-sided split-conformal lower bound: `k = ⌈(n+1)α⌉`-th smallest score |
| `predict()` | 178–223 | Scores, applies threshold, returns `ConformalResult` |
| `record()` | 225–246 | Feeds labeled feedback to monitor + ACI (updates threshold) |
| `save_state()` / `load_state()` | 250–290 | JSON persistence of calibration |

Key detail: `_ABSTAIN_ON_EVERYTHING = float(np.nextafter(1.0, 2.0))` (line 41) — a value strictly above every legal score, so nothing is approved.

```python
# engine.py:140-161 (abridged)
hallucination_scores = sorted((s for s, lbl in zip(scores, labels) if lbl == 1), reverse=True)
m = len(hallucination_scores)
c_max = int(np.floor((n + 1) * alpha)) - 1
if c_max >= m:   return 0.0                      # approving all is safe
if c_max < 0:    return _ABSTAIN_ON_EVERYTHING   # cannot certify
return hallucination_scores[c_max]               # (c_max+1)-th largest
```

### 6.2 `scoring.py` — hybrid scorer (328 lines)

| Symbol | Lines | What it does |
|---|---|---|
| `_entailment_index()` | 27–41 | Reads entailment class from `model.config.id2label`; **never hardcodes label order** (different NLI checkpoints order labels differently) |
| `NLIScorer` | 44–171 | Lazy model load, process-level cache, sentence decomposition, mean claim score |
| `NLIScorer.score_batch()` | 98–144 | Batched padded forward passes → 5–10× faster calibration on CPU |
| `LLMJudge` | 174–223 | Builds sanitized prompt, calls LLM, regex-parses score |
| `ScoringPipeline` | 226–328 | NLI fast path; escalates to judge only when uncertain; 0.6/0.4 blend |

**Escalation gate:** `nli_confidence = |s_nli − 0.5| × 2`. If `> 0.7` (i.e. clear case), NLI alone is used. Otherwise the LLM judge runs and the final score is `0.6·judge + 0.4·NLI`. The gate confines paid calls to roughly the uncertain middle band.

**Judge parsing hardening** (`scoring.py:211-223`): accepts `0.85`, `85%`, `.85`, bare `0`/`1`; **rejects bare multi-digit integers** (an unconstrained `85` would clamp to 1.0 and approve anything). Unparseable → neutral 0.5, not approval.

### 6.3 `retrieval.py` — hybrid retrieval (311 lines)

| Symbol | Lines | What it does |
|---|---|---|
| `get_embedder()` | 28–34 | Process-wide `SentenceTransformer` cache (loads once) |
| `encode_passages()` / `encode_query()` | 37–50 | Applies e5 `passage:` / `query:` prefixes, L2-normalizes |
| `sparse_slot()` | 53–55 | `zlib.crc32(token) % 32768` — stable across processes |
| `reciprocal_rank_fusion()` | 58–63 | RRF with k=60 |
| `HybridRetriever.index()` | 104–121 | Builds BM25, precomputes dense embeddings, upserts to Qdrant if configured |
| `_doc_sparse_vector()` | 176–197 | Real BM25 term weights as a Qdrant `SparseVector` |
| `_query_sparse_vector()` | 199–222 | `idf·(1+log tf)` for in-vocab tokens; `None` if no match |
| Qdrant methods | 159–265 | Named vectors `dense` + `sparse`, batched upserts, `query_points` |

Why CRC32 and not Python's `hash()`: Python salts string hashes per process, so indices would differ between the indexing process and the query process. CRC32 is deterministic.

### 6.4 `online.py` — ACI (78 lines)

`OnlineConformal.update()` (50–66): appends score to a `deque(maxlen=500)`, computes the error event under the *current* threshold, adjusts the quantile level by `η(error − α)`, then sets the threshold to the λ-quantile of the window. `should_abstain()` (68–70) uses `score < threshold` — the same high=faithful convention as the engine.

### 6.5 `domain.py` — per-domain calibration (298 lines)

| Symbol | Lines | What it does |
|---|---|---|
| `DomainManager` | 25–298 | Holds one engine + monitor per domain |
| `register_domain()` | 51–58 | New engine with its own α |
| `calibrate()` | 60–85 | Stores up to 100 calibration queries for routing; invalidates centroid |
| `_detect_domain()` | 153–184 | Embedding-centroid routing among **calibrated** domains; below `route_margin` (0.55) → `"general"` |
| `_detect_domain_keyword()` | 186–211 | Legacy fallback: finance/medical/legal keyword lists |
| `monitor_drift()` | 213–246 | Flags drift when empirical FDR > 1.5α |
| `save_state()` / `load_state()` | 250–298 | Per-domain JSON + routing state |

Critical design decision: routing candidates are **only calibrated domains**. Routing to an uncalibrated domain would apply the unguaranteed 0.5 default, so a misroute can at worst land on `"general"`, never on a domain certified with the wrong threshold.

### 6.6 `monitoring.py` — FDR monitor (97 lines)

`FDRMonitor` separates two kinds of events:
- `record_prediction(abstained)` — unlabeled prediction; counts totals + abstention rate only.
- `record(predicted_hallucination, actual_hallucination)` — labeled feedback; the **only** source of the false-alarm/FDR metric.

This prevents the earlier bug where `/predict` wrote fake labels and monitoring reported fiction.

### 6.7 `guards.py` — security guards (93 lines)

| Function | Purpose |
|---|---|
| `sanitize_query()` | Redacts prompt-injection patterns (`ignore previous instructions`, `system:`, `return only 1.0`, …) |
| `clamp_alpha()` | Clamps α to `[0.001, 0.2]` |
| `check_calibration_coverage()` | Warns if `n < 100` |
| `detect_pii()` | Email / phone / SSN regex |
| `validate_label_distribution()` | Rejects degenerate labels (all same class, hallucination rate >50% or <1%) |

These are wired, not decorative: `calibrate()` calls validation and coverage checks; `LLMJudge._build_prompt()` sanitizes all three fields.

### 6.8 `llm.py` — multi-provider router (75 lines)

`LLMClient.complete()` iterates the fallback chain, returning the first success; raises only if all fail. `_call_provider()` passes the configured timeout to `litellm.completion` and invokes an optional usage callback (used by benchmark cost tracking).

Default chain: `groq/llama-3.3-70b-versatile` → `openrouter/meta-llama/llama-3.1-8b-instruct:free` → `together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` → `gemini/gemini-2.5-flash`. Temperature 0, `max_tokens=10`, 30s timeout.

### 6.9 `api.py` — FastAPI server (565 lines)

| Symbol | Lines | What it does |
|---|---|---|
| `Settings`-driven app factory | 244 | `create_app(settings)` builds engine, retriever, domain manager |
| `RateLimiter` | 138–155 | In-memory fixed-window per-IP limiter (single process) |
| `RedisRateLimiter` | 177–226 | Atomic Lua sliding window, shared across instances, **fails open** |
| `guard` middleware | 286–305 | Rate limit → 429; API-key → 401; OPTIONS passes for CORS |
| `/health` | 343 | Liveness (auth-exempt) |
| `/config` GET/POST | 347–368 | Read/rebuild engine (α + ACI toggle) |
| `/samples` | 381 | 8 built-in demo documents |
| `/index`, `/documents` | 385–402 | Replace/build the retriever index |
| `/retrieve` | 404–414 | Fused retrieval |
| `/predict` | 418–451 | Main endpoint; domain routing logic |
| `/calibrate`, `/calibration` | 453–475 | CRC calibration + status |
| `/record` | 477–494 | Labeled feedback → monitor + ACI |
| `/metrics`, `/online` | 496–505 | Live monitoring + ACI stats |
| `/domains` + `/{name}/calibrate` + `/{name}/drift` | 509–548 | Per-domain lifecycle |
| `/feedback` | 550–557 | Domain-level labeled feedback |
| `/reset` | 370–377 | Rebuild engine/retriever/domains |
| Static mount | 559–563 | Serves `frontend/` at `/` |

Blocking handlers are plain `def` (not `async def`), so FastAPI runs them on its threadpool instead of stalling the event loop.

### 6.10 `client.py` — Python SDK (140 lines)

`ConformalRAG` is the high-level façade: `index()` → `retrieve()` → `check()` → optional `calibrate()`, `register_domain()`, `monitor_drift()`. `check()` mirrors the API's domain-routing rule. `query()` deliberately **raises** if no `response` is supplied — it refuses to pretend it generates answers.

### 6.11 `models.py` / `config.py`

- `models.py`: Pydantic schemas — `Document`, `CalibrationSample`, `ScoringBreakdown`, `ConformalResult`, `Prediction`, `DriftStatus`, `CalibrationStats`, `Label` enum.
- `config.py`: `Settings` dataclass; all env vars prefixed `CONFORMAL_RAG_`; loaded in `__post_init__`; `from_env()` factory.

---

## 7. End-to-End Data Flow

### 7.1 Calibration (offline, once per domain)

```
labeled samples (query, context, response, label)
   → guards: validate labels, warn if n < 100
   → ScoringPipeline.score_batch()  (NLI batched, or NLI+judge)
   → list of faithfulness scores
   → _crc_threshold(scores, labels, α)      # Bates et al. Theorem 1
   → _conformal_lower_bound(scores, α)       # CI
   → CalibrationStats {threshold, estimated_fdr, coverage, warnings}
   → optionally persisted to CONFORMAL_RAG_STATE_DIR/engine_state.json
```

### 7.2 Inference (online, per request)

```
POST /predict {query, context[], response}
   → join context
   → choose domain engine (explicit / auto-route / main)
   → ScoringPipeline.score()
        → NLI: decompose response into sentences, score each (context, claim) entailment, mean
        → if |s−0.5|·2 ≤ 0.7: LLM judge, final = 0.6·judge + 0.4·NLI
   → abstained = score ≤ threshold        # uncalibrated → 0.5 default
   → ConformalResult {risk_score, risk_bound=α, abstained, calibrated, CI, breakdown, latency}
   → monitor.record_prediction(abstained)
```

### 7.3 Feedback / adaptation

```
POST /record {query, context, response, actual_hallucination}
   → re-score
   → monitor.record(predicted_hallucination, actual_hallucination)
   → online.update(score, is_correct) → threshold = ACI quantile
```

---

## 8. API and SDK Reference

### 8.1 REST endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness (no auth) |
| GET/POST | `/config` | Read / rebuild engine (`alpha`, `online`) |
| POST | `/reset` | Reset all state |
| GET | `/samples` | Built-in demo documents |
| POST | `/index` | Replace document index |
| GET | `/documents` | List indexed documents |
| POST | `/retrieve` | Hybrid retrieval |
| POST | `/predict` | **Main**: conformal verdict |
| POST | `/calibrate` | Calibrate main engine |
| GET | `/calibration` | Last calibration stats |
| POST | `/record` | Labeled feedback (monitor + ACI) |
| GET | `/metrics` | FDR/coverage/abstention |
| GET | `/online` | ACI state |
| GET/POST | `/domains` | List / register domains |
| POST | `/domains/{name}/calibrate` | Calibrate a domain |
| GET | `/domains/{name}/drift` | Drift status |
| POST | `/feedback` | Domain-level labeled feedback |

### 8.2 Python SDK

```python
from conformal_rag.client import ConformalRAG

rag = ConformalRAG(alpha=0.05)
rag.index([
    {"text": "Paris is the capital of France.", "metadata": {"domain": "geography"}},
])
context = rag.retrieve("What is the capital of France?")
result = rag.check("What is the capital of France?", response="Paris is the capital of France.")
print(result.risk_score, result.abstained, result.risk_bound, result.calibrated)
```

Lower-level:

```python
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.online import OnlineConformal

engine = ConformalRAGEngine(alpha=0.05, online=OnlineConformal(alpha=0.05))
engine.calibrate(queries, contexts, responses, labels)
result = engine.predict(query, context, response)
```

### 8.3 Response schema (`ConformalResult`)

| Field | Meaning |
|---|---|
| `risk_score` | **Historical name — holds faithfulness. Higher = better supported.** |
| `risk_bound` | α — the promised bound, not a per-response prediction |
| `abstained` | `risk_score <= threshold` |
| `calibrated` | Whether a CRC threshold exists (else 0.5 default, no guarantee) |
| `confidence_interval` | One-sided conformal lower bound |
| `scoring_breakdown` | NLI score, judge score, final score, latencies, whether judge ran |
| `domain`, `latency_ms`, `timestamp` | Metadata |

> ⚠️ **Naming trap for the report:** `risk_score` is a faithfulness score, not a risk. The README (`README.md:149-155`) documents this explicitly. Do not describe it as "probability of hallucination".

---

## 9. Web UI — "Ember Console"

A dependency-free vanilla-JS frontend in `frontend/`, served by the same FastAPI process at `/` (no CORS, no build step). Warm-charcoal theme (`#14110e` base, ember `#e8833a`, sage `#7ba88f`), Bricolage Grotesque + Space Mono, left sidebar rail.

Four views (`frontend/index.html:36-41`):

1. **Check an Answer** — α slider, ACI toggle, source indexing, query/context/response inputs, verdict card, semicircular SVG support dial with α-threshold tick, scoring breakdown.
2. **Teach It** — calibration/labeling workflow.
3. **Live Status** — report card, FDR/coverage/threshold charts, feedback submission.
4. **Topics** — per-domain registration, calibration, drift.

Files: `frontend/index.html`, `frontend/css/styles.css`, `frontend/js/api.js` (fetch wrapper), `frontend/js/app.js` (view routing/state), `frontend/js/charts.js` (hand-drawn SVG charts).

---

## 10. Benchmarks and Results

### 10.1 Framework

Five plugins under `benchmarks/`:

| Benchmark | What it tests | Hallucination source |
|---|---|---|
| HotpotQA | Multi-hop reasoning | Deterministic entity-substitution injection |
| RAGTruth | Real LLM hallucinations | Human-annotated |
| HaluEval | RLHF hallucinations | Native faithful/hallucinated pairs |
| FinanceBench | Financial document QA | Domain-specific |
| MultiHop-RAG | Multi-document reasoning | Complex reasoning errors |

Shared logic (`benchmarks/_common.py`):
- `inject_hallucination()` picks a different capitalized entity from the same context — the hardest negative for an NLI scorer.
- `aggregate_results()` computes `FDR = accepted_hallucinations / accepted`, coverage, abstention.
- `run_engine_sample()` records NLI/judge/total latency and cost traces.

Runner: `python -m benchmarks.runner [--benchmark NAME] [--llm-judge] [--nli-model large|small] [--limit N] [--simulate-budget]`. Calibration/test split defaults to 80/20 (`benchmarks/config.yaml`).

### 10.2 Actual recorded results (from `benchmarks/reports/`)

| Run | Date | α | Cal/Test | FDR | Coverage | Verdict |
|---|---|---|---|---|---|---|
| HotpotQA (NLI only) | 2026-07-07 | 0.05 | 24 / 6 | 0.00% | 16.67% | PASS |
| RAGTruth (NLI only) | 2026-07-07 | 0.05 | 8 / 2 | 0.00% | **0.00%** | PASS (abstains on everything) |
| HotpotQA (NLI only) | 2026-08-18 | 0.05 | 40 / 10 | **33.33%** | 30.00% | **FAIL** |
| FinanceBench, HaluEval, MultiHop-RAG | — | — | — | — | — | **Not yet run with LLM judge** |

**Honest reading of these numbers — important for the report:**

- The 2026-07-07 runs are tiny (6 and 2 test samples). A 0% FDR there is not evidence of much.
- The 2026-08-18 HotpotQA run **fails the bound** (33.33% > 5%) on 10 samples. With 3 false positives out of 9 accepted, the confidence interval is enormous; this is consistent with either a genuinely weak NLI-only scorer or small-sample noise. It is not a valid demonstration of the guarantee either way.
- RAGTruth "passing" with 0% coverage means the NLI-only scorer abstained on every sample. That is technically within the bound but practically useless — and it is exactly why the hybrid LLM judge exists.
- **No run with `--llm-judge` has been completed.** The blocker is an AppLocker policy on the development machine that blocks the `datasets` DLL. This is stated in `README.md:309`, `HANDOFF.md:70-72`, and `docs/remaining-work.md:43-50`.
- The paper draft's abstract claim that "empirical FDR consistently satisfies the α guarantee across all domains" is **ahead of the evidence** — three of five benchmarks have no numbers and the latest HotpotQA run fails. Do not repeat that claim in a report without qualification.

### 10.3 Latency (recorded)

| Run | NLI p50 | Total p50 | Notes |
|---|---|---|---|
| HotpotQA 2026-07-07 (large, CPU) | 5,914 ms | 12,328 ms | First-generation run, large model |
| HotpotQA 2026-08-18 (CPU) | 921 ms | 921 ms | Small preset, batched |
| RAGTruth 2026-07-07 (large, CPU) | 31,519 ms | 61,079 ms | 2 samples, model load dominated |

README's indicative figures: NLI 1.1 s/query after load; retrieval <100 ms after embedder load; full pipeline 1.8 s/predict. The `small` preset is ~10× faster on CPU.

---

## 11. Test Suite

**~263 test functions across 34 files** in `tests/`. (README's "192" and `docs/remaining-work.md`'s "249" are older counts.)

| Area | Files | Representative tests |
|---|---|---|
| CRC correctness | `test_crc.py` | Threshold is the ranked hallucination score; more data allows more approved errors; sufficient data approves all; too-few samples abstain; **empirical guarantee on 2,000 exchangeable samples**; tie abstention; degenerate-label rejection; record/predict parity |
| Engine | `test_engine.py`, `test_engine_fix.py` | α clamping, calibration, abstention, default threshold, CI |
| Scoring | `test_scoring.py`, `test_nli.py`, `test_llm_judge.py`, `test_scoring_integration.py` | Faithful→high, hallucination→low, batch parity, gate behaviour, judge parsing |
| Retrieval | `test_retrieval.py`, `test_retrieval_real.py`, `test_qdrant_live.py` | CRC32 stability, BM25 formula parity, unknown-token skip, upsert batching, outage tolerance, live Qdrant (6 tests, skip-gated) |
| Domains | `test_domain.py` | Keyword routing, embedding routing, margin fallback, uncalibrated exclusion, centroid refresh, state round-trip |
| Monitoring / online | `test_monitoring.py`, `test_online.py`, `test_online_fix.py` | FDR math, coverage, violation flag, ACI convergence |
| API | `test_api.py` | 22 endpoint tests incl. domain routing |
| Hardening | `test_hardening.py` | Auth (5), rate limiter (5), CORS (2), persistence (4), retrieval prefixes/precompute (5) |
| Redis | `test_redis_rate_limit.py` | Fail-open, per-key isolation, 429, sliding window (skip-gated) |
| Benchmarks | `test_benchmark_*.py` | Plugin loading, injection, aggregation, reporting, runner end-to-end |

Known failures: 2 benchmark tests (and ~17 benchmark/retrieval tests per `HANDOFF.md:71`) fail on the development machine because AppLocker blocks the `datasets` compiled extension. This is environmental, not a code defect.

---

## 12. Deployment

### 12.1 Docker

- Multi-stage `Dockerfile`: builder installs with `uv`, runtime is `python:3.11-slim`, copies site-packages, `src/`, `frontend/`; healthcheck hits `/health`; CMD runs `uvicorn conformal_rag.api:create_app --factory --host 0.0.0.0 --port 8000`.
- `docker-compose.yml`: Qdrant v1.18.0 (custom bash `/dev/tcp` healthcheck — image has no curl), Redis Alpine, PostgreSQL 16, and the app wired to all three via env vars.
- `fly.toml`: 4 GB / 2 shared CPUs, `force_https`, auto-stop/start, London region, α=0.05.

### 12.2 CI

`.github/workflows/ci.yml`: matrix Python 3.10/3.11/3.12; Qdrant + Redis service containers; `ruff check src/ tests/`; `pytest -v -x` with `CONFORMAL_RAG_QDRANT_TEST_URL` / `CONFORMAL_RAG_REDIS_TEST_URL` enabling the live tests.

### 12.3 Configuration (all opt-in)

| Variable | Effect |
|---|---|
| `CONFORMAL_RAG_API_KEY` | Require `X-API-Key` on all routes except `/health` + static UI |
| `CONFORMAL_RAG_RATE_LIMIT_PER_MINUTE` | Per-IP limit; 0 = unlimited |
| `CONFORMAL_RAG_CORS_ORIGINS` | Comma-separated allowed origins |
| `CONFORMAL_RAG_STATE_DIR` | Persist engine calibration + index + domains as JSON |
| `CONFORMAL_RAG_LOG_LEVEL` | Package log level |
| `CONFORMAL_RAG_ALPHA` | Default α (clamped) |
| `CONFORMAL_RAG_QDRANT_URL` / `_COLLECTION` | Enable Qdrant; empty = in-memory |
| `CONFORMAL_RAG_REDIS_URL` | Enables Redis rate limiter |
| `CONFORMAL_RAG_DOMAIN_ROUTE_MARGIN` | Domain routing cosine margin (default 0.55) |
| `CONFORMAL_RAG_POSTGRES_URL` | Reserved for persistence |

---

## 13. Demo / Presentation Assets

| Script | Purpose |
|---|---|
| `panel_demo.py` | 9-step interactive viva demo (retrieval → NLI → CRC → live predict → held-out verification → per-domain → ACI → FDR monitoring → SDK). Pauses between steps; `--auto` to run through. |
| `demo.py` | Verifies the guarantee **in aggregate** on a held-out set; caches NLI scores in `.demo_cache/`; `--dataset tech\|med`, `--alpha`, `--small` (deliberately weak preset to show a violation). |
| `demo_data.py` | Dataset definitions for the demos. |
| `benchmarks/eval.py` | Tiny reproducible smoke eval. |
| `PANEL_PRESENTATION_GUIDE.md` | Script and Q&A for the viva. |

---

## 14. Repository Map

```
Final_year/
├── src/conformal_rag/
│   ├── __init__.py        exports
│   ├── engine.py          CRC engine (Bates 2021) + persistence
│   ├── scoring.py         NLI + LLM judge hybrid
│   ├── retrieval.py       dense + sparse + RRF, Qdrant
│   ├── online.py          quantile-level ACI
│   ├── domain.py          per-domain engines + embedding routing
│   ├── monitoring.py      FDR monitor
│   ├── guards.py          injection sanitizer, α clamp, label validation, PII
│   ├── models.py          Pydantic schemas
│   ├── config.py          env-driven Settings
│   ├── api.py             FastAPI app + auth/rate-limit/persistence
│   └── client.py          Python SDK
├── frontend/              Ember Console (index.html, css/, js/)
├── benchmarks/            5 plugins + runner + profiling + reports/results
├── tests/                 ~263 tests / 34 files
├── docs/                  specs, plans, paper draft, remaining work
├── .github/workflows/     ci.yml, pypi.yml
├── Dockerfile, docker-compose.yml, fly.toml
├── pyproject.toml, uv.lock, LICENSE
├── README.md, RESEARCH.md, HANDOFF.md, AGENTS.md
├── PANEL_PRESENTATION_GUIDE.md
├── panel_demo.py, demo.py, demo_data.py
├── archive/                 superseded files (quickstart.py, first review deck)
└── PROJECT_REPORT.md      (this document)
```

---

## 15. Advantages

1. **A real statistical guarantee, not a score.** The CRC construction is implemented exactly (rank-order threshold, `(n+1)` correction, strict inequality, degenerate-case handling) and tested on 2,000 exchangeable samples.
2. **Intellectual honesty.** The project states the marginal bound, explicitly disclaims the unachievable conditional bound, flags uncalibrated mode, and refuses to certify when data is insufficient (abstain-on-everything).
3. **Cost-aware scoring.** NLI fast path handles clear cases locally/free; the paid judge only runs on the uncertain band. Community-reported ~70% cost reduction motivates the design.
4. **Per-domain calibration.** Different risk appetites for finance/medical/legal are supported, with embedding-based routing restricted to calibrated domains.
5. **Drift awareness.** Online ACI + FDR monitor + 1.5α drift flag give a production feedback loop.
6. **Honest monitoring.** FDR is computed only from labeled feedback, never from assumed labels.
7. **Production hygiene.** Opt-in auth, rate limiting (in-memory and Redis), CORS, request-size limits, JSON persistence, non-blocking handlers, logging.
8. **Hybrid retrieval done properly.** e5 instruction prefixes, precomputed embeddings, CRC32-stable sparse slots, real BM25 weights, live-tested Qdrant path.
9. **Defensive scoring.** Prompt-injection sanitization and judge-output parsing that rejects the "85 → 1.0" approval exploit.
10. **Complete package.** SDK, REST API, web UI, benchmarks, tests, Docker, CI, paper draft, and a viva demo script.
11. **Zero-config local dev.** Every hardening feature is opt-in; `uvicorn` + `PYTHONPATH=src` just works.

---

## 16. Disadvantages and Limitations

### 16.1 Statistical / methodological

1. **Exchangeability assumption.** The finite-sample bound holds only if calibration and test data are exchangeable. Real traffic drifts; ACI adapts but **surrenders the certificate**.
2. **Marginal, not conditional.** It cannot promise "this particular approved answer is ≤α likely wrong". Coverage is uncontrolled — a valid system may abstain on almost everything (RAGTruth NLI-only: 100% abstention).
3. **Calibration data cost.** Needs per-domain labeled data (≥100 recommended, ≥⌈1/α⌉−1 minimum). Labeling is the real deployment bottleneck.
4. **Small-sample instability.** The 2026-08-18 HotpotQA run (33.33% FDR on 10 test samples) shows how noisy tiny evaluations are. The guarantee is an expectation over many draws, not a per-run promise.
5. **NLI scorer ceiling.** DeBERTa-v3 handles simple contradictions but struggles with multi-hop reasoning, omissions, and synthesis — hence the judge, hence the cost.
6. **Judge shares generator biases.** An LLM judging an LLM can approve plausible hallucinations (self-enhancement bias), though prompt sanitization and the 0.6/0.4 blend mitigate it.
7. **ACI is heuristic.** Convergence is asymptotic and assumes slow drift; the code and paper both say so.

### 16.2 Engineering

8. **CPU latency is high.** Large NLI model: seconds per query; RAGTruth run hit ~60 s/sample including model load. Real-time use needs GPU, the small preset, or distillation.
9. **Benchmark evidence is incomplete.** Three of five benchmarks have no LLM-judge results; the `datasets` AppLocker block prevents the authoritative run on this machine.
10. **PostgreSQL is configured but unused** by the core path (SQLAlchemy/asyncpg are dependencies; no models/migrations wired).
11. **Redis rate limiter fails open.** Deliberate (availability over strictness), but worth stating as a security trade-off.
12. **Single-process defaults.** In-memory rate limiter and in-memory index are per-process; multi-instance correctness requires Redis + Qdrant.
13. **Paper draft has stale pseudocode.** §4.3 of `docs/paper/draft.md` still writes `risk_score = 1 − s` and `abstained = risk_score > τ̂`, which contradicts the implemented convention (`risk_score = s`, `abstained = s ≤ τ̂`). The code and README are correct; the draft's inference pseudocode is not.
14. **README results table is optimistic.** It lists HotpotQA "PASS 0.00%" but the newest run fails; treat the table as historical.
15. **No async SDK surface.** Wrapping blocking torch calls in an async framework (LangChain/LlamaIndex) would stall an event loop — noted as a prerequisite in `docs/remaining-work.md:85`.

### 16.3 Scope / project

16. **No public deployment yet.** HuggingFace Space is planned, not done; no shareable URL.
17. **No ecosystem integrations.** No LangChain/LlamaIndex wrapper, no Chrome extension (both roadmap items).
18. **Publication not submitted.** The UncertaiNLP@EMNLP 2026 deadline passed; arXiv is pending benchmark numbers.
19. **Environmental pins.** `scikit-learn==1.4.2` and `scipy==1.14.0` are pinned because AppLocker blocks newer binaries — a portability wart.

---

## 17. Honest Status Summary (for the report's "current state" section)

| Area | Status |
|---|---|
| Core CRC math | ✅ Implemented and statistically tested |
| Scoring pipeline | ✅ Implemented (NLI + judge; trailing edge) |
| Retrieval (in-memory) | ✅ Working |
| Retrieval (Qdrant) | ✅ Implemented, live-tested against v1.18.0 |
| Per-domain calibration + routing | ✅ Implemented (embedding routing default) |
| Online ACI | ✅ Implemented (documented heuristic) |
| FDR monitoring | ✅ Implemented (labeled-only) |
| REST API | ✅ 18 endpoints, auth/rate-limit/CORS/persistence |
| Python SDK | ✅ Implemented |
| Web UI | ✅ Four-view Ember Console |
| Tests | ✅ ~263 tests; 2 benchmark tests blocked by AppLocker |
| Docker / Compose / Fly / CI | ✅ Configured |
| Benchmarks with LLM judge | ❌ Not run (AppLocker `datasets` block) |
| HuggingFace Space | ❌ Not deployed |
| Chrome extension / LangChain wrapper | ❌ Not built |
| arXiv submission | ❌ Pending benchmark numbers |

**Bottom line:** the *software* is complete and defensible; the *empirical validation* is the open gap.

---

## 18. Quick Reference — The Ten Most Important Code Locations

| # | What | Where |
|---|---|---|
| 1 | CRC threshold (the theorem) | `src/conformal_rag/engine.py:140-161` |
| 2 | Approve/abstain decision | `src/conformal_rag/engine.py:202-208` |
| 3 | NLI label-order safety | `src/conformal_rag/scoring.py:27-41` |
| 4 | Escalation gate + 0.6/0.4 blend | `src/conformal_rag/scoring.py:251-294` |
| 5 | Judge score parsing hardening | `src/conformal_rag/scoring.py:211-223` |
| 6 | Sparse slot + RRF | `src/conformal_rag/retrieval.py:53-63` |
| 7 | Qdrant named-vector upsert | `src/conformal_rag/retrieval.py:224-243` |
| 8 | ACI update | `src/conformal_rag/online.py:50-66` |
| 9 | Domain routing among calibrated domains | `src/conformal_rag/domain.py:153-184` |
| 10 | App factory + middleware + routes | `src/conformal_rag/api.py:244-563` |

---

## 19. Roadmap (from `docs/remaining-work.md` and `README.md`)

| Priority | Task | Blocker |
|---|---|---|
| P1 | Run benchmarks with `--llm-judge` and fill paper §6 | Clean machine + API keys |
| P1 | Deploy to HuggingFace Spaces | None (hardening done) |
| P2 | Chrome MV3 extension (highlight → conformal verdict) | Deployed backend |
| P2 | LangChain / LlamaIndex wrappers | Async SDK surface |
| P2 | arXiv submission | Benchmark numbers |
| — | Remove scikit-learn/scipy pins | AppLocker allowlist update |

---

## 20. References (as cited by the project)

1. Bates, S., Angelopoulos, A. N., et al. (2021). *Distribution-Free, Risk-Controlling Prediction Sets.* JACM. — CRC Theorem 1.
2. Gibbs, I., & Candès, E. (2021). *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS.
3. Angelopoulos, A. N., et al. (2024). *Conformal Risk Control.* JMLR.
4. Li, S., et al. (2024). *TRAQ: Trustworthy Retrieval Augmented Question Answering via Conformal Prediction.* NAACL.
5. Kang, M., et al. (2024). *C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models.* ICML.
6. Feng, N., et al. (2025). *Conformal-RAG: Response Quality Assessment via Conditional Conformal Factuality.* ACM.
7. Rouzrokh, P., et al. (2024). *CONFLARE: CONFormal LArge language model REtrieval.*
8. Jang, C., et al. (2025). *CalibRAG: Reliable Decision Making via Calibration-Oriented RAG.* NeurIPS.
9. Sinha, D. (2025). *The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems.* arXiv:2512.15068.
10. (2025). *Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction.* arXiv:2511.17908.
11. (2025). *ConfRAG: Confidence-Guided Retrieval-Augmenting Generation.* arXiv:2506.07309.
12. (2025). *Dependable RAG: Factual Confidence Prediction.* arXiv:2605.05244.
13. (2026). *Provable Bounds on Hallucination Rate.* arXiv:2604.12017.
14. (2026). *When Can CRC Certify LLM Outputs?* arXiv:2606.07787.
15. Niu, J., et al. (2024). *RAGTruth: A Hallucination Corpus for Trustworthy RAG.* ACL.
16. Min, S., et al. (2023). *FactScore.* EMNLP.
17. Manakul, P., et al. (2023). *SelfCheckGPT.* EMNLP.

Full literature review and competitor analysis: `RESEARCH.md`.

---

## 21. Suggested Report Skeleton (map to this doc)

1. **Abstract** — §1 + §3.4 (state the marginal guarantee precisely).
2. **Introduction / motivation** — §2.
3. **Literature review** — §20 + `RESEARCH.md`.
4. **Methodology** — §3 (theory) + §6 (implementation) + §7 (data flow).
5. **System design** — §4, §5, §8.
6. **Implementation details** — §6.1–6.11 with the code table in §18.
7. **Results** — §10, but present the numbers *as recorded*, including the failing 2026-08-18 run and the missing LLM-judge runs. Do not copy the optimistic README table.
8. **Discussion / limitations** — §16.
9. **Conclusion & future work** — §17, §19.
10. **Appendix** — API table (§8.1), env vars (§12.3), repository map (§14).

---

*Generated from a direct read of the source, tests, benchmarks, deployment config, and documentation on 2026-11-01. Where the repository's own docs disagreed with the code, the code was treated as authoritative and the disagreement is noted (see §16.2 items 13–14).*