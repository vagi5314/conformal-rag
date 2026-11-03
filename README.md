# conformal-rag

**Production-grade conformal RAG with FDR-controlled hallucination bounds.**

`conformal-rag` wraps any RAG pipeline with Conformal Risk Control (CRC, Bates et al. 2021). Once calibrated on exchangeable labeled data, it provides the finite-sample guarantee **E[P(answer approved AND wrong)] ≤ α** — of all answers it approves, the expected error rate is bounded by your tolerance α. Instead of heuristic confidence scores, you get a provable bound.

> **Honest scope:** the guarantee is the marginal CRC bound above. The conditional claim "P(wrong | approved) ≤ α" is not achievable in general for CRC (see *When Can CRC Certify LLM Outputs?*, 2026). Uncalibrated engines use a default threshold with **no** guarantee and report `calibrated: false`.

## Why This?

Current RAG systems use heuristic thresholds for confidence. No production tool provides mathematically guaranteed hallucination bounds. `conformal-rag` fills this gap:

| Feature | Other Tools | conformal-rag |
|---------|------------|---------------|
| Guarantee type | Heuristic thresholds | E[P(approved ∧ wrong)] ≤ α (Bates et al. 2021) |
| Scoring | Embeddings or basic NLI | Hybrid NLI + LLM judge |
| Calibration | Static | Online adaptive (handles drift) |
| Domain support | Global threshold | Per-domain (demo set: medicine α=0.05, finance and general α=0.15) |
| LLM routing | Single provider | Multi-provider with fallbacks |

## Quick Start

```bash
git clone https://github.com/vagi5314/conformal-rag.git
cd conformal-rag
pip install -e .
```

```python
from conformal_rag.client import ConformalRAG

# Initialize with target FDR (5% hallucination rate)
rag = ConformalRAG(alpha=0.05)

# Index documents
rag.index([
    {"text": "Paris is the capital of France.", "metadata": {"domain": "geography"}},
    {"text": "Python is a programming language.", "metadata": {"domain": "tech"}},
])

# Retrieve context
context = rag.retrieve("What is the capital of France?")

# Risk-check an LLM response
result = rag.check("What is the capital of France?", response="Paris is the capital of France.")
print(result.risk_score)     # 0.02 (calibrated)
print(result.abstained)      # False (risk below threshold)
print(result.risk_bound)     # 0.05 (guaranteed FDR)
```

### Web UI

One command seeds the demo corpus, starts the API, and opens the console:

```bat
run.cmd
```

It opens at http://127.0.0.1:8000. To start the server without the demo state:

```bash
set PYTHONPATH=src
uvicorn conformal_rag.api:create_app --factory --port 8000
```

A dependency-free vanilla JS frontend (Geist and Geist Mono, no build step, no CORS) is served by the same FastAPI process. Four views: **Check** (a question, a source passage and an answer in, a verdict out, with the threshold and the routed topic), **Topics** (one tolerance, threshold and calibration set per topic, plus the sample count each tolerance needs), **Status** (the two honest rates, the calibration score separation, drift per topic) and **Method** (what the bound does and does not claim). Calibration and demo data are pre-seeded, so the console works on the first load.

The Check view has two modes:

- **Paste an answer** — the original flow. Paste a source passage and an answer, pick the tolerance, and the engine returns the verdict against that topic's calibrated threshold.
- **Upload a document** — drop a PDF or text file (up to 10 MB), type a question, and the server retrieves passages from *that* document only, writes the answer, and checks it. Uploads are scored against the `general` calibration unless the request names another topic, because an uploaded document sits outside the demo calibration set and centroid routing would stamp it with a topic from the demo corpus. If the model finds no answer, the response still carries the closest passage so the page says what the document does cover instead of nothing.

```bash
# Upload a document, then ask it a question
curl -X POST http://localhost:8000/documents/upload -F "file=@paper.pdf"
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "<id from the upload>", "query": "What is the coverage gap?"}'
```

Without a provider key the answer is quoted straight from the document and marked grounded by containment; with a key the model writes it and the NLI + judge pipeline scores it.

## How It Works

```
Query + Retrieved Context + LLM Response
              │
              ▼
    ┌─────────────────────┐
    │  NLI Fast Path      │  (~0.4s, free)
    │  nli-deberta-v3-    │  Catches clear cases
    │  small (default)    │
    └──────────┬──────────┘
               │ confidence < 0.7?
               ▼ (30% of cases)
    ┌─────────────────────┐
    │  LLM Judge Path     │  (~1.3s, paid)
    │  via LiteLLM        │  Groq → OpenRouter → Together → Gemini
    └──────────┬──────────┘
               │
               ▼
┌─────────────────────┐
     │  Conformal Engine   │
     │  CRC (Bates 2021)   │  E[P(approved ∧ wrong)] ≤ α
     │  Threshold: 0.5*    │  *default until calibrated
     └──────────┬──────────┘
                │
                ▼
     ┌─────────────────────┐
     │  Approve / Abstain  │  score > threshold → approve
     └──────────┬──────────┘  (ties abstain — required for the bound)
                │
                ▼  true label arrives later
     ┌─────────────────────┐
     │  Monitor + ACI      │  observed FDR, and a feedback
     │  (Gibbs & Candès)   │  controller that re-tunes the threshold
     └─────────────────────┘
               │
               ▼
    Risk Score + Abstention + Confidence Interval
```

### Key Components

| Component | What It Does | Library |
|-----------|-------------|---------|
| **NLI Scorer** | Claim-level faithfulness scoring | `cross-encoder/nli-deberta-v3-small` (default) |
| **LLM Judge** | Deterministic faithfulness verification | LiteLLM (multi-provider) |
| **CRC Engine** | FDR control via conformal prediction | Hand-rolled CRC (Bates et al. 2021) |
| **Online Adaptive** | Handles distribution shift | ACI algorithm |
| **Hybrid Retrieval** | Dense + sparse with RRF | Sentence Transformers + BM25 |
| **Per-Domain Calibration** | Different α per domain | Custom |

### How a claim is scored

The response is split into claims, and each claim is checked against the
context **one sentence at a time and against overlapping three-sentence
windows**, keeping the best match. That is not a detail: the small
cross-encoder is only reliable on a local premise. Measured on a real
document, the same faithful claim scored 0.99 against its three-sentence
paragraph and 0.005 against the whole document, while an unrelated claim
reached 0.02 against the whole document — wrong in both directions, including
a false approval. Single sentences catch verbatim restatements; the windows
keep the context that resolves pronouns.

## Per-Domain Calibration

The competitive edge: no other tool calibrates per domain.

```python
from conformal_rag.client import ConformalRAG

rag = ConformalRAG(alpha=0.05)

# Index domain-specific documents
rag.index([
    {"text": "Apple's P/E ratio is 28.5.", "metadata": {"domain": "finance"}},
    {"text": "Aspirin reduces fever.", "metadata": {"domain": "medical"}},
])

# Retrieve and risk-check per query
result = rag.check("What is the P/E ratio of Apple?", response="28.5")
# Applies calibration for the finance domain
```

## Real-Time Performance

Tested on CPU (no GPU required), `cross-encoder/nli-deberta-v3-small` (the default):

| Component | First Call | Subsequent |
|-----------|-----------|------------|
| NLI Scorer | 10.5s (model load) | 1.1s/query |
| Hybrid Retrieval | 8.7s (embedder load) | <100ms |
| Full Pipeline | 19s (calibration) | 1.8s/predict |

The `small` preset (`cross-encoder/nli-deberta-v3-small`) is ~10x faster on CPU and is what the bundled CLI demo uses. Calibration batches NLI claims through padded forward passes, which is another 5-10x over per-claim calls.

**Scoring behaviour (illustrative, uncalibrated threshold 0.5):**
- Faithful response → faithfulness≈1.00, NOT abstained ✓
- Hallucination → faithfulness≈0.00, ABSTAINED ✓

Note `risk_score` holds *faithfulness*: **higher = more supported**. See "Score Convention" below.

## Score Convention

`risk_score` is a historical name that holds a **faithfulness** score: **higher means the response is better supported by the context.**

- `abstained = (risk_score <= threshold)` — ties abstain, which is what preserves the finite-sample guarantee
- An uncalibrated engine uses a plain threshold of 0.5 and reports `calibrated: false`
- `risk_bound` is α — the promised bound, not a prediction about this response

## API Reference

### Python SDK

```python
from conformal_rag.client import ConformalRAG
from conformal_rag.engine import ConformalRAGEngine

# High-level client
rag = ConformalRAG(alpha=0.05, llm_provider="groq")
rag.index(documents)
context = rag.retrieve("What is...")
result = rag.check("What is...", response="the answer")
print(result.abstained, result.risk_score, result.risk_bound)

# Low-level engine
engine = ConformalRAGEngine(alpha=0.05)
engine.calibrate(queries, contexts, responses, labels)
result = engine.predict(query, context, response)
```

### REST API

```bash
# Start server (also serves the web UI at /)
uvicorn conformal_rag.api:create_app --factory --port 8000

# Predict with risk control
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?", "context": ["Python is a language."], "response": "Python is a snake."}'

# Calibrate
curl -X POST http://localhost:8000/calibrate \
  -H "Content-Type: application/json" \
  -d '{"data": [...], "alpha": 0.05}'

# Health check
curl http://localhost:8000/health

# Upload a document and ask it a question
curl -X POST http://localhost:8000/documents/upload -F "file=@paper.pdf"
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "<id>", "query": "What is the coverage gap?"}'
```

### Production Configuration

All hardening is opt-in via environment variables — with none set, the server
runs exactly as before (no auth, in-memory state):

| Variable | Effect |
|----------|--------|
| `CONFORMAL_RAG_API_KEY` | Require this key in the `X-API-Key` header on every route except `/health` and the static frontend |
| `CONFORMAL_RAG_RATE_LIMIT_PER_MINUTE` | Per-client-IP request limit (429 when exceeded); 0 = unlimited |
| `CONFORMAL_RAG_CORS_ORIGINS` | Comma-separated allowed origins for CORS; empty = none (frontend is same-origin) |
| `CONFORMAL_RAG_STATE_DIR` | Directory for persisting calibration, document index, and domain state as JSON; restored at startup |
| `CONFORMAL_RAG_LOG_LEVEL` | Package log level (default `INFO`) |
| `CONFORMAL_RAG_ALPHA` | Default α (clamped to [0.001, 0.2]) |
| `CONFORMAL_RAG_NLI_MODEL` | NLI model preset: `small` (default) or `large` |
| `CONFORMAL_RAG_LLM_MODELS` | Comma-separated override for the LLM judge fallback chain (default: `groq/qwen/qwen3.8-27b`, `groq/openai/gpt-oss-120b`). Groq's Llama models are retired on the current key. |
| `CONFORMAL_RAG_USE_LLM_JUDGE` | `1` (default) or `0` to disable the paid judge and use NLI only |

```bash
# Example: hardened deploy
CONFORMAL_RAG_API_KEY=*** \
CONFORMAL_RAG_RATE_LIMIT_PER_MINUTE=60 \
CONFORMAL_RAG_STATE_DIR=./state \
uvicorn conformal_rag.api:create_app --factory --port 8000
```

## Benchmarks

| Dataset | What It Tests | Status | FDR (α=0.05) |
|---------|--------------|--------|---------------|
| HotpotQA | Multi-hop reasoning + adversarial injection | ✅ Complete | 0.00% ≤ 5% PASS |
| RAGTruth | Real LLM hallucinations | ✅ Complete (NLI only) | 0.00% ≤ 5% PASS (coverage 0% w/o LLM judge) |
| HaluEval | RLHF hallucinations | ✅ Plugin ready | — (needs LLM judge) |
| FinanceBench | Financial document QA | ✅ Plugin ready | — (needs LLM judge) |
| MultiHop-RAG | Multi-document reasoning | ✅ Plugin ready | — (needs LLM judge) |

Real run: `python -m benchmarks.runner --benchmark hotpotqa --limit 15`  
All 5 plugins: `python -m benchmarks.runner` (with `--llm-judge` for LLM judge)

## Project Structure

```
conformal-rag/
├── src/conformal_rag/
│   ├── __init__.py          # Package exports
│   ├── engine.py            # CRC engine (Bates et al. 2021)
│   ├── scoring.py           # NLI + LLM judge hybrid
│   ├── retrieval.py         # Hybrid dense + sparse retrieval
│   ├── online.py            # Online adaptive conformal
│   ├── llm.py               # LiteLLM multi-provider routing
│   ├── domain.py            # Per-domain calibration
│   ├── monitoring.py        # FDR monitoring + drift detection
│   ├── guards.py            # Security (alpha clamping, query sanitization)
│   ├── models.py            # Pydantic schemas
│   ├── config.py            # Configuration
│   ├── api.py               # FastAPI endpoints
│   ├── client.py            # Python client SDK
│   ├── documents.py         # PDF/text extraction, chunking, answer drafting
│   └── demo_corpus.py       # Seeded demo passages
├── sample_docs/             # Two sample PDFs + verified question list for the upload flow
├── tests/                   # 307 core tests (307 passing, 10 skipped)
├── benchmarks/              # Evaluation scripts
│   ├── runner.py            # Multi-benchmark CLI runner
│   ├── _common.py           # Shared helpers (scoring, injection, aggregation)
│   ├── hotpotqa.py          # HotpotQA with hallucination injection
│   ├── ragtruth.py          # RAGTruth (native LLM hallucinations)
│   ├── financebench.py      # Financial QA plugin
│   ├── halueval.py          # HaluEval plugin
│   ├── multihoprag.py       # Multi-hop reasoning plugin
│   ├── reporting.py         # Markdown report generation
│   └── profiling/           # Latency + cost tracing
├── docs/                    # Specs, plans, research
│   ├── paper/draft.md       # Paper draft (v0.2)
│   └── superpowers/         # Design specs, plans
├── frontend/                # Vanilla JS web UI (served by FastAPI)
│   ├── index.html           # Check (paste + upload) / Topics / Status / Method
│   ├── css/styles.css       # Geist type scale, dark surfaces, motion tokens
│   └── js/                  # api client, view logic, SVG calibration chart
├── archive/                 # Superseded files (quickstart.py, first review deck)
├── scripts/seed_demo.py     # Writes the demo index + per-domain calibrations
├── pyproject.toml           # Package config
├── Dockerfile               # Container config
├── docker-compose.yml       # Local services
└── LICENSE                  # Apache 2.0
```

## Development

```bash
# Install
git clone https://github.com/vagi5314/conformal-rag.git
cd conformal-rag
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Start services (optional)
docker-compose up -d  # Qdrant + Redis + PostgreSQL

# Run server
uvicorn conformal_rag.api:create_app --factory --reload --port 8000
```

## Roadmap

### Phase 1: Core (Complete)
- [x] Real NLI scorer (DeBERTa-v3-large)
- [x] Real LLM judge (litellm integration)
- [x] Correct confidence interval (conformal quantile)
- [x] Correct FDR computation (ACI fix)
- [x] Real hybrid retrieval (in-memory + Qdrant)

### Phase 2: Evaluation (Framework Complete — LLM-judge run PENDING)
- [x] HotpotQA benchmark plugin
- [x] RAGTruth benchmark plugin
- [x] Latency profiling (p50/p95/p99)
- [x] Cost tracking + budget simulation
- [x] 3 additional plugin benchmarks (FinanceBench, HaluEval, MultiHop-RAG)
- [x] Statistical CRC tests on exchangeable synthetic samples (`tests/test_crc.py`)
- [ ] **Benchmark run with `--llm-judge` on real data** — blocked by an AppLocker policy on this machine that blocks the `datasets` DLL. NLI-only runs pass trivially because the model abstains on 100% of RAGTruth. Until this is run, the results section has no real numbers.

### Phase 3: Deployment (Complete)
- [x] Docker multi-stage build (uv, slim, healthcheck)
- [x] docker-compose.yml with full service stack
- [x] Fly.io deployment config
- [x] CI/CD workflows (GitHub Actions, incl. Qdrant + Redis service containers)
- [x] PyPI metadata audit

### Phase 4: Publication
- [x] Paper draft skeleton (IMRaD outline)
- [x] Paper citations updated (2025-2026 papers: Semantic Illusion, Conformal-RAG, TRAQ, C-RAG)
- [ ] arXiv submission (fill LLM-judge results first)
- [ ] Open source launch (after arXiv)

### Phase 5: Elevation
- [x] Web UI (Check / Topics / Status / Method, pre-seeded demo, served by FastAPI)
- [x] Research findings documented (RESEARCH.md updated)
- [ ] HuggingFace Spaces deployment (public shareable URL)
- [ ] Chrome extension (highlight text → conformal verdict)
- [ ] LangChain/LlamaIndex wrapper (ecosystem integration)

## Research

This project is built on solid academic foundations:

- **TRAQ** (NAACL 2024) — First conformal prediction for RAG
- **C-RAG** (ICML 2024) — Certified generation risks
- **Conformal-RAG** (ACM 2025) — Conditional conformal factuality
- **Semantic Illusion** (2025) — Embedding-based detection fails (100% FPR)

See `RESEARCH.md` for full literature review.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Author

VAGI S — SNS College of Engineering, Chennai
