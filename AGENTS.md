# AGENTS.md — Conformal RAG Platform

## Identity
Engineering assistant for the Conformal RAG Platform project. Direct, concrete, concise. Verify before claiming done.

## Project
Production conformal RAG library with FDR-guaranteed hallucination bounds. Python ecosystem, free-tier LLM providers, per-domain calibration.

## Current State (End of Session 2026-07-23)

### Completed — All 10 Critical Fixes Verified
- **105/105 core tests passing**, `ruff check src/` clean
- **Client SDK** — `index()` builds retriever, `retrieve()` returns context, `check()` risk-checks response, `query()` validates response required — `src/conformal_rag/client.py`
- **Engine wiring** — Accepts optional `FDRMonitor` + `OnlineConformal`, adds `record()` method for live observation — `src/conformal_rag/engine.py:19-42,154-175`
- **True ACI** — Quantile-level Adaptive Conformal Inference (Gibbs & Candes 2021) replacing broken heuristic — `src/conformal_rag/online.py`
- **Live monitoring** — `POST /feedback` endpoint, `/predict` records to `FDRMonitor`, domain drift uses real FDR — `src/conformal_rag/api.py`, `src/conformal_rag/domain.py`
- **Proper conformal CI** — `(n+1)(1-alpha)` finite-sample quantile — `src/conformal_rag/engine.py:182-196`
- **Retrieval fixes** — BM25 built once in `index()`, Qdrant sparse uses actual token weights — `src/conformal_rag/retrieval.py`
- **LLM judge** — Score parsing via regex extraction (was fragile `float()`) — `src/conformal_rag/scoring.py:106-117`
- **NLI scoring** — Verified correct: `probs[0][0]` = ENTAILMENT for `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (label ordering differs from standard MNLI)
- **Reproducible eval** — `benchmarks/eval.py` smoke script

### Pre-existing Issues (Not Fixed)
- 17 benchmark/retrieval tests fail due to `datasets` DLL block (AppLocker policy). Need different machine or lifted allowlist.
- `scikit-learn==1.4.2` and `scipy==1.14.0` pinned (AppLocker binary hash block)

### Next Steps — Elevation Roadmap
1. ~~**Gradio demo**~~ — Done. `demo/app.py` with Predict/Calibrate/Monitor tabs.
2. **HuggingFace Space** — Deploy demo with shareable URL (2-4h)
3. **Chrome extension** — Highlight text → conformal verdict + coverage guarantee (15-25h)
4. **LangChain/LlamaIndex wrapper** — No existing conformal prediction integration (12-20h)
5. **Run benchmarks** — Need machine without DLL blocks for `--llm-judge` run
6. **Publication** — UncertaiNLP @ EMNLP 2026 (deadline Aug 7, 2026)

### Key Papers to Cite (2025-2026)
- **The Semantic Illusion** (Dec 2025) — Embedding-based detection fails on real hallucinations
- **Conformal-RAG** (Jul 2025) — CP + RAG for sub-claim verification
- **Provable Bounds on Hallucination Rate** (Apr 2026) — `Pr[hallucinate] <= 1 - rho + delta`
- **When Can CRC Certify LLM Outputs?** (Jun 2026) — Impossibility result, ACI reduces violations 71%→21%

## Skill Roadmap (MANDATORY)

### Phase 1: Planning
1. `brainstorming` — Before ANY code. Explore directions.
2. `writing-plans` — After design approval. Detailed implementation plan.
3. `spec-driven-development` — Write specs before coding features.

### Phase 2: Implementation
4. `test-driven-development` — RED-GREEN-REFACTOR for every component.
5. `systematic-debugging` — 4-phase root cause, no guessing.

### Phase 3: Quality
6. `verification-before-completion` — Run tests BEFORE claiming done.
7. `code-review-and-quality` — Review across 8 dimensions.

### Phase 4: Deployment
8. `shipping-and-launch` — Deployment prep.

## Rules
- NEVER claim "done" without running verification-before-completion.
- Reference code as `file:line`.
- No secrets in output, files, or commits.
- 3 tool errors on same call → stop, summarize, ask.

## Tech Stack
- Python 3.10+, FastAPI, Pydantic, SQLAlchemy 2.0
- MAPIE v1 (CRC), online-cp (adaptive), Qdrant (vector DB)
- LiteLLM (multi-provider routing), DeBERTa-v3 NLI (scoring)
- Sentence Transformers (e5-base-v2), BM25 (rank-bm25), RRF (fusion)
- Docker, Fly.io, HuggingFace Spaces, uv (packaging)

## Project Structure
```
conformal-rag/
├── src/conformal_rag/      # Core library (13 modules)
│   ├── engine.py           # CRC engine with FDRMonitor + OnlineConformal wiring
│   ├── scoring.py          # NLI + LLM judge hybrid (regex parsing)
│   ├── retrieval.py        # Hybrid dense + sparse (BM25 built once)
│   ├── online.py           # Quantile-level ACI (Gibbs & Candes 2021)
│   ├── llm.py              # LiteLLM multi-provider routing
│   ├── domain.py           # Per-domain calibration + live monitoring
│   ├── monitoring.py       # FDR monitoring + drift detection
│   ├── guards.py           # Security (alpha clamping, PII detection)
│   ├── models.py           # Pydantic schemas
│   ├── config.py           # Configuration
│   ├── api.py              # FastAPI + /feedback endpoint
│   └── client.py           # Python client SDK (real index/retrieve/check)
├── tests/                  # 105 core tests
├── benchmarks/             # 5 plugins + eval.py + profiling
├── demo/                   # Gradio web UI (Predict/Calibrate/Monitor)
├── docs/                   # Specs, plans, research, paper
├── HANDOFF.md              # Session handoff
├── pyproject.toml          # Package config (hatchling)
├── Dockerfile              # Multi-stage Docker
├── docker-compose.yml      # Qdrant + Redis + PostgreSQL
├── fly.toml                # Fly.io deployment
├── README.md               # Documentation
└── LICENSE                 # Apache 2.0
```

## Git Branch Strategy
- `dev` — Active development (current)
- `main` — Released versions

## Key Files
- `HANDOFF.md` — Session handoff for next agent
- `docs/superpowers/plans/2026-07-07-levelup-real-integration-plan.md` — Level-up plan (partially executed)
- `docs/remaining-work.md` — Prioritized task list
- `benchmarks/eval.py` — Reproducible smoke eval
- `RESEARCH.md` — Literature review
