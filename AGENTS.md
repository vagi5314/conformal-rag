# AGENTS.md — Conformal RAG Platform

## Identity
Engineering assistant for the Conformal RAG Platform project. Direct, concrete, concise. Verify before claiming done.

## Project
Production conformal RAG library with FDR-guaranteed hallucination bounds. Python ecosystem, free-tier LLM providers, per-domain calibration.

## Current State (End of Session 2026-08-21)

### Completed — All 10 Critical Fixes + Web UI + Phase 1, 2 & 3 Audit Fixes
- **Phase 3 audit fixes (2026-08-21)** — all three remaining items done, verified live against Qdrant v1.18.0 + Redis containers:
  - **Qdrant path fixed end-to-end** (was broken: no collection creation, no upserts, meaningless sparse indices). Stable sparse encoding via `zlib.crc32(token) % 32768` with real BM25 term weights (index side) / `idf·(1+log tf)` (query side); `_ensure_collection` recreates named vectors `dense`+`sparse`; batched upserts in `index()`; `load_state()` re-populates; search migrated to `query_points`. Opt-in API wiring: env `CONFORMAL_RAG_QDRANT_URL` (+ `CONFORMAL_RAG_QDRANT_COLLECTION`); `/index`+`/retrieve` → 503 when the vector DB is unreachable. docker-compose pins v1.18.0 and fixes its healthcheck (image has no curl).
  - **Embedding domain routing** — `DomainManager(routing="embedding", route_margin=0.55)` default: routes among **calibrated** domains only, argmax cosine vs calibration-query centroids, below-margin or none-calibrated → "general", embedder failure → keyword fallback (`_detect_domain_keyword`). Routing queries persisted to `domains/domain_routing.json`. `/predict` + `client.check()` now route through DomainManager when it can apply a calibrated threshold.
  - **Redis rate limiter** — `RedisRateLimiter` (Lua atomic sliding window, wall clock, lazy connect) selected whenever `CONFORMAL_RAG_REDIS_URL` is set; fails open on Redis errors. In-memory `RateLimiter` remains the zero-config default.
  - **CI** — qdrant:v1.18.0 + redis service containers; live tests (`tests/test_qdrant_live.py`, `tests/test_redis_rate_limit.py`) skip-gated on `CONFORMAL_RAG_QDRANT_TEST_URL`/`CONFORMAL_RAG_REDIS_TEST_URL`.
- **213/215 core tests passing** (2 failures = pre-existing AppLocker `datasets` block), `ruff check src/` clean
- **Phase 2 production hardening (2026-08-16)** — all opt-in via env vars, local dev unchanged: `CONFORMAL_RAG_API_KEY` (X-API-Key auth, /health + frontend exempt), `CONFORMAL_RAG_RATE_LIMIT_PER_MINUTE` (per-IP fixed window, 429), `CONFORMAL_RAG_CORS_ORIGINS` (comma-sep), `CONFORMAL_RAG_STATE_DIR` (JSON persistence of engine calibration + index + domains, restored at startup), `CONFORMAL_RAG_LOG_LEVEL`. Retrieval: e5 `query:`/`passage:` prefixes, embeddings precomputed at index time, module-level embedder cache, index save/load. 21 new tests in `tests/test_hardening.py`.
- **Phase 1 audit fixes (2026-08-16)** — real CRC threshold (Bates et al. 2021 Theorem 1), score direction unified (high=faithful), NLI `id2label` lookup, guards wired (calibration validation + judge-prompt sanitization), API handlers off the event loop, no fake labels in monitoring, LLM timeout passed, `mapie`/`online-cp` deps removed, statistical CRC tests in `tests/test_crc.py`. Guarantee stated honestly: E[P(approved ∧ wrong)] ≤ α.
- **Client SDK** — `index()` builds retriever, `retrieve()` returns context, `check()` risk-checks response, `query()` validates response required — `src/conformal_rag/client.py`
- **Engine wiring** — Accepts optional `FDRMonitor` + `OnlineConformal`, adds `record()` method for live observation — `src/conformal_rag/engine.py:19-42,154-175`
- **True ACI** — Quantile-level Adaptive Conformal Inference (Gibbs & Candes 2021) replacing broken heuristic — `src/conformal_rag/online.py`
- **Live monitoring** — `POST /feedback` endpoint, `/predict` records to `FDRMonitor`, domain drift uses real FDR — `src/conformal_rag/api.py`, `src/conformal_rag/domain.py`
- **Proper conformal CI** — `(n+1)(1-alpha)` finite-sample quantile — `src/conformal_rag/engine.py:182-196`
- **Retrieval fixes** — BM25 built once in `index()` (its stats also weight the Qdrant sparse vectors), e5 instruction prefixes + precomputed embeddings + index persistence (Phase 2), Qdrant collection create/upsert/search fully implemented and live-tested (Phase 3) — `src/conformal_rag/retrieval.py`
- **LLM judge** — Score parsing via regex extraction (was fragile `float()`) — `src/conformal_rag/scoring.py:106-117`
- **NLI scoring** — Entailment index read from `model.config.id2label` (raises if absent); verified ENTAILMENT for `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (label ordering differs from standard MNLI). Fixed in Phase 1 — no longer hardcoded.
- **Reproducible eval** — `benchmarks/eval.py` smoke script

### Pre-existing Issues (Not Fixed)
- 2 benchmark tests fail due to `datasets` DLL block (AppLocker policy): `tests/test_benchmark_hotpotqa.py`. Need different machine or lifted allowlist.
- `scikit-learn==1.4.2` and `scipy==1.14.0` pinned (AppLocker binary hash block)

### Next Steps — Elevation Roadmap
1. ~~**Web UI**~~ — Done. Vanilla JS "Ember Console" frontend in `frontend/` served by FastAPI at `/` (replaced Gradio demo): Check an Answer / Teach It / Live Status / Topics, semicircular SVG support dial. Full API: `/config`, `/samples`, `/index`, `/retrieve`, `/predict`, `/calibrate`, `/record`, `/metrics`, `/online`, `/domains` — `src/conformal_rag/api.py`
2. **HuggingFace Space** — Deploy web UI with shareable URL (2-4h)
3. **Chrome extension** — Highlight text → conformal verdict + coverage guarantee (15-25h)
4. **LangChain/LlamaIndex wrapper** — No existing conformal prediction integration (12-20h)
5. **Run benchmarks** — Need machine without DLL blocks for `--llm-judge` run
6. **Publication** — UncertaiNLP @ EMNLP 2026 deadline (Aug 7, 2026) has PASSED; re-target arXiv or next venue
7. ~~**Audit fixes Phase 1 (2026-08-16)**~~ — Done: real CRC threshold, score direction unified, NLI id2label, guards wired, API off event loop, honest monitoring.
8. ~~**Audit fixes Phase 2 (2026-08-16)**~~ — Done: API auth/rate-limit/CORS, calibration+index+domain persistence, logging config, retrieval productionization (e5 prefixes, precomputed embeddings).
9. ~~**Audit fixes Phase 3 (2026-08-21)**~~ — Done: Qdrant path fixed end-to-end + live-tested, embedding domain routing wired into `/predict`, Redis-backed rate limiter. See HANDOFF.md "Done (2026-08-21 session)".
10. **Frontend charts + UI enhancements** — user-requested, next session: charts in the Ember Console views (Live Status history, calibration coverage, domain metrics) plus general polish.

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
- MAPIE v1 (removed 2026-08-16 — never imported; CRC is hand-rolled per Bates et al. 2021), Qdrant (vector DB)
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
│   ├── api.py              # FastAPI: full REST API + serves frontend/ at /
│   └── client.py           # Python client SDK (real index/retrieve/check)
├── tests/                  # 192 core tests
├── benchmarks/             # 5 plugins + eval.py + profiling
├── frontend/               # Vanilla JS "Ember Console" web UI (served by FastAPI at /)
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

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
