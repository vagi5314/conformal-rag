# AGENTS.md — Conformal RAG Platform

## Identity
Engineering assistant for the Conformal RAG Platform project. Direct, concrete, concise. Verify before claiming done.

## Project
Production conformal RAG library with FDR-guaranteed hallucination bounds. Python ecosystem, free-tier LLM providers, per-domain calibration.

## Current State (End of Session 2026-07-07)

### Completed
- **Phase 1: Core** — All 5 critical fixes implemented and verified
  - Real NLI scorer (DeBERTa-v3-large, 1.1s/query)
  - Real LLM judge (litellm, temperature=0.0)
  - Correct confidence interval (conformal quantile)
  - Correct FDR computation (ACI update fix)
  - Real hybrid retrieval (in-memory + Qdrant)
- **Phase 2: Evaluation** — 5 benchmarks, latency/cost profiling, markdown reporting
  - HotpotQA: FDR=0% ≤ 5% PASS (30 samples, synthetic hallucination injection)
  - RAGTruth: FDR=0% ≤ 5% PASS (native LLM hallucinations, needs LLM judge for coverage)
  - FinanceBench, HaluEval, MultiHop-RAG: all plugin-ready, tests pass offline
  - 53 benchmark tests + latency/cost trace capture
- **Phase 3: Deployment** — Multi-stage Docker (uv), docker-compose with app service, fly.toml, CI/CD workflows, pyproject.toml audited for PyPI
- **Phase 4: Publication** — Paper draft skeleton (IMRaD outline, 5-benchmark eval, ablation studies)
- **Test suite**: 161/161 tests passing
- **Real-time verification**: Faithful→0.9995 (not abstained), Hallucination→0.0004 (abstained)
- **GitHub repo**: https://github.com/vagi5314/conformal-rag (private)
- **License**: Apache 2.0

### Next Session Options
- **Benchmark tuning**: run `--llm-judge` with API keys (Groq/Together/SiliconFlow) to improve coverage on RAGTruth
- **Deployment**: `flyctl launch && flyctl deploy` or `docker compose up`
- **PyPI publish**: trigger `pypi.yml` workflow on tag
- **arXiv submission**: fill paper results with LLM-judge numbers, convert to LaTeX

## Skill Roadmap (MANDATORY — Follow This Sequence Every Session)

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
8. `pr-review` — Before merging anything.

### Phase 4: Security & Deployment
9. `security-and-hardening` — Before deployment.
10. `ci-cd-pipeline-builder` — Automated testing.
11. `shipping-and-launch` — Deployment prep.

### Phase 5: Monitoring
12. `incident-commander` — If production issues arise.
13. `performance-profiler` — Optimize bottlenecks.

## Rules
- NEVER skip the skill roadmap. Every session starts with brainstorming if new work, or continues the sequence.
- NEVER claim "done" without running verification-before-completion.
- NEVER merge without code-review-and-quality.
- Reference code as `file:line`.
- No secrets in output, files, or commits.
- 3 tool errors on same call → stop, summarize, ask.

## Tech Stack
- Python 3.10+, FastAPI, asyncpg, SQLAlchemy 2.0, Redis, PostgreSQL
- MAPIE v1 (CRC), online-cp (adaptive), Qdrant (vector DB)
- LiteLLM (multi-provider routing), DeBERTa-v3 NLI (scoring)
- Sentence Transformers (embeddings), BM25 (sparse), RRF (fusion)
- Docker, Fly.io (deployment), uv (packaging)

## Project Structure
```
conformal-rag/
├── src/conformal_rag/      # Core library (13 modules)
│   ├── engine.py           # CRC engine (MAPIE v1)
│   ├── scoring.py          # NLI + LLM judge hybrid
│   ├── retrieval.py        # Hybrid dense + sparse retrieval
│   ├── online.py           # Online adaptive conformal
│   ├── llm.py              # LiteLLM multi-provider routing
│   ├── domain.py           # Per-domain calibration
│   ├── monitoring.py       # FDR monitoring + drift detection
│   ├── guards.py           # Security (alpha clamping, query sanitization)
│   ├── models.py           # Pydantic schemas
│   ├── config.py           # Configuration
│   ├── api.py              # FastAPI endpoints
│   └── client.py           # Python client SDK
├── tests/                  # 108 tests
├── benchmarks/             # Evaluation scripts (Phase 2)
├── docs/                   # Specs, plans, research
├── pyproject.toml          # Package config (hatchling)
├── Dockerfile              # Container config
├── docker-compose.yml      # Local services (Qdrant, Redis, PostgreSQL)
├── README.md               # Project documentation
└── LICENSE                 # Apache 2.0
```

## Git Branch Strategy
- `dev` — Active development (current)
- `main` — Released versions (after Phase 3)

## Key Files for Next Session
- `docs/superpowers/plans/2026-07-03-phase1-critical-fixes-plan.md` — Completed plan
- `docs/superpowers/specs/2026-07-03-phase1-critical-fixes-design.md` — Completed spec
- `.superpowers/progress.md` — Task tracker (needs update)
- `RESEARCH.md` — Full literature review (644 lines)
- `docs/superpowers/specs/2026-07-03-conformal-rag-design.md` — Original design spec
