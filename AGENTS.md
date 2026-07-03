# AGENTS.md — Conformal RAG Platform

## Identity
Engineering assistant for the Conformal RAG Platform project. Direct, concrete, concise. Verify before claiming done.

## Project
Production conformal RAG library with FDR-guaranteed hallucination bounds. Python ecosystem, free-tier LLM providers, per-domain calibration.

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
- Docker, Fly.io (deployment), uv (packaging)

## Project Structure
```
conformal-rag/
├── src/conformal_rag/    # Core library
├── tests/                # TDD tests
├── benchmarks/           # Evaluation
├── docs/                 # Specs, plans, research
└── pyproject.toml        # Package config
```
