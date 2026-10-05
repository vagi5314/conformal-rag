---
generated_at: 2026-08-21
goal: "Frontend charts + UI enhancements in the Ember Console; HuggingFace Space deploy; benchmarks on a clean machine."
---

# Handoff — 2026-08-21

## Goal of next session

1. **Frontend charts + UI enhancements** (user-requested): charts in the Ember Console views — Live Status history, calibration coverage, domain metrics — plus general polish
2. Deploy the web UI to HuggingFace Spaces for a public shareable URL
3. Run full benchmark suite with `--llm-judge` on a clean machine (no AppLocker DLL blocks)

## State of play

### Done (2026-08-21 session — Phase 3 audit fixes)
All three remaining audit items fixed and verified live against real Qdrant v1.18.0 + Redis containers (`docker compose up -d qdrant redis`):
- **Qdrant path actually works now** — it was broken, not just untested: no collection creation, no upserts, sparse query vectors used positional indices that could never match an indexed representation.
  - Stable sparse encoding: token → slot via `zlib.crc32(token) % 32768`; index-side weights are real BM25 term contributions from the already-built `BM25Okapi`, query-side `idf * (1+log tf)` for in-vocab tokens; empty vector skips the server call.
  - `_ensure_collection()` recreates the collection with named vectors `dense` (dim from embedder, cosine) + `sparse`; `_upsert_all()` batches PointStructs (id=i, payload text). `index()` always builds BM25 (sparse weights need its idf/doc stats).
  - Search migrated off deprecated `client.search()` to `query_points(..., using="dense"/"sparse")`.
  - API wiring: new `Settings.qdrant_url` (env `CONFORMAL_RAG_QDRANT_URL`, empty = in-memory) + `qdrant_collection` (`CONFORMAL_RAG_QDRANT_COLLECTION`); `_build_retriever(settings)` used in `create_app`, `/index`, `/reset`. `/index` + `/retrieve` return HTTP 503 on connection errors (no silent fallback by design). docker-compose pins `qdrant/qdrant:v1.18.0`, passes `CONFORMAL_RAG_QDRANT_URL`, and fixes the healthcheck (image ships no curl — bash /dev/tcp probe instead).
  - `load_state()` re-populates Qdrant after restore (covers server restarts), wrapped so startup never crashes when the server is down.
- **Embedding-based domain routing** — replaces keyword substring matching as default (`routing="embedding"`; `"keyword"` opt-out via constructor):
  - Candidates = **calibrated domains only** (an uncalibrated domain would apply the unguaranteed 0.5 default); argmax cosine vs per-domain centroid of calibration-query embeddings; below `route_margin` (default 0.55, env `CONFORMAL_RAG_DOMAIN_ROUTE_MARGIN`) or no calibrated candidates → "general"; embedder load failure → keyword fallback with warning.
  - `DomainManager.calibrate()` captures queries (engine keeps only scores/labels); centroid cached, invalidated on recalibration; domain-name embedding until calibrated. Lazy model load — `DomainManager()` stays zero-cost to construct.
  - Routing state persisted in `domains/domain_routing.json` (capped 100 queries × 400 chars); old state dirs still load (name-only routing until next calibration).
  - **`/predict` and `client.check()` now route through DomainManager** (previously zero production call sites): explicit registered domain → that domain's engine (its calibrated threshold); unregistered domain → main engine with tag (unchanged); no explicit domain + ≥1 calibrated domain registered → auto-routing; else main engine exactly as before. Domain monitor counts abstentions via `record_prediction`.
- **Redis-backed rate limiter** — `RedisRateLimiter` in api.py, same interface as `RateLimiter`: atomic Lua sliding window (ZREMRANGEBYSCORE → ZCARD → conditional ZADD+EXPIRE), wall-clock time, lazy `redis.Redis.from_url` (2s timeouts). Selected whenever `settings.redis_url` is set (in-memory otherwise — all hermetic tests unaffected). **Fails open** on Redis errors (warn + allow): rate limiting is best-effort, an outage must not take the API down.
- **CI service containers** — `.github/workflows/ci.yml` adds qdrant:v1.18.0 + redis services; skip-gated live tests run there via `CONFORMAL_RAG_QDRANT_TEST_URL` / `CONFORMAL_RAG_REDIS_TEST_URL`.
- **Tests**: retrieval unit tests (crc32 stability, BM25 weight formula parity, unknown-token skip, upsert batching, outage-tolerant load_state); domain routing tests (keyword pinned, fake-embedder embedding routing, margin fallback, uncalibrated exclusion, routing round-trip); API predict-routing tests; `tests/test_qdrant_live.py` + `tests/test_redis_rate_limit.py` (socket-probe skip gating). Live runs verified locally: 6/6 qdrant, 4/4 redis.

### Done (2026-08-16 session)
- **Phase 2 production hardening — all opt-in via env vars** (213/215 tests passing; 2 failures = pre-existing AppLocker `datasets` block; `ruff check src/` clean):
  - **API-key auth** — `CONFORMAL_RAG_API_KEY` non-empty → every route except `/health`/docs/static frontend requires `X-API-Key` header; OPTIONS passes for CORS preflight. Empty = auth off (local dev unchanged). `src/conformal_rag/api.py` guard middleware.
  - **Rate limiting** — `CONFORMAL_RAG_RATE_LIMIT_PER_MINUTE` → fixed-window per-client-IP limiter (X-Forwarded-For aware), 429 over limit. 0 = unlimited. `RateLimiter` class in api.py.
  - **CORS** — `CONFORMAL_RAG_CORS_ORIGINS` (comma-separated) → CORSMiddleware. Empty = none (bundled frontend is same-origin).
  - **Persistence** — `CONFORMAL_RAG_STATE_DIR` → engine calibration, document index (+ embeddings), and per-domain calibrations saved as JSON on every mutation, restored at startup. `engine.save_state/load_state`, `retriever.save_state/load_state`, `domain_manager.save_state/load_state`. Empty = in-memory only.
  - **Logging** — `CONFORMAL_RAG_LOG_LEVEL` sets the `conformal_rag` package level.
  - **Retrieval fixes** — e5 instruction prefixes (`query: `/`passage: `) applied; dense embeddings precomputed once at `index()` time (queries no longer re-encode all docs); module-level embedder cache (model loads once per process); L2-normalized cosine via dot product. `src/conformal_rag/retrieval.py`.
  - **Config alive** — `config.py` Settings now drives api.py (`create_app(settings)`); new fields `api_key`, `rate_limit_per_minute`, `cors_origins`, `state_dir`, `log_level` all env-loaded.
  - **21 new tests** — `tests/test_hardening.py`: auth (5), rate limiter unit + API 429 (5), CORS preflight/disabled (2), engine/domain/API persistence round-trips (4), retrieval prefixes/precompute/persistence (5). `tests/test_integration.py` retrieval test rewritten with a deterministic fake embedder (old one mocked `np` wholesale and was coupled to the removed implementation).
- **Phase 1 audit fixes — the math is now real** (192/194 tests passing; 2 failures = pre-existing AppLocker `datasets` block):
  - **Real CRC threshold** — `engine.py` now implements Bates et al. 2021 Theorem 1 exactly: threshold = (c_max+1)-th largest hallucination score, c_max = floor((n+1)α)−1; approve strictly above (ties abstain); too-few-samples → abstain-on-everything + warning instead of a fake guarantee. Guarantee stated honestly: E[P(approved ∧ wrong)] ≤ α.
  - **Score direction unified** — high = faithful everywhere; `online.py` ACI error event and `should_abstain` flipped to match; `engine.record()` approval rule now matches `predict()`; both online test files rewritten to the correct convention.
  - **NLI `id2label` fix** — entailment index read from `model.config.id2label` (raises if absent), loaded model + index logged; fallback model can no longer silently invert scores.
  - **Guards wired** — `validate_label_distribution` rejects degenerate calibration (ValueError → HTTP 422), `check_calibration_coverage` warnings surfaced in `CalibrationStats.warnings`, `sanitize_query` applied to all judge-prompt inputs.
  - **Judge parsing hardened** — accepts "0.85"/"85%"/bare "0"/"1", rejects bare multi-digit integers (the "85" → 1.0 approval exploit); unparseable output logged.
  - **API hardened** — blocking handlers (`/predict`, `/calibrate`, `/index`, `/retrieve`, `/record`, domain calibrate) are now plain `def` (threadpool, not event loop); `/predict` no longer records fake labels (new `FDRMonitor.record_prediction`); request size limits via Pydantic `Field(max_length=...)`; LLM `timeout` actually passed to litellm.
  - **Honest monitoring** — `FDRMonitor` computes false-alarm rate from labeled feedback only; added `labeled_feedback` metric.
  - **New statistical tests** — `tests/test_crc.py`: threshold-rank math, empirical guarantee check on 2000 exchangeable samples, tie-abstention, degenerate-label rejection, record/predict rule parity.
  - **Deps cleaned** — removed never-imported `mapie` and `online-cp` from pyproject.toml.
- **Frontend rebuilt as "Ember Console"** — complete redesign of `frontend/` (index.html + css/styles.css + js/api.js + js/app.js + js/charts.js), served by FastAPI at `/`. Warm-charcoal dark theme (#14110e base, ember #e8833a, sage #7ba88f), Bricolage Grotesque + Space Mono, left sidebar rail nav. Four views: Check an Answer, Teach It, Live Status, Topics. Verdict rendered as a semicircular SVG support dial with α-threshold tick (replaced the old linear gauge). An intermediate "paperistic" design was built and scrapped the same day — any doc mentioning paper/cream/editorial/Spectral/DM Serif is stale.
- **Web UI replaces Gradio demo** — vanilla JS frontend in `frontend/`, served by FastAPI at `/` (same origin, no CORS, no build step)
- **API refactor** — `src/conformal_rag/api.py` now has: `/config` (GET/POST, engine re-init with alpha + online ACI toggle), `/samples`, `/index`, `/documents`, `/retrieve`, `/predict`, `/calibrate` (typed `CalibrationSample`), `/calibration`, `/record` (engine.record → monitor + ACI), `/metrics` (reads engine's own monitor — fixed disconnected-monitor bug), `/online`, `/domains` (GET/POST), `/domains/{name}/calibrate`, `/domains/{name}/drift`, `/feedback`, `/reset`, `/health`
- **22 API tests** — `tests/test_api.py` (was 2). 132/132 core tests passing, ruff clean
- **Live smoke test verified** — index → retrieve → predict (accepted faithful / abstained hallucinated) → calibrate → record → metrics → online → domain register/calibrate/drift, all against real NLI model
- **Cleanup** — deleted `demo/` (Gradio), removed junk files `({title` and `{` from repo root, updated README/AGENTS/HANDOFF

### Done (2026-07-23 session — 10 critical fixes)
- **Client SDK** — `index()`, `retrieve()`, `check()`, `query()` — `src/conformal_rag/client.py`
- **Engine wiring** — `FDRMonitor` + `OnlineConformal` via `__init__` + `record()` — `src/conformal_rag/engine.py:19-42,154-175`
- **True ACI** — Quantile-level Adaptive Conformal Inference (Gibbs & Candès 2021) — `src/conformal_rag/online.py`
- **Proper conformal CI** — `(n+1)(1-α)` finite-sample quantile — `src/conformal_rag/engine.py:182-196`
- **Retrieval fixes** — BM25 built once, Qdrant sparse uses real token weights — `src/conformal_rag/retrieval.py`
- **LLM judge** — Regex-based score parsing — `src/conformal_rag/scoring.py:106-117`
- **NLI scoring** — Verified correct: `probs[0][0]` = ENTAILMENT for `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- **Paper citations** — Added 2025-2026 papers (Semantic Illusion, Conformal-RAG, TRAQ, C-RAG, etc.)

### Blocked
- 17 benchmark/retrieval tests fail due to `datasets` DLL block (AppLocker). Need different machine.
- `scikit-learn==1.4.2` and `scipy==1.14.0` pinned (AppLocker binary hash block)

## Quick reference

### Running the web UI
```bash
cd C:\Users\ELCOT\Desktop\Final_year
set PYTHONPATH=src
uvicorn conformal_rag.api:create_app --factory --port 8000
# Opens at http://localhost:8000 (UI served at /, API on same origin)
```

### Client SDK usage
```python
from conformal_rag.client import ConformalRAG

rag = ConformalRAG(alpha=0.05)
rag.index([
    {"text": "Paris is the capital of France.", "metadata": {"domain": "geography"}},
])
context = rag.retrieve("What is the capital of France?")
result = rag.check("What is the capital of France?", response="Paris")
print(result.abstained)      # False
print(result.risk_score)     # 0.02
print(result.risk_bound)     # 0.05
```

### Engine API (lower level)
```python
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.online import OnlineConformal

engine = ConformalRAGEngine(
    alpha=0.05,
    online=OnlineConformal(alpha=0.05),
)
engine.calibrate(queries, contexts, responses, labels)
result = engine.predict(query, context, response)
```

## Open decisions

- **Benchmarks**: Run on different machine or lift AppLocker allowlist. This machine can't import `datasets`.
- **HuggingFace Space**: Deploy web UI as public URL for class presentation. Single-process FastAPI app serves both API and frontend — needs `uvicorn conformal_rag.api:create_app --factory` as the run command and model caching configured.
- **Chrome extension**: 15-25h effort. Start after HF Space is deployed.

### Audit findings (2026-08-16) — Phase 1 + Phase 2 items FIXED, remainder below
- ~~MAPIE never imported~~ — FIXED: dependency removed, engine documents Bates et al. 2021 construction.
- ~~Calibration threshold heuristic~~ — FIXED: real CRC threshold with finite-sample correction.
- ~~Online ACI score-direction inverted~~ — FIXED: unified high=faithful.
- ~~NLI `probs[0][0]` label-order landmine~~ — FIXED: `id2label` lookup.
- ~~Guards unwired~~ — FIXED: validation + warnings + judge-prompt sanitization.
- ~~API blocking-async~~ — FIXED: plain `def` handlers on threadpool.
- ~~`/predict` fake labels~~ — FIXED: `record_prediction` counts abstentions only.
- ~~`abstained` direction inconsistent between predict/record~~ — FIXED in Phase 1: both use `score <= threshold` → abstain.
- ~~No API auth/CORS/rate-limit~~ — FIXED in Phase 2 (opt-in env vars).
- ~~No calibration persistence~~ — FIXED in Phase 2 (`CONFORMAL_RAG_STATE_DIR`).
- ~~No logging config~~ — FIXED in Phase 2 (`CONFORMAL_RAG_LOG_LEVEL`).
- ~~Retrieval re-encodes all docs per query / no e5 prefixes / no persistence~~ — FIXED in Phase 2.
- ~~Keyword domain routing can misroute~~ — FIXED 2026-08-21 (Phase 3): embedding-centroid routing among calibrated domains, keyword fallback retained.
- ~~Qdrant sparse path untested against a live server~~ — FIXED 2026-08-21 (Phase 3): path was broken end-to-end; collection create + BM25-weighted upserts implemented, live-server tests added and passing.
- ~~Rate limiter single-process in-memory~~ — FIXED 2026-08-21 (Phase 3): `RedisRateLimiter` selected via `CONFORMAL_RAG_REDIS_URL`, atomic Lua sliding window, fail-open.

## Artifacts

- Branch: `dev`
- Web UI: `frontend/` (vanilla JS, served by FastAPI at `/`)
- API: `src/conformal_rag/api.py` (full REST API, 18 endpoints)
- Paper draft: `docs/paper/draft.md` (v0.2, updated citations)
- Research: `RESEARCH.md` (updated with 2025-2026 findings)
- Remaining work: `docs/remaining-work.md` (elevation roadmap)
- Eval script: `benchmarks/eval.py`
- Handoff: This file
