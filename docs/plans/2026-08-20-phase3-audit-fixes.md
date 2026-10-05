# Phase 3 Audit Fixes — Implementation Plan (2026-08-20)

Three items from HANDOFF.md:114: (1) Qdrant sparse path is broken and untested, (2) keyword domain routing can misroute + has no production call sites, (3) rate limiter is single-process. Scope decisions locked: full Qdrant fix + API wiring, embedding routing + wire `/predict`, CI service containers with local skip. **Frontend charts/enhancements are explicitly out of scope — deferred to a later session.**

## Workstream A — Qdrant: make it work, then test live

**A1. Factor shared embedding helpers** (`retrieval.py`): move `_get_embedder`/`_encode_passages`/`_encode_query` (retrieval.py:55-78) to module-level `get_embedder(model)`, `encode_passages(embedder, texts)`, `encode_query(embedder, query)` reusing `_embedder_cache`, so domain.py can reuse them. `HybridRetriever` methods become thin delegates.

**A2. Stable sparse encoding**: token→index via `zlib.crc32(token) % 32768` (stable across processes; `hash()` is salted). Index-side weights: real BM25 term weights from the already-built `BM25Okapi` (`idf * (tf*(k1+1))/(tf + k1*(1-b+b*dl/avgdl))`, negative IDF clamped to 0). Query-side: `max(idf,0) * (1+log(tf))`, unknown tokens skipped; empty vector → return `[]` without calling Qdrant. Replaces the meaningless positional indices at retrieval.py:132-146.

**A3. Collection lifecycle + upsert**: `_ensure_collection()` — dim from `embedder.get_sentence_embedding_dimension()` (not hardcoded 768); delete-if-exists then `create_collection` with named vectors `dense` (VectorParams, Cosine) + `sparse` (SparseVectorParams). Recreate-on-index matches the API's replace semantics (api.py:294 builds a fresh retriever per `/index`). `_upsert_all()` — batched upserts (64) of `PointStruct(id=i, vector={"dense":..., "sparse":...}, payload={"text": doc})`. `index()` always builds BM25 (needed for sparse weights); Qdrant mode → ensure + encode + upsert. `load_state()` re-upserts when a client is set (idempotent, covers Qdrant restarts), wrapped in try/except so startup never crashes on a down server.

**A4. Migrate search to `query_points`** (client.search is deprecated in qdrant-client 1.18): `_qdrant_dense_search` / `_qdrant_sparse_search` use `client.query_points(..., using="dense"/"sparse")`, read `payload.get("text","")`.

**A5. Config + API wiring**: `config.py` gains `qdrant_url: str = ""` (env `CONFORMAL_RAG_QDRANT_URL`, empty = in-memory — explicit opt-in since `qdrant_host` defaults to localhost) and `qdrant_collection: str = "docs"` (env `CONFORMAL_RAG_QDRANT_COLLECTION`). `api.py` gets `_build_retriever(settings)` used at all three construction sites (api.py:173, 279, 294). `/index` and `/retrieve` catch Qdrant connection errors → HTTP 503 (honest failure, no silent fallback). `docker-compose.yml`: pin `qdrant/qdrant:v1.18.0`, app env → `CONFORMAL_RAG_QDRANT_URL: http://qdrant:6333`.

**A6. Tests**: unit tests in `tests/test_retrieval.py` (crc32 stability, BM25 weight correctness, unknown-token skip, empty-vector guard with mock client). New `tests/test_qdrant_live.py` skip-gated on `CONFORMAL_RAG_QDRANT_TEST_URL` (default `http://localhost:6333`) via socket probe; unique collection per run, deleted after. Covers: collection creation, dense+sparse fusion round-trip, sparse-only match, dense-only match, re-index leaves no stale points, load_state re-populates. Loads real e5 (already done by existing tests).

## Workstream B — Embedding domain routing + wire `/predict`

**B1. Router** (`domain.py`): `DomainManager.__init__(self, routing="embedding", route_margin=0.55)` — still no-arg constructible, **no model load in `__init__`** (lazy on first routed query, keeps API tests fast). New `_domain_queries: Dict[str, List[str]]` captured in `calibrate()` (engine discards queries, engine.py:106-107). Domain representation: re-normalized centroid of L2-normalized calibration-query embeddings (passage prefix); fallback to domain-name embedding until calibrated. `_detect_domain(query)` (name preserved — patched at tests/test_domain.py:47): candidates = **calibrated** domains only (routing to uncalibrated would apply the unguaranteed 0.5 default); none → `"general"`; argmax cosine; best < `route_margin` → `"general"`; embedder load failure → warn + keyword fallback. Old keyword logic kept verbatim as `_detect_domain_keyword` (used when `routing="keyword"` and as failure fallback). `predict()` also calls `monitors[domain].record_prediction(abstained=result.abstained)` mirroring api.py:323. Test fix: tests/test_domain.py:49 mock becomes `MagicMock(domain="finance", abstained=False)` (record_prediction does arithmetic on abstained).

**B2. Persistence**: `save_state()` also writes `domain_routing.json` (`{domain: [queries]}`, capped 100×400 chars); `load_state()` reads it if present, absent → name-only routing until next calibration. Existing `domain_NNN.json` format untouched.

**B3. Wire `/predict`** (`api.py`, `client.py check()`): explicit registered domain → `domain_manager.predict(..., domain=...)` (that domain's calibrated threshold — the point of per-domain guarantees); explicit unregistered domain → main engine with tag (current behavior); `domain=None` + ≥1 calibrated registered domain → `domain_manager.predict(...)` (embedding routing); else main engine exactly as today (keeps tests/test_api.py:126-146 passing).

**B4. Config**: `domain_route_margin: float = 0.55` + env `CONFORMAL_RAG_DOMAIN_ROUTE_MARGIN`, passed to `DomainManager()` in `create_app`.

**B5. Tests**: keyword-mode still maps finance/medical/legal/general; embedding mode with injected fake embedder (pattern from tests/test_integration.py:11-24) — similar→match, dissimilar→general, uncalibrated excluded, centroid updates after calibrate, routing-state round-trip, predict records to domain monitor. API tests: register+calibrate "finance" then `/predict` without domain routes there (patch `dm._detect_domain` to avoid loading e5; assert `result.domain=="finance"` + domain monitor count); explicit unregistered domain still main engine.

## Workstream C — Redis-backed rate limiter

**C1. `RedisRateLimiter`** (`api.py`): same interface — `(limit_per_minute, redis_url, prefix, window=60.0)`, sync `allow(key, now=None) -> bool`, `limit<=0` → True. Sliding window via sorted set + atomic Lua script (ZREMRANGEBYSCORE → ZCARD → conditional ZADD + EXPIRE), matching the semantics pinned by `test_window_slides` (tests/test_hardening.py:87-93). `now=None` → `time.time()` (wall clock — monotonic is per-process). Lazy `redis.Redis.from_url(..., socket_connect_timeout=2, socket_timeout=2)`. **Fail-open**: any redis error → warn + return True (Redis outage must not take down the API).

**C2. Selection** at api.py:189: `settings.redis_url` non-empty → Redis backend, else in-memory. Existing tests use empty `redis_url` → stay hermetic.

**C3. Tests** — new `tests/test_redis_rate_limit.py`, skip-gated on `CONFORMAL_RAG_REDIS_TEST_URL` (default `redis://localhost:6379/0`): blocks over limit, window slides with injected `now`, per-key isolation, fail-open against dead port, API-level 200/200/429 with live Redis. No new deps (`redis>=5.0` already core; real container in CI, no fakeredis).

## CI + docs

- `.github/workflows/ci.yml`: service containers `qdrant: qdrant/qdrant:v1.18.0` (6333) + `redis: redis:alpine` (6379); job env sets both `*_TEST_URL` vars so skip-gated tests run in CI.
- `HANDOFF.md`: strike the three Phase 3 items (line 114), record changes, add a "Frontend enhancements (charts)" item to the next-steps list. `AGENTS.md`: Phase 3 done, fix misleading "actual token weights" note, document new env vars.

## Execution order

1. Save plan file → 2. Config fields (A5, B4) → 3. Retrieval A1-A4 + unit tests → 4. Domain routing B1-B2 + tests → 5. API/client wiring (A5 api, B3) + tests → 6. Redis limiter C1-C2 + tests → 7. docker-compose + CI → 8. Live tests (A6, C3) → 9. Docs + full verification.

## Verification

- `uv run ruff check src/ tests/`
- **Stop any running uvicorn first** (torch access-violation quirk on this machine), then `uv run pytest tests/ -q` — live tests skip locally unless `docker compose up -d qdrant redis` is running
- If Docker Desktop is available: bring up qdrant+redis, run the two live test files, tear down (timebox ~3 min; else rely on CI)
- Optional smoke: app with `CONFORMAL_RAG_QDRANT_URL` set → POST `/index` + `/retrieve`; stop server before further pytest

## Risks

- qdrant image tag `v1.18.0` must exist to pair with client 1.18.0; verify and adjust both ci.yml and docker-compose if the closest 1.18.x differs
- e5 download adds ~30-60s once in CI (already incurred by existing real-model tests)
- `route_margin=0.55` is a heuristic; env-tunable, and routing only picks among calibrated domains, so a bad margin degrades to "general", never a wrong certified domain
- Qdrant mode fails with 503 instead of silent in-memory fallback, by design

## Deferred (not in this session)

- Frontend charts + UI enhancements (user-requested, scheduled later)
