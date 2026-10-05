# Remaining Work — Conformal RAG Platform

**Branch:** `dev` (active)  
**Last Session:** 2026-08-21 — Phase 3 audit fixes complete (Qdrant path, embedding domain routing, Redis rate limiter), 249 tests passing + 10 skip-gated live tests

---

## 1. Audit Fixes (P0 — Correctness Before Any "Guaranteed" Claims) — ✅ DONE 2026-08-16

All items from the 2026-08-16 engineering audit are fixed (details in HANDOFF.md "Audit findings"):

| Task | Status |
|------|--------|
| Unify score direction: high = faithful everywhere | ✅ `online.py` rewritten |
| NLI label lookup via `model.config.id2label` | ✅ `scoring.py` |
| Real CRC threshold (Bates et al. 2021 Theorem 1, (n+1) correction) | ✅ `engine.py`, MAPIE dep removed |
| Wire the guards (sanitize, validate, coverage warnings) | ✅ wired into calibrate + judge prompt |
| API concurrency: blocking handlers as plain `def` | ✅ threadpool |
| `/predict` no longer records fake labels | ✅ `record_prediction` |
| `timeout` passed to `litellm.completion` | ✅ |
| Statistical CRC tests | ✅ `tests/test_crc.py` |

---

## 2. Production Hardening (P0 — Before Any Public Deploy) — ✅ DONE 2026-08-16

All opt-in via env vars; local dev unchanged with zero config.

| Task | Status |
|------|--------|
| API auth + rate limiting | ✅ `CONFORMAL_RAG_API_KEY` (X-API-Key), `CONFORMAL_RAG_RATE_LIMIT_PER_MINUTE` (per-IP, 429) |
| CORS middleware | ✅ `CONFORMAL_RAG_CORS_ORIGINS` (comma-sep) |
| Request size limits | ✅ Pydantic `Field(max_length=...)` (Phase 1) |
| Calibration persistence | ✅ `CONFORMAL_RAG_STATE_DIR` — engine + index + domains JSON, restored at startup |
| Dockerfile: `COPY frontend/ frontend/` | ✅ |
| Logging | ✅ `CONFORMAL_RAG_LOG_LEVEL`, package loggers throughout |
| Retrieval productionization | ✅ e5 `query:`/`passage:` prefixes, embeddings precomputed at index time, embedder cache, index save/load |

**Phase 3 leftovers:** ✅ ALL DONE 2026-08-21 — embedding-based domain routing (default, keyword retained as fallback), Qdrant sparse path fixed end-to-end and live-tested against qdrant v1.18.0, Redis-backed rate limiter for multi-instance deploys. See HANDOFF.md "Done (2026-08-21 session)".

---

## 3. Run Benchmarks with LLM Judge (P1)

| Task | Prerequisites | Effort |
|------|---------------|--------|
| Run `python -m benchmarks.runner --llm-judge --limit 50` on clean machine | API keys (Groq, OpenRouter, Together, Gemini), no AppLocker DLL block | 5 min |
| Fill paper Section 6 with real FDR/coverage/latency/cost numbers | Above run complete | 30 min |

**Why:** Without this, the paper's results section has gap rows. NLI-only path is conservative — RAGTruth abstains 100%.

---

## 4. HuggingFace Space (P1)

| Task | Prerequisites | Effort |
|------|---------------|--------|
| Deploy FastAPI app + Ember Console frontend to HF Spaces | Audit fixes (esp. blocking-async), Dockerfile frontend fix | 2-4h |

**Why:** Public shareable URL for class presentation. The web UI is done; only deployment remains (Gradio path was deleted 2026-08-16).

---

## 5. Chrome Extension (P2 — "Wow" Factor)

| Task | Prerequisites | Effort |
|------|---------------|--------|
| Build Chrome MV3 extension shell (content script + service worker) | None | 4-6h |
| Wire to FastAPI backend for conformal verdict | Backend deployed + auth | 4h |
| UI polish: verdict badge + coverage guarantee display | Shell complete | 6-8h |
| Testing + Chrome Web Store prep | All above | 3h |

**Why:** No existing fact-checking extension offers statistical coverage guarantees. Unique selling point.

---

## 6. LangChain/LlamaIndex Wrapper (P2 — Ecosystem)

| Task | Prerequisites | Effort |
|------|---------------|--------|
| LangChain `BaseRetriever` wrapper with conformal metadata | Working client SDK + async surface | 6h |
| LlamaIndex `QueryEngine` wrapper | Working client SDK | 6h |
| Tests + docs | Wrappers complete | 4h |

**Why:** No existing conformal prediction integration in either framework. Fills genuine ecosystem gap. Note: SDK is sync-only today; wrapping blocking torch calls will stall an event loop — add an async surface first.

---

## 7. Publication

| Task | Prerequisites | Effort |
|------|---------------|--------|
| **UncertaiNLP @ EMNLP 2026** — deadline Aug 7, 2026 — **PASSED** | — | — |
| arXiv preprint | LLM-judge benchmark results; Section 6 numbers filled from a real run | 1-2 weeks |
| Next workshop venue (check EMNLP 2026 main, NeurIPS workshops) | arXiv ready | varies |

**Why:** Method sections of `docs/paper/draft.md` were corrected 2026-08-21 to match the code — §3.2 now states the actual Bates et al. Theorem 1 construction (c_max order statistic, strict inequality, minimal sample size) and the marginal-only guarantee; §3.4 now describes quantile-level ACI and labels it heuristic; §4.1 documents the real escalation gate, the 0.6-judge/0.4-NLI blend, and the injection/parse hardening; §5.3 lists the actual model IDs, provider chain, and hyperparameters. **Remaining blocker is empirical:** Section 6 still has no real FDR/coverage/latency numbers.

---

## Summary Table

| Priority | Task | Blocked By | Effort |
|----------|------|------------|--------|
| ✅ Done | Audit fixes (score direction, conformal threshold, guards, async) | — | — |
| ✅ Done | Production hardening (auth, CORS, persistence, Dockerfile, logging) | — | — |
| 🟡 P1 | Benchmark run with LLM judge | Clean machine + API keys | 5 min + 30 min |
| 🟡 P1 | HuggingFace Space | None (hardening done) | 2-4h |
| 🟢 P2 | Chrome extension | Backend deployed + auth | 15-25h |
| 🟢 P2 | LangChain/LlamaIndex wrapper | Async SDK surface | 12-20h |
| 🟢 P2 | arXiv / next venue | Benchmarks | 2-3 weeks |

**P0 is complete. Next: HF Space deploy, then benchmarks on a clean machine.**
