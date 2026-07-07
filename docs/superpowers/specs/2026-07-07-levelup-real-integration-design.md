# Conformal RAG — Level-Up: Real Integration & Honest Claims

**Date:** 2026-07-07
**Status:** Approved scope (all 6 gaps)
**Goal:** Remove fabricated/stubbed behavior and upgrade to a genuinely integrated,
correct, and honestly-documented library. No shortcuts, TDD-driven.

## Context

Audit of the repo (after fixing the AppLocker/env blocker that prevented any test run)
found 161/161 tests passing, but several headline claims were fabricated, stubbed, or
overstated. Tests were green largely because they did not assert real behavior. This
spec upgrades the 6 identified gaps so the library does what AGENTS.md claims.

## Gap 1 — Client SDK is fake (`src/conformal_rag/client.py`)

**Problem:** `index()` is a no-op (`pass`); `query()` injects `"placeholder context"`
and `response=""` into the engine, so it scores an empty response and performs no
retrieval. The "Python client SDK" does not work.

**Fix (real integration):**
- `index(documents)`: build a `HybridRetriever` (in-memory by default; Qdrant when
  `qdrant_url` is configured) and store it on the client. Accept `Document`-like
  objects (`doc.text` / `doc.page_content` + `doc.metadata`) or plain dicts/strings.
- `retrieve(query) -> list[str]`: returns top-k fused context via the retriever.
- `check(query, response, context=None) -> ConformalResult`: if `context` is None and a
  retriever is indexed, retrieve it; run `engine.predict(query, context, response)`.
- Keep `query()` as a thin, clearly-documented convenience OR remove it; do not ship a
  method that silently fakes context/response.
- The client never generates LLM responses — that is the caller's RAG pipeline. The SDK
  retrieves context and risk-checks a provided response.

**Tests:** real retrieval returns non-empty context; `check()` with a retrieved context
and a real response yields a `ConformalResult` with a sane `risk_score`; `index()` on a
small corpus retrieves the relevant doc.

## Gap 2 — Core engine is disconnected from online/monitoring

**Problem:** `ConformalRAGEngine` never imports or uses `OnlineConformal` or
`FDRMonitor`. "Online adaptive conformal" and "FDR monitoring" are orphan modules.

**Fix (wire it in):**
- Add optional `fdr_monitor: FDRMonitor | None` and `online: OnlineConformal | None` to
  `ConformalRAGEngine.__init__`.
- Add `record(query, context, response, actual_hallucination: bool)`:
  - append to `fdr_monitor` (predicted = abstained/risk>threshold, actual = label),
  - call `online.update(score, is_correct=not actual_hallucination)`,
  - if `online` is enabled, sync `self._threshold` from `online.threshold` (adaptive).
- `predict()` optionally emits the monitor record when a label is available.

**Tests:** after `record()` calls, `fdr_monitor.get_metrics()` reflects real counts;
`online.threshold` moves toward keeping FDR near `alpha`.

## Gap 3 — `online.py` ACI is mislabeled

**Problem:** docstring claims ACI formula `α_{t+1}=α_t+η(α−I(rej)·I(correct))` but the
code does `threshold += η·(empirical_fdr − α)` — a different heuristic, and it adjusts
the threshold instead of the quantile level.

**Fix (correct ACI, per Gibbs & Candès 2021):**
- Implement the standard ACI on the nonconformity score quantile level `λ_t`:
  `λ_{t+1} = clamp(λ_t + η·(miscoverage_t − α), 0, 1)` where
  `miscoverage_t = 1` if the new score exceeds the current `(1−λ_t)` quantile of the
  calibration window, else `0`.
- Threshold = `(1−λ_t)` quantile of the window; recompute after each update.
- Keep the sliding window; document the convergence claim honestly (heuristic
  convergence under exchangeability, not a hard guarantee).
- Rename internal method to reflect it is ACI-on-threshold, or implement true
  quantile-level ACI. We implement true quantile-level ACI.

**Tests:** simulated streaming miscoverage drives `λ_t` so that long-run rejection rate
  approaches `alpha` on exchangeable data.

## Gap 4 — Monitoring dead in serving

**Problem:** `monitor_drift()` always returns "no drift" (its buffer `_domain_drift` is
never populated); API `/metrics` always returns zeros because `predict` never records to
the monitor.

**Fix (live monitoring):**
- `DomainManager.monitor_drift()` derives recent FDR from the domain engine's
  `fdr_monitor` (populate from records), not an empty buffer.
- API: `predict` records `predicted_hallucination` to the monitor; add a `/feedback`
  endpoint `POST /feedback {query, actual_hallucination}` that records the actual label
  and feeds `online.update`. `/metrics` then returns live data.

**Tests:** after predict+feedback, `/metrics` returns non-zero, consistent values;
`monitor_drift` reports drift when simulated FDR exceeds `alpha·1.5`.

## Gap 5 — "Correct confidence interval" overstated

**Problem:** `engine._compute_confidence_interval` returns `(quantile, 1.0)` — only a
lower bound, not a conformal CI.

**Fix (real conformal PI):**
- Compute a one-sided conformal interval for the risk score using the calibration
  nonconformity scores with the finite-sample `(n+1)(1−α)` quantile index.
- Return `(lower, upper)` where `upper = 1.0` (risk is bounded) and `lower` is the
  calibrated `(1−α)` quantile, with the correct `⌈(n+1)(1−α)⌉` index.
- Document the guarantee precisely.

**Tests:** interval lower bound increases as calibration correct-scores spread; index
  matches the finite-sample formula for small `n`.

## Gap 6 — Headline numbers unverifiable

**Problem:** `Faithful→0.9995`, `Hallucination→0.0004`, `1.1s/query`, `FDR=0%` appear
nowhere in code; they read as aspirational.

**Fix (substantiate or retract):**
- Add `benchmarks/eval.py` (or extend `runner.py`) that runs the engine on a small
  labeled sample (reuse HotpotQA/RAGTruth sample loaders) and prints measured:
  per-domain FDR, abstention rate, latency per query (NLI path), and risk scores for one
  faithful vs one hallucinated example.
- Replace the fabricated numbers in AGENTS.md/README with measured values from this run,
  or remove them if not reproducible. No numbers stand without a script that produces
  them.

**Tests:** eval script runs end-to-end on the sample set and emits a structured report
(no hardcoded expected values that fake success).

## Dependencies / risks
- Downgraded `scikit-learn==1.4.2`, `scipy==1.14.0` pinned in `pyproject.toml` (AppLocker
  block on newer binaries). Keep pinned until the policy is lifted.
- No new third-party deps required.
- All changes keep the existing 161 tests green and add new ones; TDD for each gap.

## Verification
- `pytest tests/ -q` stays green (target: 161 + new tests passing).
- New tests per gap assert real behavior (no placeholder/empty-response hacks).
- `ruff check src/ tests/` clean.
- Eval script produces a real report; AGENTS.md numbers match or are removed.
