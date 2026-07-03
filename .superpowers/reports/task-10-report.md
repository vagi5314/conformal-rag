# Task 10: FastAPI Endpoints — Report

## What I Implemented

Created `src/conformal_rag/api.py` with a `create_app()` factory that builds a FastAPI app with four endpoints:

- `GET /health` — returns `{"status": "ok", "version": "0.1.0"}`
- `GET /metrics` — returns FDR monitoring metrics from `FDRMonitor`
- `POST /predict` — accepts `PredictRequest`, runs `ConformalRAGEngine.predict()`, returns `ConformalResult`
- `POST /calibrate` — accepts `CalibrateRequest`, runs `ConformalRAGEngine.calibrate()`, returns `CalibrationStats`

Two Pydantic models (`PredictRequest`, `CalibrateRequest`) handle request validation. The app initializes `ConformalRAGEngine`, `DomainManager`, and `FDRMonitor` as shared state.

## TDD Evidence

### RED Phase
```
tests/test_api.py::TestHealthEndpoint::test_health_returns_ok FAILED (ImportError)
ModuleNotFoundError: No module named 'conformal_rag.api'
```

### GREEN Phase
```
tests/test_api.py::TestHealthEndpoint::test_health_returns_ok PASSED
tests/test_api.py::TestMetricsEndpoint::test_metrics_returns_stats PASSED
2 passed in 1.60s
```

## Test Results

- **2/2 tests passing**
- Output pristine: no warnings, no noise
- pytest-asyncio with `auto` mode working correctly
- `httpx.AsyncClient` + `ASGITransport` pattern for async endpoint testing

## Files Changed

| File | Action |
|------|--------|
| `src/conformal_rag/api.py` | Created — 65 lines |
| `tests/test_api.py` | Created — 30 lines |

## Self-Review Findings

- No issues found. All interfaces consumed correctly:
  - `ConformalRAGEngine` — used for `predict()` and `calibrate()`
  - `DomainManager` — instantiated (available for future domain-specific routing)
  - `FDRMonitor` — used for `get_metrics()`
- Request models use `Optional` fields where plan specified defaults
- Route handlers use `model_dump()` for Pydantic v2 serialization
- `httpx` already in `pyproject.toml` dependencies — no new deps needed

## Commit

- `ada3380` — `feat: FastAPI endpoints (health, metrics, predict, calibrate)`

## Concerns

None. All tests passing, code matches plan exactly.
