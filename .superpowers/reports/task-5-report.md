# Task 5 Report: Conformal Risk Control Engine

## Status: DONE

## What Was Implemented
- `src/conformal_rag/engine.py` — `ConformalRAGEngine` class with `calibrate()` and `predict()` methods
- `tests/test_engine.py` — 6 tests covering initialization, alpha clamping, calibration, prediction, abstention, and default threshold

## TDD Evidence

### RED Phase
- Created `tests/test_engine.py` with 6 tests importing `ConformalRAGEngine`
- Ran tests → `ModuleNotFoundError: No module named 'conformal_rag.engine'` — confirmed RED

### GREEN Phase (Initial)
- Implemented `engine.py` with full `ConformalRAGEngine` class
- Ran tests → 3 passed, 3 failed (Pydantic rejected `MagicMock` as `ScoringBreakdown` input)

### Fix 1: Test Mock Type
- Updated tests to use `_make_breakdown()` helper returning real `ScoringBreakdown` instances instead of `MagicMock`
- Ran tests → 4 passed, 2 failed (abstention logic inverted)

### Fix 2: Abstention Logic
- Plan specified `abstained = risk_score > threshold` but tests expected the opposite
- Tests: `final_score=0.8, threshold=0.45` → `abstained is False` and `final_score=0.2, threshold=0.45` → `abstained is True`
- Fixed to `abstained = risk_score < threshold` (abstain when faithfulness score falls below threshold)
- Ran tests → **6/6 PASSING**

## Test Results
```
tests/test_engine.py::TestConformalRAGEngine::test_initializes_with_alpha PASSED
tests/test_engine.py::TestConformalRAGEngine::test_clamps_invalid_alpha PASSED
tests/test_engine.py::TestConformalRAGEngine::test_calibrate_with_samples PASSED
tests/test_engine.py::TestConformalRAGEngine::test_predict_with_threshold PASSED
tests/test_engine.py::TestConformalRAGEngine::test_abstains_when_risk_too_high PASSED
tests/test_engine.py::TestConformalRAGEngine::test_predict_without_calibration_uses_default PASSED

6 passed in 2.46s
```

Full suite: **43/43 passing**, output pristine (no warnings, no noise).

## Files Changed
| File | Action |
|------|--------|
| `src/conformal_rag/engine.py` | Created |
| `tests/test_engine.py` | Created |

## Self-Review Findings

1. **Abstention direction**: The plan's `abstained = risk_score > threshold` contradicts the plan's own tests. The `final_score` is a faithfulness score (higher = more faithful), so abstention should fire when score falls *below* the threshold. Fixed to match test expectations.

2. **Mock type safety**: Tests needed real `ScoringBreakdown` instances because Pydantic's strict mode rejects `MagicMock`. Added `_make_breakdown()` helper for clean mock construction.

3. **Consumed interfaces correctly**:
   - `ScoringPipeline` (Task 4) — used via `self.scoring.score()` returning `ScoringBreakdown`
   - `guards.clamp_alpha` (Task 2) — used in `__init__` to clamp alpha to [0.001, 0.2]

4. **Produced interfaces match plan**:
   - `ConformalRAGEngine.calibrate()` → returns `CalibrationStats`
   - `ConformalRAGEngine.predict()` → returns `ConformalResult`

## Concerns
- The plan's abstention logic (`risk_score > threshold`) was inverted relative to its own tests. The tests are the source of truth; the code matches the tests.
- The `_threshold` default of 0.5 when uncalibrated is a reasonable fallback but should be documented as not providing formal FDR guarantees until calibration runs.
