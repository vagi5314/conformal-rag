# Task 6: Online Adaptive Conformal — Report

## Status: DONE

## What I Implemented

**`src/conformal_rag/online.py`** — `OnlineConformal` class implementing Online Adaptive Conformal Inference (ACI):
- ACI update rule: adjusts threshold each round based on empirical FDR vs target α
- Sliding window with configurable max size (default 500) via `deque(maxlen=...)`
- Learning rate (η) controls adaptation speed (default 0.01)
- Threshold clamped to [0, 1]
- `update(score, is_correct)` — observes a labeled sample, updates threshold
- `should_abstain(score)` — returns True if score exceeds threshold (risk too high)
- `_compute_fdr()` — False Discovery Rate from window (FP / rejected)
- `get_stats()` — returns current threshold, window size, update count, FDR

**`tests/test_online.py`** — 5 tests covering all interfaces.

## TDD Evidence

### RED Phase
```
ERROR collecting tests/test_online.py
ModuleNotFoundError: No module named 'conformal_rag.online'
```

### GREEN Phase
```
tests/test_online.py::TestOnlineConformal::test_initializes_with_defaults PASSED [ 20%]
tests/test_online.py::TestOnlineConformal::test_updates_threshold PASSED       [ 40%]
tests/test_online.py::TestOnlineConformal::test_abstains_on_high_risk PASSED   [ 60%]
tests/test_online.py::TestOnlineConformal::test_computes_fdr PASSED            [ 80%]
tests/test_online.py::TestOnlineConformal::test_window_maxsize PASSED          [100%]
5 passed in 0.20s
```

### Cleanup Phase
Removed unused imports (`Optional` from online.py, `pytest`/`deque` from test). Tests re-verified: 5/5 passing.

## Files Changed
| File | Action |
|------|--------|
| `src/conformal_rag/online.py` | Created (90 lines) |
| `tests/test_online.py` | Created (43 lines) |

## Test Results
- **5/5 passing**, output pristine (no warnings, no noise)
- Full suite: **48/48 passing** (no regressions)

## Commit
- `33e077d` — `feat: online adaptive conformal inference (ACI)`

## Self-Review Findings
- **Clean property pattern**: `_threshold` backing field + `threshold` property allows test to set `oc._threshold` directly while `oc.threshold` still reads correctly via property getter.
- **No unused imports** after cleanup pass.
- **Matches plan exactly**: interfaces, parameters, update rule, FDR computation all match the spec at lines 1254-1415.
- **Edge cases handled**: window < 10 samples returns FDR=0.0, empty rejection set returns 0.0, threshold clamped to [0,1].

## Issues or Concerns
None. Task 6 complete and verified.
