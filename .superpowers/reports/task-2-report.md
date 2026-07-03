# Task 2 Report: Guards Module (P0 Security)

**Status:** DONE  
**Commit:** `75cca00` — feat: add guards module (P0 security)

## What Was Implemented

5 security guard functions in `src/conformal_rag/guards.py`:

| Function | Purpose |
|----------|---------|
| `sanitize_query()` | Regex-based prompt injection removal (7 patterns) |
| `clamp_alpha()` | Clamps α to valid range [0.001, 0.2] for CRC guarantees |
| `check_calibration_coverage()` | Warns if calibration set < 100 samples |
| `detect_pii()` | Detects email, phone, SSN patterns in text |
| `validate_label_distribution()` | Rejects degenerate label distributions (all-same, >50% hallucination) |

## TDD Evidence

### RED Phase
```
ModuleNotFoundError: No module named 'conformal_rag.guards'
```
Tests written first, confirmed import failure.

### GREEN Phase
14/14 passing after implementing `guards.py`.

### Refinement
Initial implementation had two thresholds in `check_calibration_coverage` (< 100 and < 500). The test `test_no_warnings_above_minimum` expected no warnings at 200 samples. Fixed by removing the < 500 threshold — only the critical < 100 minimum fires.

## Test Results

```
27 passed in 0.23s
```

- 14 new guards tests
- 13 existing model tests (no regressions)

## Files Changed

- `src/conformal_rag/guards.py` — 97 lines (created)
- `tests/test_guards.py` — 76 lines (created)

## Self-Review Findings

1. **Pattern coverage:** 7 injection patterns cover common attacks. Could expand with more adversarial patterns later.
2. **PII detection:** Uses regex only. No ML-based PII detection — acceptable for v1 guard, ML detection can be added in a future task.
3. **`validate_label_distribution` empty case:** Returns `(False, "Empty label list")` — not tested but handled.
4. **No type:** No type errors. Uses `from __future__ import annotations` for forward references.
5. **Edge case:** `clamp_alpha(float('nan'))` returns `0.001` (NaN comparisons are always False, so `max(0.001, min(0.2, nan))` → `max(0.001, nan)` → `0.001`). Acceptable defensive behavior.

## Concerns

None. All 14 tests pass, output is pristine, no warnings.
