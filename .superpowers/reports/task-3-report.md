# Task 3: Fix Confidence Interval Formula

## Summary
Replaced the incorrect proportion CI formula with a conformal quantile-based CI in `src/conformal_rag/engine.py`.

## What Changed

### Problem
The CI was computed as `1.96 * sqrt(threshold * (1-threshold) / n)` — a normal-approximation proportion CI, which is statistically wrong for conformal prediction. Conformal CIs should be derived from the quantile of non-conformity scores.

### Fix
- **engine.py:134-137** — Replaced proportion CI block with `_compute_confidence_interval()` call
- **engine.py:155-165** — Added `_compute_confidence_interval()` method:
  - Filters calibration scores to correct-only (label == 0)
  - Returns `(quantile(scores, 1-alpha), 1.0)` — the conformal lower bound
  - Falls back to `(0.0, 1.0)` on empty calibration data

### Tests
- **tests/test_engine_fix.py** — 3 new tests:
  1. `test_confidence_interval_uses_quantile` — CI bounds valid, lower < upper
  2. `test_confidence_interval_empty_calibration` — returns `(0.0, 1.0)` fallback
  3. `test_confidence_interval_not_proportion_ci` — CI width < 0.5 (not the wide proportion CI)

## Verification
- `pytest tests/test_engine_fix.py -v` — 3/3 passed
- `pytest tests/ -v` — 87/87 passed
