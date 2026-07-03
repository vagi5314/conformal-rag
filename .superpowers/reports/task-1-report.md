# Task 1: NLI Scorer — Real DeBERTa-v3 Inference

- **Status**: DONE
- **Commit**: `5285c09` on `dev`

## Test Results

All 80 tests pass (5 new NLI tests + 75 existing):

```
tests/test_nli.py::TestNLIScorer::test_score_returns_float_between_0_and_1 PASSED
tests/test_nli.py::TestNLIScorer::test_faithful_response_scores_high PASSED
tests/test_nli.py::TestNLIScorer::test_hallucination_scores_low PASSED
tests/test_nli.py::TestNLIScorer::test_decompose_splits_sentences PASSED
tests/test_nli.py::TestNLIScorer::test_score_claim_returns_float PASSED
```

## Changes

1. **`tests/test_nli.py`** — Created with 5 tests for NLIScorer
2. **`src/conformal_rag/scoring.py`** — Replaced placeholder NLIScorer with real DeBERTa-v3 inference

## Implementation Notes

- Model: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (fallback: `cross-encoder/nli-deberta-v3-small`)
- Label order: `{0: entailment, 1: neutral, 2: contradiction}` — score uses entailment probability directly
- Class-level caching via `_nli_cache` dict prevents reloading 1.4GB model across test instances
- Lazy loading: model loads on first `score()` or `_score_claim()` call
- `_score_claim` calls `_load_model()` for direct invocation safety

## Concerns

- Model download (~1.4GB) happens on first use; no progress indicator during download
- Class-level cache persists across tests in same process (intentional for performance)
