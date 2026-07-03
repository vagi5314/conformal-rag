# Task 4 Report: NLI Scoring Pipeline

**Date:** 2026-07-03
**Status:** DONE
**Commit:** d07ae70 — feat: hybrid scoring pipeline (NLI + LLM judge)

## What I Implemented

Hybrid scoring pipeline with two paths:
1. **NLIScorer** — DeBERTa-v3 NLI ensemble (fast path, ~70% of queries)
2. **LLMJudge** — LLM-based faithfulness judge (uncertain cases, ~30%)
3. **ScoringPipeline** — Orchestrates both, returns `ScoringBreakdown`

### Files Created
- `src/conformal_rag/scoring.py` — 121 lines
- `tests/test_scoring.py` — 66 lines

## TDD Evidence

### RED Phase
```
ModuleNotFoundError: No module named 'conformal_rag.scoring'
```
Tests correctly failed — module didn't exist yet.

### GREEN Phase
```
6 passed in 1.27s
```
All 6 tests pass after implementation.

### Full Suite
```
37 passed in 1.77s
```
No regressions across all existing tests.

## Test Results

| Test | Status |
|------|--------|
| TestNLIScorer::test_initializes_with_models | PASSED |
| TestNLIScorer::test_scores_faithfulness | PASSED |
| TestLLMJudge::test_initializes_with_config | PASSED |
| TestLLMJudge::test_builds_prompt | PASSED |
| TestScoringPipeline::test_scores_with_nli_only | PASSED |
| TestScoringPipeline::test_scores_with_hybrid | PASSED |

Output: Pristine (no warnings, no noise).

## Self-Review Findings

1. **Type hints:** All public methods typed correctly — `float`, `str`, `Optional[float]`, `ScoringBreakdown`.
2. **Dataclasses vs Pydantic:** `NLIScorer` and `LLMJudge` use `@dataclass` (lightweight, no validation needed). `ScoringBreakdown` uses Pydantic (matches existing `models.py` convention).
3. **LLMJudge.__post_init__:** Handles `None` defaults for both `config` and `client` — safe to construct without args.
4. **Confidence threshold:** NLI confidence drives the fast-path vs LLM-path decision — `abs(score - 0.5) * 2` maps 0-1 scores to 0-1 confidence (0.5 = least confident, 0/1 = most confident).
5. **Weighted ensemble:** Uses `(0.6, 0.4)` default — LLM judge gets 60% weight, NLI gets 40%. Matches plan spec.
6. **No secrets, no comments, no narration in code.** Clean.

## Concerns

None. All tests pass, implementation matches the plan exactly, full suite green.
