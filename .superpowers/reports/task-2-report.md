# Task 2: LLM Judge — litellm Integration

## Summary
Replaced `NotImplementedError` stub in `_call_provider()` with real `litellm.completion()` calls. Added 4 new tests. All 84 tests pass.

## Changes

### `src/conformal_rag/llm.py`
- Removed `import httpx` and `self._client = httpx.Client(...)` — not needed for litellm
- Replaced `_call_provider()` body: now calls `litellm.completion(model=provider, messages=[...], temperature=..., max_tokens=...)`
- Response extracted via `response.choices[0].message.content.strip()`
- Added back `self.fallback_chain` attribute (backward compat with existing test)

### `tests/test_llm_judge.py` (new)
- `test_complete_returns_string` — verifies mock response string is returned
- `test_complete_uses_temperature_0` — asserts `temperature=0.0` passed to litellm
- `test_fallback_on_first_provider_failure` — verifies fallback tries second provider
- `test_all_providers_fail_raises` — verifies `RuntimeError` when all providers fail

## Test Results
```
84 passed, 7 warnings in 30.90s
```

## Notes
- `LLMJudge` in `scoring.py` uses `LLMClient` — no changes needed there
- Fallback chain: Groq → OpenRouter → Together AI → Gemini
- All calls use `temperature=0.0` for deterministic scoring
