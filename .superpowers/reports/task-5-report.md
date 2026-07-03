# Task 5: Qdrant Retrieval — Real Hybrid Search

## Status: DONE

## What Changed

### `src/conformal_rag/retrieval.py`
- Replaced stub `_dense_search()` and `_sparse_search()` with real implementations
- **In-memory mode**: `sentence-transformers` (`intfloat/e5-base-v2`) for dense, `rank-bm25` (BM25Okapi) for sparse
- **Qdrant mode**: Optional — activated by passing `qdrant_url` to constructor
- Hybrid RRF fusion unchanged (already correct)
- Lazy embedder loading (only when first search is called)

### `tests/test_retrieval_real.py` (new)
- 7 tests: RRF merge, RRF empty, index adds docs, retrieve returns results, retrieve finds relevant doc, dense returns list, sparse returns list
- All pass in in-memory mode (no Qdrant needed)

### Dependencies added
- `sentence-transformers>=5.6` — dense embeddings
- `rank-bm25>=0.2` — BM25 sparse retrieval

## Test Results
```
tests/test_retrieval_real.py — 7/7 PASSED
Full suite — 95/97 PASSED (2 pre-existing test_api.py async failures)
```

## Files Modified
- `src/conformal_rag/retrieval.py`
- `tests/test_retrieval_real.py` (new)
- `pyproject.toml`
- `uv.lock`

## Commit
`feat: implement Qdrant hybrid retrieval with in-memory fallback`
