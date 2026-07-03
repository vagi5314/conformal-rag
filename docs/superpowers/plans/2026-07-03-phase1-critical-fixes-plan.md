# Phase 1: Critical Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the conformal-rag core functional by fixing 5 critical issues: real NLI scorer, real LLM judge, correct confidence interval, correct FDR computation, and real Qdrant retrieval.

**Architecture:** Each fix is isolated — fix NLI scorer (scoring.py), fix LLM judge (llm.py), fix confidence interval (engine.py), fix FDR label mapping (online.py), wire up Qdrant retrieval (retrieval.py). TDD for each fix.

**Tech Stack:** Python 3.10+, transformers (DeBERTa-v3), litellm, qdrant-client, sentence-transformers, rank-bm25, numpy, pytest

## Global Constraints

- Python 3.10+ (no `??`, no `??=`, no walrus operator)
- pytest for testing (no unittest)
- No comments in generated code unless asked
- Commit after each task
- TDD: RED-GREEN-REFACTOR for every step
- Verify before claiming done

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/conformal_rag/scoring.py` | NLI scorer (real DeBERTa-v3 inference) |
| `src/conformal_rag/llm.py` | LLM judge (litellm integration) |
| `src/conformal_rag/engine.py` | CRC engine (correct CI formula) |
| `src/conformal_rag/online.py` | Online adaptive (correct FDR) |
| `src/conformal_rag/retrieval.py` | Hybrid retrieval (real Qdrant) |
| `src/conformal_rag/models.py` | Pydantic schemas (already done) |
| `src/conformal_rag/__init__.py` | Package exports |
| `tests/test_nli.py` | NLI scorer tests |
| `tests/test_llm_judge.py` | LLM judge tests |
| `tests/test_engine_fix.py` | Engine CI fix tests |
| `tests/test_online_fix.py` | Online FDR fix tests |
| `tests/test_retrieval_real.py` | Qdrant retrieval tests |
| `tests/test_integration.py` | End-to-end integration test |

---

### Task 1: NLI Scorer — Real DeBERTa-v3 Inference

**Files:**
- Create: `tests/test_nli.py`
- Modify: `src/conformal_rag/scoring.py`

**Interfaces:**
- Consumes: `ScoringBreakdown` from `models.py`
- Produces: `NLIScorer.score(query, context, response) -> float` (0-1, higher = more faithful)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_nli.py
import pytest
from conformal_rag.scoring import NLIScorer


class TestNLIScorer:
    def setup_method(self):
        self.scorer = NLIScorer()

    def test_score_returns_float_between_0_and_1(self):
        score = self.scorer.score(
            query="What is Python?",
            context="Python is a programming language created by Guido van Rossum.",
            response="Python is a programming language."
        )
        assert 0.0 <= score <= 1.0

    def test_faithful_response_scores_high(self):
        score = self.scorer.score(
            query="What is Python?",
            context="Python is a programming language.",
            response="Python is a programming language."
        )
        assert score > 0.7

    def test_hallucination_scores_low(self):
        score = self.scorer.score(
            query="What is Python?",
            context="Python is a programming language.",
            response="Python is a type of snake found in Africa."
        )
        assert score < 0.5

    def test_decompose_splits_sentences(self):
        claims = self.scorer._decompose("Paris is the capital. France is in Europe.")
        assert len(claims) >= 2

    def test_score_claim_returns_float(self):
        score = self.scorer._score_claim(
            context="Python is a language.",
            claim="Python is a language."
        )
        assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nli.py -v`
Expected: FAIL (model not loaded, `_predict` returns 0.5)

- [ ] **Step 3: Implement real NLI scorer**

```python
# src/conformal_rag/scoring.py — replace NLIScorer class
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from conformal_rag.models import ScoringBreakdown
from conformal_rag.llm import LLMClient, LLMConfig


@dataclass
class NLIScorer:
    """NLI ensemble scorer using DeBERTa-v3-large."""

    primary_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    fallback_model: str = "cross-encoder/nli-deberta-v3-small"
    confidence_threshold: float = 0.7
    _tokenizer: Optional[object] = field(default=None, repr=False)
    _model: Optional[object] = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)

    def _load_model(self):
        if self._loaded:
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.primary_model)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.primary_model)
            self._model.eval()
            self._loaded = True
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(self.fallback_model)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.fallback_model)
            self._model.eval()
            self._loaded = True

    def score(self, query: str, context: str, response: str) -> float:
        self._load_model()
        return self._predict(query, context, response)

    def _predict(self, query: str, context: str, response: str) -> float:
        claims = self._decompose(response)
        if not claims:
            return 0.5
        scores = [self._score_claim(context, claim) for claim in claims]
        return sum(scores) / len(scores)

    def _decompose(self, text: str) -> list:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _score_claim(self, context: str, claim: str) -> float:
        inputs = self._tokenizer(
            context, claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # contradiction=0, neutral=1, entailment=2
        return 1.0 - probs[0][0].item()

    def get_confidence(self, query: str, context: str, response: str) -> float:
        return abs(self.score(query, context, response) - 0.5) * 2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nli.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_nli.py src/conformal_rag/scoring.py
git commit -m "feat: implement real NLI scorer with DeBERTa-v3-large"
```

---

### Task 2: LLM Judge — litellm Integration

**Files:**
- Create: `tests/test_llm_judge.py`
- Modify: `src/conformal_rag/llm.py`

**Interfaces:**
- Consumes: `LLMConfig` (existing)
- Produces: `LLMClient.complete(prompt) -> str`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm_judge.py
import pytest
from unittest.mock import patch, MagicMock
from conformal_rag.llm import LLMClient, LLMConfig


class TestLLMClient:
    def setup_method(self):
        self.config = LLMConfig()

    @patch("conformal_rag.llm.litellm")
    def test_complete_returns_string(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.85"
        mock_litellm.completion.return_value = mock_response

        client = LLMClient(self.config)
        result = client.complete("Test prompt")
        assert result == "0.85"

    @patch("conformal_rag.llm.litellm")
    def test_complete_uses_temperature_0(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.9"
        mock_litellm.completion.return_value = mock_response

        client = LLMClient(self.config)
        client.complete("Test prompt")
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    @patch("conformal_rag.llm.litellm")
    def test_fallback_on_first_provider_failure(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            Exception("Groq failed"),
            MagicMock(choices=[MagicMock()]),
        ]
        mock_litellm.completion.return_value.choices[0].message.content = "0.7"

        client = LLMClient(self.config)
        result = client.complete("Test prompt")
        assert mock_litellm.completion.call_count == 2

    @patch("conformal_rag.llm.litellm")
    def test_all_providers_fail_raises(self, mock_litellm):
        mock_litellm.completion.side_effect = Exception("All failed")

        client = LLMClient(self.config)
        with pytest.raises(RuntimeError):
            client.complete("Test prompt")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_judge.py -v`
Expected: FAIL (`litellm` not imported, `_call_provider` raises `NotImplementedError`)

- [ ] **Step 3: Implement litellm integration**

```python
# src/conformal_rag/llm.py — replace entire file
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import litellm


@dataclass
class LLMConfig:
    """LLM configuration."""

    temperature: float = 0.0
    max_tokens: int = 10
    timeout: float = 30.0
    fallback_chain: List[str] = field(default_factory=lambda: [
        "groq/llama-3.3-70b-versatile",
        "openrouter/meta-llama/llama-3.1-8b-instruct:free",
        "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "gemini/gemini-2.5-flash",
    ])


class LLMClient:
    """Multi-provider LLM client with fallback routing."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.providers = self.config.fallback_chain

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 10,
    ) -> str:
        last_error = None
        for provider in self.providers:
            try:
                response = self._call_provider(provider, prompt, temperature, max_tokens)
                return response
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def _call_provider(
        self,
        provider: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 10,
    ) -> str:
        response = litellm.completion(
            model=provider,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def close(self):
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_judge.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_llm_judge.py src/conformal_rag/llm.py
git commit -m "feat: implement litellm integration with fallback chain"
```

---

### Task 3: Fix Confidence Interval Formula

**Files:**
- Create: `tests/test_engine_fix.py`
- Modify: `src/conformal_rag/engine.py`

**Interfaces:**
- Consumes: `ScoringPipeline.score()` (from Task 1)
- Produces: `ConformalResult.confidence_interval` (correct quantile-based)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_engine_fix.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import ScoringBreakdown


class TestEngineCIFix:
    def setup_method(self):
        self.engine = ConformalRAGEngine(alpha=0.05)

    @patch.object(self.engine, 'scoring')
    def test_confidence_interval_uses_quantile(self, mock_scoring):
        mock_scoring.score.return_value = ScoringBreakdown(
            nli_score=0.9, final_score=0.9, nli_latency_ms=10.0, used_llm_judge=False
        )
        # Calibrate with known scores
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.engine._calibration_scores = scores
        self.engine._calibration_labels = labels
        self.engine._threshold = np.percentile(scores, 95)

        result = self.engine.predict("test", "context", "response")
        ci = result.confidence_interval
        assert ci[0] >= 0.0
        assert ci[1] <= 1.0
        assert ci[0] < ci[1]

    def test_confidence_interval_empty_calibration(self):
        result = self.engine.predict("test", "context", "response")
        ci = result.confidence_interval
        assert ci == (0.0, 1.0)

    @patch.object(self.engine, 'scoring')
    def test_confidence_interval_not_proportion_ci(self, mock_scoring):
        mock_scoring.score.return_value = ScoringBreakdown(
            nli_score=0.9, final_score=0.9, nli_latency_ms=10.0, used_llm_judge=False
        )
        # Large calibration set
        np.random.seed(42)
        scores = np.random.uniform(0.1, 0.9, 1000).tolist()
        labels = [0] * 1000
        self.engine._calibration_scores = scores
        self.engine._calibration_labels = labels
        self.engine._threshold = np.percentile(scores, 95)

        result = self.engine.predict("test", "context", "response")
        ci = result.confidence_interval
        # Quantile CI should be tighter than proportion CI for large n
        assert ci[1] - ci[0] < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_engine_fix.py -v`
Expected: FAIL (CI formula uses proportion, not quantile)

- [ ] **Step 3: Fix confidence interval formula**

```python
# src/conformal_rag/engine.py — replace _compute_confidence_interval method and its usage
# In predict() method, replace lines 134-140 with:

        # Confidence interval (conformal quantile-based)
        ci = self._compute_confidence_interval(
            self._calibration_scores, self._calibration_labels, self.alpha
        )
        confidence_interval = ci

# Add this method to ConformalRAGEngine class:

    def _compute_confidence_interval(
        self, scores: list, labels: list, alpha: float
    ) -> tuple:
        if not scores:
            return (0.0, 1.0)
        correct_scores = [s for s, l in zip(scores, labels) if l == 0]
        if not correct_scores:
            return (0.0, 1.0)
        quantile = float(np.quantile(correct_scores, 1 - alpha))
        return (quantile, 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_engine_fix.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_engine_fix.py src/conformal_rag/engine.py
git commit -m "fix: replace proportion CI with conformal quantile CI"
```

---

### Task 4: Fix FDR Label Mapping

**Files:**
- Create: `tests/test_online_fix.py`
- Modify: `src/conformal_rag/online.py`

**Interfaces:**
- Consumes: `OnlineConformal.update(score, is_correct)` (existing)
- Produces: Correct FDR computation

- [ ] **Step 1: Write failing tests**

```python
# tests/test_online_fix.py
import pytest
from conformal_rag.online import OnlineConformal


class TestOnlineFDRFix:
    def setup_method(self):
        self.oc = OnlineConformal(alpha=0.05, window_size=100)

    def test_fdr_computed_correctly(self):
        # Add 80 correct (label=0, is_correct=True) and 20 hallucinations (label=1, is_correct=False)
        for _ in range(80):
            self.oc.update(0.3, is_correct=True)
        for _ in range(20):
            self.oc.update(0.8, is_correct=False)

        fdr = self.oc._compute_fdr()
        # With threshold ~0.5, all 20 hallucinations are rejected (score > threshold)
        # No correct samples are rejected (score < threshold)
        # FDR = 0 / 20 = 0.0
        assert fdr == 0.0

    def test_fdr_with_false_positives(self):
        # All scores above threshold, mix of correct and hallucination
        for _ in range(50):
            self.oc.update(0.9, is_correct=True)  # FP: correct but high score
        for _ in range(50):
            self.oc.update(0.9, is_correct=False)  # TP: hallucination with high score

        fdr = self.oc._compute_fdr()
        # All 100 rejected, 50 are FP (correct with high score)
        # FDR = 50 / 100 = 0.5
        assert abs(fdr - 0.5) < 0.01

    def test_threshold_adjusts_correctly(self):
        # Start with high FDR, threshold should increase
        for _ in range(100):
            self.oc.update(0.9, is_correct=True)  # All FP

        fdr_before = self.oc._compute_fdr()
        # FDR should be 1.0 (all rejected are correct)
        assert fdr_before == 1.0

        # After update, threshold should increase
        assert self.oc.threshold > 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_online_fix.py -v`
Expected: FAIL (`labels[i] is True` counts wrong samples as FP)

- [ ] **Step 3: Fix FDR label mapping**

```python
# src/conformal_rag/online.py — line 76, replace:
# Old: fp = sum(1 for i in rejected_idx if labels[i] is True)
# New:
        fp = sum(1 for i in rejected_idx if labels[i] is False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_online_fix.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_online_fix.py src/conformal_rag/online.py
git commit -m "fix: correct FDR label mapping in online conformal"
```

---

### Task 5: Qdrant Retrieval — Real Hybrid Search

**Files:**
- Create: `tests/test_retrieval_real.py`
- Modify: `src/conformal_rag/retrieval.py`

**Interfaces:**
- Consumes: `Document` from `models.py`
- Produces: `HybridRetriever.retrieve(query) -> list[str]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_retrieval_real.py
import pytest
from unittest.mock import MagicMock, patch
from conformal_rag.retrieval import HybridRetriever, reciprocal_rank_fusion
from conformal_rag.models import Document


class TestRRF:
    def test_rrf_merges_rankings(self):
        ranking1 = ["doc_a", "doc_b", "doc_c"]
        ranking2 = ["doc_b", "doc_c", "doc_d"]
        result = reciprocal_rank_fusion([ranking1, ranking2])
        assert result[0] == "doc_b"  # Appears in both, highest RRF score

    def test_rrf_empty_rankings(self):
        result = reciprocal_rank_fusion([[], []])
        assert result == []


class TestHybridRetriever:
    def setup_method(self):
        self.retriever = HybridRetriever(top_k=5, final_k=3)

    @patch.object(HybridRetriever, '_dense_search')
    @patch.object(HybridRetriever, '_sparse_search')
    def test_retrieval_returns_results(self, mock_sparse, mock_dense):
        mock_dense.return_value = ["doc1", "doc2", "doc3"]
        mock_sparse.return_value = ["doc2", "doc3", "doc4"]
        results = self.retriever.retrieve("test query")
        assert len(results) > 0
        assert len(results) <= 3

    def test_dense_search_returns_list(self):
        results = self.retriever._dense_search("test")
        assert isinstance(results, list)

    def test_sparse_search_returns_list(self):
        results = self.retriever._sparse_search("test")
        assert isinstance(results, list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retrieval_real.py -v`
Expected: FAIL (`_dense_search` and `_sparse_search` return `[]`, retrieval returns empty)

- [ ] **Step 3: Implement Qdrant retrieval**

```python
# src/conformal_rag/retrieval.py — replace entire file
from __future__ import annotations

from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60) -> List[str]:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


class HybridRetriever:
    """Hybrid dense + sparse retrieval with RRF.

    Supports two modes:
    - In-memory: documents stored locally, no Qdrant needed
    - Qdrant: production vector store with hybrid search
    """

    def __init__(
        self,
        top_k: int = 10,
        final_k: int = 5,
        qdrant_url: Optional[str] = None,
        collection: str = "docs",
    ):
        self.top_k = top_k
        self.final_k = final_k
        self.qdrant_url = qdrant_url
        self.collection = collection
        self._documents: List[str] = []
        self._embedder = None
        self._client = None

        if qdrant_url:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=qdrant_url)

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("intfloat/e5-base-v2")
        return self._embedder

    def index(self, documents: List) -> None:
        for doc in documents:
            self._documents.append(doc.text)

    def retrieve(self, query: str) -> List[str]:
        dense_results = self._dense_search(query)
        sparse_results = self._sparse_search(query)
        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        return fused[: self.final_k]

    def _dense_search(self, query: str) -> List[str]:
        if self._client:
            return self._qdrant_dense_search(query)
        return self._in_memory_dense_search(query)

    def _sparse_search(self, query: str) -> List[str]:
        if self._client:
            return self._qdrant_sparse_search(query)
        return self._in_memory_sparse_search(query)

    def _in_memory_dense_search(self, query: str) -> List[str]:
        if not self._documents:
            return []
        embedder = self._get_embedder()
        query_emb = embedder.encode(query)
        doc_embs = embedder.encode(self._documents)
        similarities = np.dot(doc_embs, query_emb) / (
            np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        return [self._documents[i] for i in top_indices]

    def _in_memory_sparse_search(self, query: str) -> List[str]:
        if not self._documents:
            return []
        tokenized_docs = [doc.lower().split() for doc in self._documents]
        bm25 = BM25Okapi(tokenized_docs)
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        return [self._documents[i] for i in top_indices]

    def _qdrant_dense_search(self, query: str) -> List[str]:
        embedder = self._get_embedder()
        embedding = embedder.encode(query).tolist()
        results = self._client.search(
            collection_name=self.collection,
            query_vector=("dense", embedding),
            limit=self.top_k,
        )
        return [r.payload["text"] for r in results]

    def _qdrant_sparse_search(self, query: str) -> List[str]:
        from qdrant_client.models import SparseVector
        tokens = query.lower().split()
        sparse = SparseVector(
            indices=list(range(len(tokens))),
            values=[1.0] * len(tokens),
        )
        results = self._client.search(
            collection_name=self.collection,
            query_vector=("sparse", sparse),
            limit=self.top_k,
        )
        return [r.payload["text"] for r in results]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retrieval_real.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_retrieval_real.py src/conformal_rag/retrieval.py
git commit -m "feat: implement Qdrant hybrid retrieval with in-memory fallback"
```

---

### Task 6: Update ScoringPipeline to Use Real NLI

**Files:**
- Modify: `src/conformal_rag/scoring.py`
- Create: `tests/test_scoring_integration.py`

**Interfaces:**
- Consumes: `NLIScorer` (from Task 1), `LLMJudge` (from Task 2)
- Produces: `ScoringPipeline.score(query, context, response) -> ScoringBreakdown`

- [ ] **Step 1: Write failing test**

```python
# tests/test_scoring_integration.py
import pytest
from unittest.mock import patch, MagicMock
from conformal_rag.scoring import ScoringPipeline
from conformal_rag.models import ScoringBreakdown


class TestScoringPipelineIntegration:
    def setup_method(self):
        self.pipeline = ScoringPipeline(use_llm_judge=False)

    @patch.object(self.pipeline.nli, '_predict')
    def test_pipeline_returns_scoring_breakdown(self, mock_predict):
        mock_predict.return_value = 0.85
        result = self.pipeline.score("query", "context", "response")
        assert isinstance(result, ScoringBreakdown)
        assert result.nli_score == 0.85
        assert result.final_score == 0.85
        assert result.used_llm_judge is False

    @patch.object(self.pipeline.nli, '_predict')
    def test_pipeline_with_low_confidence_triggers_llm(self, mock_predict):
        mock_predict.return_value = 0.5  # Low confidence
        pipeline = ScoringPipeline(use_llm_judge=True, nli_confidence_threshold=0.3)
        with patch.object(pipeline.llm_judge, 'score') as mock_llm:
            mock_llm.return_value = 0.9
            result = pipeline.score("query", "context", "response")
            assert result.used_llm_judge is True
            assert result.llm_judge_score == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring_integration.py -v`
Expected: FAIL (NLI returns 0.5, confidence check broken)

- [ ] **Step 3: Verify ScoringPipeline works with real NLI**

The `ScoringPipeline` class already calls `self.nli.score()` which now uses real DeBERTa-v3. No changes needed to `ScoringPipeline` itself — it works automatically after Task 1.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scoring_integration.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_scoring_integration.py
git commit -m "test: add scoring pipeline integration tests"
```

---

### Task 7: Integration Test — End-to-End Pipeline

**Files:**
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: All previous tasks
- Produces: End-to-end test proving the pipeline works

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
import pytest
from unittest.mock import patch, MagicMock
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.retrieval import HybridRetriever
from conformal_rag.models import ScoringBreakdown, Document


class TestEndToEnd:
    def setup_method(self):
        self.engine = ConformalRAGEngine(alpha=0.05)
        self.retriever = HybridRetriever()

    @patch.object(self.engine.scoring, 'score')
    def test_full_pipeline(self, mock_score):
        # Simulate calibration
        mock_score.return_value = ScoringBreakdown(
            nli_score=0.9, final_score=0.9, nli_latency_ms=10.0, used_llm_judge=False
        )
        queries = ["q"] * 10
        contexts = ["c"] * 10
        responses = ["r"] * 10
        labels = [0] * 8 + [1] * 2

        stats = self.engine.calibrate(queries, contexts, responses, labels)
        assert stats.n_samples == 10
        assert stats.estimated_fdr <= 0.1

        # Predict
        mock_score.return_value = ScoringBreakdown(
            nli_score=0.85, final_score=0.85, nli_latency_ms=10.0, used_llm_judge=False
        )
        result = self.engine.predict("test query", "test context", "test response")
        assert 0.0 <= result.risk_score <= 1.0
        assert isinstance(result.abstained, bool)

    def test_retrieval_integration(self):
        docs = [
            Document(text="Paris is the capital of France.", domain="geography"),
            Document(text="Python is a programming language.", domain="tech"),
        ]
        self.retriever.index(docs)
        results = self.retriever.retrieve("capital of France")
        assert len(results) > 0
        assert any("Paris" in r for r in results)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_integration.py -v`
Expected: PASS (2 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests"
```

---

### Task 8: Update Package Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add new dependencies**

```toml
# Add to [project] dependencies:
dependencies = [
    ...
    "transformers>=4.36",
    "torch>=2.1",
    "litellm>=1.30",
    "qdrant-client>=1.7",
    "sentence-transformers>=2.2",
    "rank-bm25>=0.2",
    "numpy>=1.24",
]
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -e ".[dev]"`
Expected: Success

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (75 original + ~20 new = ~95 total)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add transformers, litellm, qdrant-client, sentence-transformers"
```

---

### Task 9: Update __init__.py Exports

**Files:**
- Modify: `src/conformal_rag/__init__.py`

- [ ] **Step 1: Update exports**

```python
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.scoring import ScoringPipeline, NLIScorer, LLMJudge
from conformal_rag.retrieval import HybridRetriever
from conformal_rag.online import OnlineConformal
from conformal_rag.client import ConformalRAG
from conformal_rag.models import ConformalResult, CalibrationStats, Document

__all__ = [
    "ConformalRAGEngine",
    "ScoringPipeline",
    "NLIScorer",
    "LLMJudge",
    "HybridRetriever",
    "OnlineConformal",
    "ConformalRAG",
    "ConformalResult",
    "CalibrationStats",
    "Document",
]
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from conformal_rag import ConformalRAGEngine, NLIScorer, HybridRetriever; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/conformal_rag/__init__.py
git commit -m "feat: update package exports with new components"
```

---

### Task 10: Final Verification — All Tests Pass

**Files:**
- None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Verify package installs**

Run: `pip install -e ".[dev]" && python -c "from conformal_rag import ConformalRAG; print('Package OK')"`
Expected: `Package OK`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: Phase 1 complete — all critical fixes verified"
```

---

*Plan complete. 10 tasks, TDD throughout, each task independently testable.*
