# Phase 1 Design: Critical Fixes

**Date**: July 3, 2026
**Status**: Approved
**Parent spec**: `2026-07-03-conformal-rag-design.md`

---

## 1. Problem

The conformal-rag library has 5 critical issues that make the core non-functional:

1. NLI scorer returns constant `0.5` (placeholder)
2. LLM judge raises `NotImplementedError` (stub)
3. Confidence interval formula uses proportion CI instead of conformal quantile
4. FDR computation in online.py has inverted label mapping
5. Retrieval returns empty lists (no vector store integration)

---

## 2. Fix 1: Real NLI Scorer

**File**: `src/conformal_rag/scoring.py`

**Current**: `_predict()` returns `0.5` always.

**New**: Load DeBERTa-v3-large MNLI model, real inference with claim decomposition.

```python
class NLIScorer:
    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def _predict(self, query: str, context: str, response: str) -> float:
        claims = self._decompose(response)
        scores = [self._score_claim(context, claim) for claim in claims]
        return sum(scores) / len(scores) if scores else 0.5

    def _decompose(self, text: str) -> list[str]:
        # Split into sentences, then atomic claims
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _score_claim(self, context: str, claim: str) -> float:
        inputs = self.tokenizer(context, claim, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # contradiction=0, neutral=1, entailment=2
        return 1.0 - probs[0][0].item()  # faithfulness = 1 - contradiction
```

**Performance**: ~486ms CPU. Batch size 32 for calibration.

**Dependencies**: `transformers>=4.36`, `torch>=2.1`

---

## 3. Fix 2: Real LLM Judge

**File**: `src/conformal_rag/llm.py`

**Current**: `_call_provider()` raises `NotImplementedError`.

**New**: Use `litellm.completion()` with temperature=0.0.

```python
class LLMClient:
    def _call_provider(self, provider, prompt, temperature=0.0, max_tokens=10):
        import litellm
        response = litellm.completion(
            model=provider,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
```

**Key constraints**:
- `temperature=0.0` (deterministic scoring)
- `max_tokens=10` (score is a single float)
- Fallback chain: Groq → OpenRouter → Together → Gemini

---

## 4. Fix 3: Confidence Interval Formula

**File**: `src/conformal_rag/engine.py`

**Current**: Uses proportion CI: `1.96 * sqrt(threshold * (1-threshold) / n)` — wrong for conformal scores.

**New**: Use conformal quantile:

```python
def _compute_confidence_interval(self, scores, labels, alpha):
    correct_scores = [s for s, l in zip(scores, labels) if l == 0]
    if not correct_scores:
        return (0.0, 1.0)
    quantile = np.quantile(correct_scores, 1 - alpha)
    return (quantile, 1.0)
```

**Why**: Conformal prediction controls FDR via quantile of non-conformity scores, not via proportion standard errors.

---

## 5. Fix 4: FDR Label Mapping

**File**: `src/conformal_rag/online.py`

**Current** (line 76):
```python
fp = sum(1 for i in rejected_idx if labels[i] is True)
```

This counts `label=1` (hallucination) as "correct", so FP is wrong.

**New**:
```python
fp = sum(1 for i in rejected_idx if labels[i] == 0)
```

`label=0` = correct. FP = rejected but actually correct.

---

## 6. Fix 5: Qdrant Retrieval

**File**: `src/conformal_rag/retrieval.py`

**Current**: `_dense_search()` and `_sparse_search()` return `[]`.

**New**: Real Qdrant integration with hybrid search.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, VectorParams, Distance
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self, qdrant_url="http://localhost:6333", collection="docs"):
        self.client = QdrantClient(url=qdrant_url)
        self.collection = collection
        self.embedder = SentenceTransformer("intfloat/e5-base-v2")

    def index(self, documents: list[Document]):
        # Create collection with dense + sparse vectors
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        # Index each document
        for doc in documents:
            dense = self.embedder.encode(doc.text).tolist()
            sparse = self._bm25_encode(doc.text)
            self.client.upsert(
                collection_name=self.collection,
                points=[PointStruct(
                    id=doc.doc_id,
                    vector={"dense": dense, "sparse": sparse},
                    payload={"text": doc.text, "metadata": doc.metadata},
                )],
            )

    def _dense_search(self, query):
        embedding = self.embedder.encode(query).tolist()
        results = self.client.search(
            collection_name=self.collection,
            query_vector=("dense", embedding),
            limit=self.top_k,
        )
        return [r.payload["text"] for r in results]

    def _sparse_search(self, query):
        sparse = self._bm25_encode(query)
        results = self.client.search(
            collection_name=self.collection,
            query_vector=("sparse", sparse),
            limit=self.top_k,
        )
        return [r.payload["text"] for r in results]
```

---

## 7. Input/Output Interface

No frontend required. Three input methods:

### 7.1 Python SDK (Primary)
```python
from conformal_rag import ConformalRAG

rag = ConformalRAG(alpha=0.05, llm_provider="groq")
rag.index(documents)
result = rag.query("What is the capital of France?")
print(result.response, result.risk_score, result.abstained)
```

### 7.2 REST API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "context": ["Paris is the capital."]}'
```

### 7.3 CLI (optional, Phase 2)
```bash
conformal-rag predict "What is the capital of France?" --alpha 0.05
```

---

## 8. Testing Strategy

TDD for each fix:
1. **NLI**: Test `_score_claim()` returns 0.0-1.0, test decomposition
2. **LLM judge**: Mock `litellm.completion`, test fallback chain
3. **CI formula**: Test quantile computation on known distributions
4. **FDR fix**: Test with known label arrays
5. **Retrieval**: Test with in-memory Qdrant (`:memory:`)

---

## 9. Dependencies

```
transformers>=4.36    # DeBERTa-v3 NLI
torch>=2.1            # Model inference
litellm>=1.30         # Multi-provider LLM routing
qdrant-client>=1.7    # Vector store
sentence-transformers>=2.2  # Embeddings
rank-bm25>=0.2        # BM25 sparse encoding
```

---

*Phase 1 design — 5 critical fixes to make the core functional*
