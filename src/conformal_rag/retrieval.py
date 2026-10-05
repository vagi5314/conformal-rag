from __future__ import annotations

import json
import logging
import math
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# intfloat/e5 models expect instruction prefixes; omitting them degrades
# embedding quality measurably.
_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "

# Shared across retriever instances so the model loads once per process.
_embedder_cache: dict = {}

# Sparse vocabulary size. Tokens map to slots via crc32 (stable across
# processes, unlike Python's salted hash()); collisions merge weights.
_SPARSE_DIM = 32768


def get_embedder(model_name: str):
    """Lazily load a SentenceTransformer, cached per model name."""
    if model_name not in _embedder_cache:
        from sentence_transformers import SentenceTransformer
        _embedder_cache[model_name] = SentenceTransformer(model_name)
        logger.info("Embedder loaded: %s", model_name)
    return _embedder_cache[model_name]


def encode_passages(embedder, texts: List[str]) -> np.ndarray:
    """Encode texts as L2-normalized passage embeddings."""
    emb = embedder.encode([_PASSAGE_PREFIX + t for t in texts])
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def encode_query(embedder, query: str) -> np.ndarray:
    """Encode a single query as an L2-normalized embedding."""
    emb = np.asarray(embedder.encode(_QUERY_PREFIX + query), dtype=np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def sparse_slot(token: str) -> int:
    """Stable token -> sparse-vector slot. crc32 is process-independent."""
    return zlib.crc32(token.encode("utf-8")) % _SPARSE_DIM


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60) -> List[str]:
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        top_k: int = 10,
        final_k: int = 5,
        qdrant_url: Optional[str] = None,
        collection: str = "docs",
        embedding_model: str = "intfloat/e5-base-v2",
    ):
        self.top_k = top_k
        self.final_k = final_k
        self.qdrant_url = qdrant_url
        self.collection = collection
        self.embedding_model = embedding_model
        self._documents: List[str] = []
        self._tokenized_docs: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._embedder = None
        self._doc_embeddings: Optional[np.ndarray] = None  # L2-normalized
        self._client = None

        if qdrant_url:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=qdrant_url)

    def _get_embedder(self):
        # Instance attribute first: tests inject fakes here, and a previously
        # loaded model is kept on the instance.
        if self._embedder is None:
            self._embedder = get_embedder(self.embedding_model)
        return self._embedder

    def _encode_passages(self, texts: List[str]) -> np.ndarray:
        return encode_passages(self._get_embedder(), texts)

    def _encode_query(self, query: str) -> np.ndarray:
        return encode_query(self._get_embedder(), query)

    def index(self, documents: List) -> None:
        for doc in documents:
            self._documents.append(doc.text)
        self._tokenized_docs = [doc.lower().split() for doc in self._documents]
        # BM25 is always built: its idf/term stats weight the sparse vectors
        # even when the vectors live in Qdrant.
        self._bm25 = BM25Okapi(self._tokenized_docs) if self._tokenized_docs else None
        if not self._documents:
            return
        if self._client:
            self._ensure_collection()
            # index() appends; force re-embedding of all accumulated documents
            # so a second call on the same retriever doesn't reuse a stale array.
            self._doc_embeddings = None
            self._upsert_all()
        else:
            # Precompute dense embeddings once; queries then only encode themselves.
            self._doc_embeddings = self._encode_passages(self._documents)

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
        if self._doc_embeddings is None:
            self._doc_embeddings = self._encode_passages(self._documents)
        query_emb = self._encode_query(query)
        similarities = self._doc_embeddings @ query_emb
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        return [self._documents[i] for i in top_indices]

    def _in_memory_sparse_search(self, query: str) -> List[str]:
        if not self._documents or self._bm25 is None:
            return []
        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        return [self._documents[i] for i in top_indices]

    # ---------- Qdrant ----------

    def _ensure_collection(self) -> None:
        """Recreate the collection with dense + sparse named vectors.

        Recreating matches the replace semantics of index(): the API builds a
        fresh retriever per /index call, and _upsert_all re-adds every
        accumulated document, so no stale points survive.
        """
        from qdrant_client.models import Distance, SparseVectorParams, VectorParams
        dim = self._get_embedder().get_embedding_dimension()
        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

    def _doc_sparse_vector(self, doc_i: int):
        """BM25 term weights for one document as a Qdrant SparseVector.

        Weight per term is its BM25 contribution (idf clamped at 0 times the
        saturated tf factor), matching what get_scores sums over query terms.
        """
        from qdrant_client.models import SparseVector
        doc_tokens = self._bm25.doc_freqs[doc_i]
        dl = self._bm25.doc_len[doc_i]
        slots: Dict[int, float] = {}
        for token, tf in doc_tokens.items():
            idf = max(self._bm25.idf.get(token, 0.0), 0.0)
            if idf <= 0.0 or tf <= 0:
                continue
            denom = tf + self._bm25.k1 * (
                1 - self._bm25.b + self._bm25.b * dl / self._bm25.avgdl
            )
            weight = idf * (tf * (self._bm25.k1 + 1)) / denom
            slot = sparse_slot(token)
            slots[slot] = slots.get(slot, 0.0) + weight
        indices = sorted(slots)
        return SparseVector(indices=indices, values=[slots[i] for i in indices])

    def _query_sparse_vector(self, query: str):
        """Sparse query vector: idf * (1 + log tf) for in-vocabulary tokens.

        Returns None when no query token can match anything indexed; callers
        must skip the Qdrant call in that case.
        """
        from qdrant_client.models import SparseVector
        if self._bm25 is None:
            return None
        counts: Dict[str, int] = {}
        for t in query.lower().split():
            counts[t] = counts.get(t, 0) + 1
        slots: Dict[int, float] = {}
        for token, tf in counts.items():
            idf = self._bm25.idf.get(token)
            if idf is None or idf <= 0.0:
                continue
            weight = idf * (1.0 + math.log(tf))
            slot = sparse_slot(token)
            slots[slot] = slots.get(slot, 0.0) + weight
        if not slots:
            return None
        indices = sorted(slots)
        return SparseVector(indices=indices, values=[slots[i] for i in indices])

    def _upsert_all(self) -> None:
        from qdrant_client.models import PointStruct
        if not self._documents:
            return
        if self._doc_embeddings is None:
            self._doc_embeddings = self._encode_passages(self._documents)
        batch = 64
        for start in range(0, len(self._documents), batch):
            points = [
                PointStruct(
                    id=i,
                    vector={
                        "dense": self._doc_embeddings[i].tolist(),
                        "sparse": self._doc_sparse_vector(i),
                    },
                    payload={"text": self._documents[i]},
                )
                for i in range(start, min(start + batch, len(self._documents)))
            ]
            self._client.upsert(collection_name=self.collection, points=points)

    def _qdrant_dense_search(self, query: str) -> List[str]:
        embedding = self._encode_query(query).tolist()
        resp = self._client.query_points(
            collection_name=self.collection,
            query=embedding,
            using="dense",
            limit=self.top_k,
        )
        return [p.payload.get("text", "") for p in resp.points if p.payload.get("text")]

    def _qdrant_sparse_search(self, query: str) -> List[str]:
        sparse = self._query_sparse_vector(query)
        if sparse is None:
            return []
        resp = self._client.query_points(
            collection_name=self.collection,
            query=sparse,
            using="sparse",
            limit=self.top_k,
        )
        return [p.payload.get("text", "") for p in resp.points if p.payload.get("text")]

    # ---------- persistence ----------

    def save_state(self, path: Union[str, Path]) -> None:
        """Persist the in-memory index (documents + embeddings) to JSON."""
        state = {
            "version": 1,
            "documents": self._documents,
            "embeddings": (
                self._doc_embeddings.tolist()
                if self._doc_embeddings is not None else None
            ),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
        logger.info("Saved index state to %s (%d docs)", path, len(self._documents))

    def load_state(self, path: Union[str, Path]) -> bool:
        """Restore the index from JSON. Returns True if loaded."""
        path = Path(path)
        if not path.is_file():
            return False
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            documents = state.get("documents", [])
            self._documents = list(documents)
            self._tokenized_docs = [d.lower().split() for d in self._documents]
            self._bm25 = BM25Okapi(self._tokenized_docs) if self._tokenized_docs else None
            embs = state.get("embeddings")
            self._doc_embeddings = (
                np.asarray(embs, dtype=np.float32) if embs is not None else None
            )
            logger.info("Loaded index state from %s (%d docs)", path, len(self._documents))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("Could not load index state from %s: %s", path, e)
            return False
        if self._client and self._documents:
            # Re-populate Qdrant (idempotent): covers server restarts with
            # empty storage. Startup must not crash if the server is down.
            try:
                self._ensure_collection()
                self._upsert_all()
            except Exception as e:
                logger.warning("Qdrant re-index after load failed: %s", e)
        return True
