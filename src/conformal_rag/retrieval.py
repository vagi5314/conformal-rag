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
