"""Hybrid retrieval: dense + sparse with RRF."""

from __future__ import annotations

from typing import List


def reciprocal_rank_fusion(
    rankings: List[List[str]], k: int = 60
) -> List[str]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        rankings: List of ranked document lists.
        k: RRF constant (default 60).

    Returns:
        Merged ranking.
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


class HybridRetriever:
    """Hybrid dense + sparse retrieval with RRF."""

    def __init__(self, top_k: int = 10, final_k: int = 5):
        self.top_k = top_k
        self.final_k = final_k

    def retrieve(self, query: str) -> List[str]:
        """Retrieve documents using hybrid search.

        Args:
            query: The search query.

        Returns:
            List of retrieved document texts.
        """
        dense_results = self._dense_search(query)
        sparse_results = self._sparse_search(query)
        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        return fused[: self.final_k]

    def _dense_search(self, query: str) -> List[str]:
        """Dense vector search. Override with Qdrant in production."""
        return []

    def _sparse_search(self, query: str) -> List[str]:
        """Sparse BM25 search. Override with Qdrant in production."""
        return []
