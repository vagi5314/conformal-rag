"""Tests for hybrid retrieval."""

import pytest
from unittest.mock import patch, MagicMock
from conformal_rag.retrieval import HybridRetriever, reciprocal_rank_fusion


class TestRRF:
    def test_basic_rrf(self):
        rankings = [["a", "b", "c"], ["b", "a", "d"]]
        result = reciprocal_rank_fusion(rankings)
        assert result[0] in ["a", "b"]  # Both rank high

    def test_single_ranking(self):
        rankings = [["x", "y", "z"]]
        result = reciprocal_rank_fusion(rankings)
        assert result == ["x", "y", "z"]

    def test_empty_rankings(self):
        result = reciprocal_rank_fusion([])
        assert result == []


class TestHybridRetriever:
    def test_initializes(self):
        retriever = HybridRetriever()
        assert retriever.top_k == 10

    def test_retrieves_with_rrf(self):
        retriever = HybridRetriever()
        with patch.object(retriever, "_dense_search", return_value=["a", "b", "c"]), \
             patch.object(retriever, "_sparse_search", return_value=["b", "a", "d"]):
            results = retriever.retrieve("test query")
            assert len(results) > 0
