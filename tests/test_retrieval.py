"""Tests for hybrid retrieval."""

import zlib

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from conformal_rag.models import Document
from conformal_rag.retrieval import (
    HybridRetriever,
    reciprocal_rank_fusion,
    sparse_slot,
    _SPARSE_DIM,
)


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


class TestSparseSlot:
    def test_stable_across_calls(self):
        assert sparse_slot("hello") == sparse_slot("hello")

    def test_matches_crc32(self):
        token = "conformal"
        assert sparse_slot(token) == zlib.crc32(token.encode("utf-8")) % _SPARSE_DIM

    def test_within_bounds(self):
        for token in ["a", "the", "conformal", "prediction", "x" * 100]:
            assert 0 <= sparse_slot(token) < _SPARSE_DIM


def _indexed_retriever(docs):
    retriever = HybridRetriever()
    retriever.index([Document(text=d) for d in docs])
    return retriever


class TestSparseVectors:
    DOCS = [
        "Paris is the capital of France",
        "Python is a programming language",
        "The sun is a star in the galaxy",
    ]

    def test_doc_sparse_uses_crc32_slots(self):
        retriever = _indexed_retriever(self.DOCS)
        vec = retriever._doc_sparse_vector(0)
        expected_slot = sparse_slot("paris")
        assert expected_slot in vec.indices

    def test_doc_sparse_weights_are_positive(self):
        retriever = _indexed_retriever(self.DOCS)
        for i in range(len(self.DOCS)):
            vec = retriever._doc_sparse_vector(i)
            assert all(v > 0 for v in vec.values)
            assert len(vec.indices) == len(vec.values)

    def test_rare_term_outweighs_ubiquitous_term(self):
        # "is" appears in every doc (idf floored near zero); "paris" in one.
        retriever = _indexed_retriever(self.DOCS)
        vec = retriever._doc_sparse_vector(0)
        weights = dict(zip(vec.indices, vec.values))
        assert weights[sparse_slot("paris")] > weights[sparse_slot("is")]

    def test_doc_sparse_matches_bm25_formula(self):
        retriever = _indexed_retriever(self.DOCS)
        bm25 = retriever._bm25
        vec = retriever._doc_sparse_vector(0)
        weights = dict(zip(vec.indices, vec.values))
        token = "paris"
        tf = bm25.doc_freqs[0][token]
        dl = bm25.doc_len[0]
        expected = max(bm25.idf[token], 0.0) * (tf * (bm25.k1 + 1)) / (
            tf + bm25.k1 * (1 - bm25.b + bm25.b * dl / bm25.avgdl)
        )
        assert weights[sparse_slot(token)] == pytest.approx(expected)

    def test_query_sparse_skips_unknown_tokens(self):
        retriever = _indexed_retriever(self.DOCS)
        vec = retriever._query_sparse_vector("paris zzznotindexed")
        assert sparse_slot("paris") in vec.indices
        assert sparse_slot("zzznotindexed") not in vec.indices

    def test_query_sparse_none_when_no_vocabulary_match(self):
        retriever = _indexed_retriever(self.DOCS)
        assert retriever._query_sparse_vector("qqq www eee") is None

    def test_query_sparse_none_without_bm25(self):
        retriever = HybridRetriever()
        assert retriever._query_sparse_vector("anything") is None


class TestQdrantSearchGuards:
    DOCS = [
        "Paris is the capital of France",
        "Python is a programming language",
        "The sun is a star in the galaxy",
    ]

    def test_sparse_search_skips_client_when_no_match(self):
        retriever = _indexed_retriever(self.DOCS)
        retriever._client = MagicMock()
        results = retriever._qdrant_sparse_search("zzz yyy xxx")
        assert results == []
        retriever._client.query_points.assert_not_called()

    def test_sparse_search_calls_query_points_with_sparse_vector(self):
        retriever = _indexed_retriever(self.DOCS)
        retriever._client = MagicMock()
        point = MagicMock()
        point.payload = {"text": "Paris is the capital of France"}
        retriever._client.query_points.return_value = MagicMock(points=[point])
        results = retriever._qdrant_sparse_search("capital of France")
        assert results == ["Paris is the capital of France"]
        kwargs = retriever._client.query_points.call_args.kwargs
        assert kwargs["using"] == "sparse"
        assert kwargs["collection_name"] == retriever.collection

    def test_dense_search_uses_dense_named_vector(self):
        retriever = HybridRetriever()
        retriever._client = MagicMock()
        point = MagicMock()
        point.payload = {"text": "doc"}
        retriever._client.query_points.return_value = MagicMock(points=[point])
        with patch.object(retriever, "_encode_query", return_value=np.ones(4, dtype=np.float32)):
            results = retriever._qdrant_dense_search("anything")
        assert results == ["doc"]
        kwargs = retriever._client.query_points.call_args.kwargs
        assert kwargs["using"] == "dense"

    def test_search_drops_points_without_text(self):
        retriever = HybridRetriever()
        retriever._client = MagicMock()
        good = MagicMock(payload={"text": "doc"})
        bad = MagicMock(payload={})
        retriever._client.query_points.return_value = MagicMock(points=[good, bad])
        with patch.object(retriever, "_encode_query", return_value=np.ones(4, dtype=np.float32)):
            results = retriever._qdrant_dense_search("anything")
        assert results == ["doc"]


class TestQdrantIndexing:
    def test_index_upserts_when_client_set(self):
        retriever = HybridRetriever()
        retriever._client = MagicMock()
        fake_emb = np.ones((2, 4), dtype=np.float32)
        with patch.object(retriever, "_ensure_collection") as ensure, \
             patch.object(retriever, "_encode_passages", return_value=fake_emb):
            retriever.index([Document(text="alpha beta"), Document(text="gamma delta")])
        ensure.assert_called_once()
        retriever._client.upsert.assert_called_once()
        points = retriever._client.upsert.call_args.kwargs["points"]
        assert len(points) == 2
        assert points[0].payload == {"text": "alpha beta"}
        assert "dense" in points[0].vector and "sparse" in points[0].vector

    def test_index_batches_upserts(self):
        retriever = HybridRetriever()
        retriever._client = MagicMock()
        docs = [Document(text=f"doc number {i}") for i in range(150)]
        fake_emb = np.ones((150, 4), dtype=np.float32)
        with patch.object(retriever, "_ensure_collection"), \
             patch.object(retriever, "_encode_passages", return_value=fake_emb):
            retriever.index(docs)
        assert retriever._client.upsert.call_count == 3  # 64 + 64 + 22

    def test_load_state_reindexes_qdrant(self, tmp_path):
        source = HybridRetriever()
        source.index([Document(text="Paris is the capital of France")])
        state_file = tmp_path / "index.json"
        source.save_state(state_file)

        target = HybridRetriever()
        target._client = MagicMock()
        fake_emb = np.ones((1, 4), dtype=np.float32)
        with patch.object(target, "_ensure_collection") as ensure, \
             patch.object(target, "_encode_passages", return_value=fake_emb):
            assert target.load_state(state_file) is True
        ensure.assert_called_once()
        target._client.upsert.assert_called_once()

    def test_load_state_survives_qdrant_outage(self, tmp_path):
        source = HybridRetriever()
        source.index([Document(text="Paris is the capital of France")])
        state_file = tmp_path / "index.json"
        source.save_state(state_file)

        target = HybridRetriever()
        target._client = MagicMock()
        target._client.collection_exists.side_effect = ConnectionError("down")
        assert target.load_state(state_file) is True
        assert target._documents == ["Paris is the capital of France"]
