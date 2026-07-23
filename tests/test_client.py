"""Tests for Python client SDK."""

import pytest
from unittest.mock import patch, MagicMock
from conformal_rag.client import ConformalRAG
from conformal_rag.models import ScoringBreakdown


class FakeScoring:
    def __init__(self, score=0.9):
        self._score = score
    def score(self, query, context, response):
        return ScoringBreakdown(
            nli_score=self._score, final_score=self._score,
            nli_latency_ms=1.0, used_llm_judge=False,
        )


def _client():
    c = ConformalRAG(alpha=0.05)
    c.engine.scoring = FakeScoring(score=0.9)
    return c


def test_index_builds_retriever_and_stores_docs():
    c = _client()
    with patch("conformal_rag.client.HybridRetriever"):
        c.index(["Python is a programming language.", "The sky is blue."])
    assert c.retriever is not None
    assert len(c._documents) == 2


def test_retrieve_returns_relevant_context():
    c = _client()
    with patch("conformal_rag.client.HybridRetriever") as MockRetriever:
        mock_ret = MagicMock()
        mock_ret.retrieve.return_value = ["Python is a programming language."]
        MockRetriever.return_value = mock_ret
        c.index(["Python is a programming language."])
        ctx = c.retrieve("What is Python?")
        assert any("Python" in d for d in ctx)


def test_check_returns_result_with_real_response():
    c = _client()
    with patch("conformal_rag.client.HybridRetriever") as MockRetriever:
        mock_ret = MagicMock()
        mock_ret.retrieve.return_value = ["Python is a programming language."]
        MockRetriever.return_value = mock_ret
        c.index(["Python is a programming language."])
        res = c.check("What is Python?", response="Python is a programming language.")
        assert res.risk_score == 0.9
        assert res.abstained is False


def test_query_requires_response():
    c = _client()
    with pytest.raises(ValueError):
        c.query("What is Python?")


def test_initializes():
    rag = ConformalRAG(alpha=0.05)
    assert rag.alpha == 0.05


def test_registers_domain():
    rag = ConformalRAG()
    rag.register_domain("finance", alpha=0.01)
    assert "finance" in rag.domains
