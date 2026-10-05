"""Tests for Phase 2 production hardening.

Covers API-key auth, per-IP rate limiting, CORS, calibration/index
persistence, and the retrieval fixes (e5 instruction prefixes +
precomputed embeddings). All tests run without loading real models.
"""

from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch

from conformal_rag.api import RateLimiter, create_app
from conformal_rag.config import Settings
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.domain import DomainManager
from conformal_rag.models import Document, ScoringBreakdown
from conformal_rag.retrieval import HybridRetriever
from conformal_rag.scoring import ScoringPipeline


def _breakdown(score: float = 0.9) -> ScoringBreakdown:
    return ScoringBreakdown(
        nli_score=score, final_score=score, nli_latency_ms=10.0, used_llm_judge=False
    )


async def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# --- API-key auth ----------------------------------------------------------


class TestAPIKeyAuth:
    @pytest.fixture
    async def client(self):
        app = create_app(Settings(api_key="secret"))
        async with await _client(app) as c:
            yield c

    @pytest.mark.asyncio
    async def test_health_exempt(self, client):
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_key_rejected(self, client):
        response = await client.get("/config")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_key_rejected(self, client):
        response = await client.get("/config", headers={"X-API-Key": "nope"})
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_correct_key_allowed(self, client):
        response = await client.get("/config", headers={"X-API-Key": "secret"})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_disabled_by_default(self):
        app = create_app(Settings())  # api_key empty -> no auth
        async with await _client(app) as c:
            response = await c.get("/config")
            assert response.status_code == 200


# --- rate limiting ---------------------------------------------------------


class TestRateLimiter:
    def test_unlimited_when_zero(self):
        limiter = RateLimiter(0)
        for i in range(100):
            assert limiter.allow("ip", now=float(i)) is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(2)
        assert limiter.allow("ip", now=0.0) is True
        assert limiter.allow("ip", now=1.0) is True
        assert limiter.allow("ip", now=2.0) is False

    def test_window_slides(self):
        limiter = RateLimiter(2)
        limiter.allow("ip", now=0.0)
        limiter.allow("ip", now=1.0)
        assert limiter.allow("ip", now=2.0) is False
        # 61s later the first two hits have aged out of the 60s window.
        assert limiter.allow("ip", now=61.0) is True

    def test_per_key_isolation(self):
        limiter = RateLimiter(1)
        assert limiter.allow("a", now=0.0) is True
        assert limiter.allow("b", now=0.0) is True
        assert limiter.allow("a", now=1.0) is False


class TestRateLimitAPI:
    @pytest.mark.asyncio
    async def test_returns_429_over_limit(self):
        app = create_app(Settings(rate_limit_per_minute=2))
        async with await _client(app) as c:
            r1 = await c.get("/config")
            r2 = await c.get("/config")
            r3 = await c.get("/config")
            assert r1.status_code == 200
            assert r2.status_code == 200
            assert r3.status_code == 429


# --- CORS ------------------------------------------------------------------


class TestCORS:
    @pytest.mark.asyncio
    async def test_preflight_allowed_origin(self):
        app = create_app(Settings(cors_origins=["https://example.com"]))
        async with await _client(app) as c:
            response = await c.options(
                "/config",
                headers={
                    "Origin": "https://example.com",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert response.headers.get("access-control-allow-origin") == "https://example.com"

    @pytest.mark.asyncio
    async def test_no_cors_headers_when_disabled(self):
        app = create_app(Settings())  # no origins configured
        async with await _client(app) as c:
            response = await c.get("/config", headers={"Origin": "https://example.com"})
            assert "access-control-allow-origin" not in response.headers


# --- persistence -----------------------------------------------------------


class TestEnginePersistence:
    def _calibrate(self, engine):
        scores = [0.2, 0.4, 0.6, 0.8, 0.9]
        labels = [0, 0, 0, 0, 1]
        with patch.object(
            ScoringPipeline, "score", side_effect=[_breakdown(s) for s in scores]
        ):
            engine.calibrate(["q"] * 5, ["c"] * 5, ["r"] * 5, labels)

    def test_save_load_round_trip(self, tmp_path):
        engine = ConformalRAGEngine(alpha=0.05)
        self._calibrate(engine)
        assert engine.calibrated is True
        threshold = engine._threshold

        path = tmp_path / "engine_state.json"
        engine.save_state(path)

        fresh = ConformalRAGEngine(alpha=0.05)
        assert fresh.calibrated is False
        assert fresh.load_state(path) is True
        assert fresh.calibrated is True
        assert fresh._threshold == threshold
        assert fresh._calibration_scores == engine._calibration_scores

    def test_load_missing_file_returns_false(self, tmp_path):
        engine = ConformalRAGEngine(alpha=0.05)
        assert engine.load_state(tmp_path / "nope.json") is False
        assert engine.calibrated is False


class TestDomainPersistence:
    def test_save_load_round_trip(self, tmp_path):
        dm = DomainManager()
        scores = [0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 0, 1]
        with patch.object(
            ScoringPipeline, "score", side_effect=[_breakdown(s) for s in scores]
        ):
            dm.calibrate("finance", ["q"] * 4, ["c"] * 4, ["r"] * 4, labels)

        dm.save_state(tmp_path)

        fresh = DomainManager()
        loaded = fresh.load_state(tmp_path)
        assert "finance" in loaded
        assert fresh.domains["finance"].calibrated is True


class TestAPIPersistence:
    CALIBRATION_DATA = [
        {"query": "q1", "context": "c1", "response": "r1", "label": 0},
        {"query": "q2", "context": "c2", "response": "r2", "label": 0},
        {"query": "q3", "context": "c3", "response": "r3", "label": 1},
    ]

    @pytest.mark.asyncio
    async def test_calibration_survives_restart(self, tmp_path):
        state_dir = str(tmp_path)
        app1 = create_app(Settings(state_dir=state_dir))
        async with await _client(app1) as c1:
            with patch.object(app1.state.engine.scoring, "score", return_value=_breakdown(0.8)):
                await c1.post("/calibrate", json={"data": self.CALIBRATION_DATA})
            assert (await c1.get("/config")).json()["calibrated"] is True

        # A brand-new app pointed at the same state dir restores calibration.
        app2 = create_app(Settings(state_dir=state_dir))
        async with await _client(app2) as c2:
            assert (await c2.get("/config")).json()["calibrated"] is True


# --- retrieval fixes -------------------------------------------------------


class _FakeEmbedder:
    """Records encode() inputs so tests can assert on prefixes."""

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        if isinstance(texts, str):
            return np.ones(4, dtype=np.float32)
        return np.ones((len(texts), 4), dtype=np.float32)


class TestRetrievalPrefixes:
    def test_passage_prefix_at_index_time(self):
        retriever = HybridRetriever()
        fake = _FakeEmbedder()
        retriever._embedder = fake
        retriever.index([Document(text="Paris is the capital of France.")])
        encoded = fake.calls[0]
        assert encoded == ["passage: Paris is the capital of France."]

    def test_query_prefix_at_search_time(self):
        retriever = HybridRetriever()
        fake = _FakeEmbedder()
        retriever._embedder = fake
        retriever._encode_query("capital of France")
        assert fake.calls[0] == "query: capital of France"

    def test_embeddings_precomputed_at_index_time(self):
        retriever = HybridRetriever()
        retriever._embedder = _FakeEmbedder()
        assert retriever._doc_embeddings is None
        retriever.index([Document(text="a"), Document(text="b")])
        assert retriever._doc_embeddings is not None
        assert retriever._doc_embeddings.shape == (2, 4)


class TestRetrievalPersistence:
    def test_save_load_round_trip(self, tmp_path):
        retriever = HybridRetriever()
        retriever._embedder = _FakeEmbedder()
        retriever.index([
            Document(text="Paris is the capital of France."),
            Document(text="Python is a programming language."),
        ])
        path = tmp_path / "index_state.json"
        retriever.save_state(path)

        fresh = HybridRetriever()
        assert fresh.load_state(path) is True
        assert fresh._documents == retriever._documents
        assert fresh._bm25 is not None
        assert fresh._doc_embeddings is not None
        # Sparse search works off the restored BM25 index without a model.
        results = fresh._in_memory_sparse_search("capital of France")
        assert any("Paris" in r for r in results)

    def test_load_missing_file_returns_false(self, tmp_path):
        retriever = HybridRetriever()
        assert retriever.load_state(tmp_path / "nope.json") is False
