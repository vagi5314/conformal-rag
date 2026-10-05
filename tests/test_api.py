"""Tests for FastAPI endpoints."""

from unittest.mock import patch

import pytest
from httpx import AsyncClient, ASGITransport

from conformal_rag.api import create_app
from conformal_rag.models import ScoringBreakdown


def _breakdown(score: float = 0.9) -> ScoringBreakdown:
    return ScoringBreakdown(
        nli_score=score, final_score=score, nli_latency_ms=10.0, used_llm_judge=False
    )


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_stats(self, client):
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "total_predictions" in response.json()


class TestConfigEndpoints:
    @pytest.mark.asyncio
    async def test_get_config_defaults(self, client):
        response = await client.get("/config")
        assert response.status_code == 200
        body = response.json()
        assert body["alpha"] == 0.05
        assert body["online"] is True
        assert body["calibrated"] is False
        assert body["documents"] == 0

    @pytest.mark.asyncio
    async def test_set_config_rebuilds_engine(self, app, client):
        old_engine = app.state.engine
        response = await client.post("/config", json={"alpha": 0.01, "online": False})
        assert response.status_code == 200
        body = response.json()
        assert body["alpha"] == 0.01
        assert body["online"] is False
        assert app.state.engine is not old_engine
        assert app.state.engine.online is None

    @pytest.mark.asyncio
    async def test_set_config_clamps_alpha(self, client):
        response = await client.post("/config", json={"alpha": 0.9, "online": True})
        assert response.json()["alpha"] == 0.2

    @pytest.mark.asyncio
    async def test_reset(self, app, client):
        await client.post("/config", json={"alpha": 0.01, "online": False})
        response = await client.post("/reset")
        assert response.json()["status"] == "reset"
        assert app.state.engine.alpha == 0.05
        assert app.state.engine.online is not None


class TestRetrievalEndpoints:
    @pytest.mark.asyncio
    async def test_samples_returns_builtin_docs(self, client):
        response = await client.get("/samples")
        assert response.status_code == 200
        docs = response.json()["documents"]
        assert len(docs) == 8
        assert any("Paris" in d for d in docs)

    @pytest.mark.asyncio
    async def test_index_and_documents(self, client):
        response = await client.post(
            "/index", json={"documents": ["Paris is the capital of France.", "  ", "Python rocks."]}
        )
        assert response.json()["indexed"] == 2
        response = await client.get("/documents")
        body = response.json()
        assert body["count"] == 2
        assert "Paris is the capital of France." in body["documents"]

    @pytest.mark.asyncio
    async def test_retrieve_round_trip(self, app, client):
        await client.post(
            "/index",
            json={"documents": [
                "Paris is the capital of France.",
                "Python is a programming language.",
            ]},
        )
        with patch.object(
            app.state.retriever,
            "_in_memory_dense_search",
            return_value=["Paris is the capital of France."],
        ):
            response = await client.post(
                "/retrieve", json={"query": "capital of France", "final_k": 2}
            )
            assert response.status_code == 200
            results = response.json()["results"]
            assert "Paris is the capital of France." in results
            assert len(results) <= 2


class TestPredictEndpoint:
    @pytest.mark.asyncio
    async def test_predict_returns_conformal_result(self, app, client):
        with patch.object(app.state.engine.scoring, "score", return_value=_breakdown(0.9)):
            response = await client.post(
                "/predict",
                json={"query": "q", "context": ["c1", "c2"], "response": "r"},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["risk_score"] == 0.9
            assert "abstained" in body
            assert "scoring_breakdown" in body

    @pytest.mark.asyncio
    async def test_predict_records_to_engine_monitor(self, app, client):
        with patch.object(app.state.engine.scoring, "score", return_value=_breakdown(0.9)):
            await client.post(
                "/predict", json={"query": "q", "context": ["c"], "response": "r"}
            )
        assert app.state.engine.fdr_monitor.get_metrics()["total_predictions"] == 1

    @pytest.mark.asyncio
    async def test_predict_routes_to_calibrated_domain(self, app, client):
        await client.post("/domains", json={"name": "finance", "alpha": 0.01})
        data = [
            {"query": "q1", "context": "c1", "response": "r1", "label": 0},
            {"query": "q2", "context": "c2", "response": "r2", "label": 1},
        ]
        dm = app.state.domain_manager
        engine = dm.domains["finance"]
        with patch.object(engine.scoring, "score", return_value=_breakdown(0.8)):
            await client.post("/domains/finance/calibrate", json={"data": data})

        # Patch routing so the test does not load the embedding model.
        with patch.object(dm, "_detect_domain", return_value="finance"):
            with patch.object(engine.scoring, "score", return_value=_breakdown(0.9)):
                response = await client.post(
                    "/predict", json={"query": "q", "context": ["c"], "response": "r"}
                )
        assert response.status_code == 200
        assert response.json()["domain"] == "finance"
        assert dm.monitors["finance"].get_metrics()["total_predictions"] == 1
        # Main engine untouched by the routed prediction.
        assert app.state.engine.fdr_monitor.get_metrics()["total_predictions"] == 0

    @pytest.mark.asyncio
    async def test_predict_explicit_registered_domain_uses_domain_manager(self, app, client):
        await client.post("/domains", json={"name": "legal", "alpha": 0.02})
        dm = app.state.domain_manager
        engine = dm.domains["legal"]
        with patch.object(engine.scoring, "score", return_value=_breakdown(0.9)):
            response = await client.post(
                "/predict",
                json={"query": "q", "context": ["c"], "response": "r", "domain": "legal"},
            )
        assert response.json()["domain"] == "legal"
        assert dm.monitors["legal"].get_metrics()["total_predictions"] == 1

    @pytest.mark.asyncio
    async def test_predict_unregistered_domain_uses_main_engine(self, app, client):
        with patch.object(app.state.engine.scoring, "score", return_value=_breakdown(0.9)):
            response = await client.post(
                "/predict",
                json={"query": "q", "context": ["c"], "response": "r", "domain": "unknown"},
            )
        assert response.json()["domain"] == "unknown"
        assert app.state.engine.fdr_monitor.get_metrics()["total_predictions"] == 1
        assert "unknown" not in app.state.domain_manager.domains


class TestCalibrateEndpoints:
    CALIBRATION_DATA = [
        {"query": "q1", "context": "c1", "response": "r1", "label": 0},
        {"query": "q2", "context": "c2", "response": "r2", "label": 0},
        {"query": "q3", "context": "c3", "response": "r3", "label": 1},
    ]

    @pytest.mark.asyncio
    async def test_calibrate_typed_samples(self, app, client):
        with patch.object(app.state.engine.scoring, "score", return_value=_breakdown(0.8)):
            response = await client.post(
                "/calibrate", json={"data": self.CALIBRATION_DATA, "domain": "default"}
            )
            assert response.status_code == 200
            body = response.json()
            assert body["n_samples"] == 3
            assert body["n_hallucinations"] == 1
            assert app.state.last_calibration is not None

    @pytest.mark.asyncio
    async def test_calibrate_rejects_invalid_label(self, client):
        bad = [{"query": "q", "context": "c", "response": "r", "label": 5}]
        response = await client.post("/calibrate", json={"data": bad})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_calibration_status(self, app, client):
        response = await client.get("/calibration")
        assert response.json() == {"calibrated": False}
        with patch.object(app.state.engine.scoring, "score", return_value=_breakdown(0.8)):
            await client.post("/calibrate", json={"data": self.CALIBRATION_DATA})
        response = await client.get("/calibration")
        body = response.json()
        assert body["calibrated"] is True
        assert body["n_samples"] == 3


class TestRecordEndpoint:
    @pytest.mark.asyncio
    async def test_record_updates_monitor_and_online(self, app, client):
        with patch.object(app.state.engine.scoring, "score", return_value=_breakdown(0.9)):
            response = await client.post(
                "/record",
                json={
                    "query": "q",
                    "context": "c",
                    "response": "r",
                    "actual_hallucination": False,
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "recorded"
            assert body["metrics"]["total_predictions"] == 1
            assert body["online"]["enabled"] is True
            assert body["online"]["n_updates"] == 1

    @pytest.mark.asyncio
    async def test_record_without_online(self, app, client):
        await client.post("/config", json={"alpha": 0.05, "online": False})
        with patch.object(app.state.engine.scoring, "score", return_value=_breakdown(0.9)):
            response = await client.post(
                "/record",
                json={
                    "query": "q",
                    "context": "c",
                    "response": "r",
                    "actual_hallucination": True,
                },
            )
            assert response.json()["online"] == {"enabled": False}


class TestOnlineEndpoint:
    @pytest.mark.asyncio
    async def test_online_enabled_by_default(self, client):
        response = await client.get("/online")
        body = response.json()
        assert body["enabled"] is True
        assert "threshold" in body
        assert "level" in body

    @pytest.mark.asyncio
    async def test_online_disabled(self, client):
        await client.post("/config", json={"alpha": 0.05, "online": False})
        assert (await client.get("/online")).json() == {"enabled": False}


class TestDomainEndpoints:
    @pytest.mark.asyncio
    async def test_register_and_list_domains(self, client):
        response = await client.post("/domains", json={"name": "finance", "alpha": 0.01})
        assert response.json()["status"] == "registered"
        body = (await client.get("/domains")).json()
        names = [d["name"] for d in body["domains"]]
        assert "finance" in names
        finance = next(d for d in body["domains"] if d["name"] == "finance")
        assert finance["alpha"] == 0.01
        assert finance["calibrated"] is False

    @pytest.mark.asyncio
    async def test_calibrate_domain(self, app, client):
        await client.post("/domains", json={"name": "medical", "alpha": 0.001})
        data = [
            {"query": "q1", "context": "c1", "response": "r1", "label": 0},
            {"query": "q2", "context": "c2", "response": "r2", "label": 1},
        ]
        engine = app.state.domain_manager.domains["medical"]
        with patch.object(engine.scoring, "score", return_value=_breakdown(0.7)):
            response = await client.post("/domains/medical/calibrate", json={"data": data})
            assert response.status_code == 200
            assert response.json()["n_samples"] == 2

    @pytest.mark.asyncio
    async def test_domain_drift(self, client):
        await client.post("/domains", json={"name": "legal", "alpha": 0.02})
        response = await client.get("/domains/legal/drift")
        body = response.json()
        assert body["domain"] == "legal"
        assert body["drifted"] is False

    @pytest.mark.asyncio
    async def test_feedback_records_to_domain_monitor(self, app, client):
        await client.post("/domains", json={"name": "general", "alpha": 0.05})
        response = await client.post(
            "/feedback",
            json={"domain": "general", "predicted_hallucination": True, "actual_hallucination": True},
        )
        assert response.json()["status"] == "recorded"
        monitor = app.state.domain_manager.monitors["general"]
        assert monitor.get_metrics()["total_predictions"] == 1
