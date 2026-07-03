"""Tests for FastAPI endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from conformal_rag.api import create_app


@pytest.fixture
def app():
    return create_app()


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_stats(self, app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200
            assert "total_predictions" in response.json()
