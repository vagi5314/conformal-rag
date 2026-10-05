"""Tests for the Redis-backed rate limiter.

Live tests require a reachable Redis and are skipped unless one responds at
CONFORMAL_RAG_REDIS_TEST_URL (default redis://localhost:6379/0). The
fail-open test runs always against a dead port.
"""

import os
import socket

import pytest
from httpx import AsyncClient, ASGITransport

from conformal_rag.api import RedisRateLimiter, create_app
from conformal_rag.config import Settings

_REDIS_URL = os.environ.get("CONFORMAL_RAG_REDIS_TEST_URL", "redis://localhost:6379/0")


def _redis_reachable(url: str) -> bool:
    # Parse host/port from redis://host:port/db
    try:
        rest = url.split("://", 1)[1]
        hostport = rest.split("/", 1)[0]
        host, _, port = hostport.partition(":")
        port = int(port or 6379)
        with socket.create_connection((host, port), timeout=1):
            return True
    except (OSError, ValueError):
        return False


_REDIS_UP = _redis_reachable(_REDIS_URL)
needs_redis = pytest.mark.skipif(not _REDIS_UP, reason="Redis not reachable")


class TestRedisRateLimiterFailOpen:
    def test_fails_open_on_dead_port(self):
        limiter = RedisRateLimiter(1, "redis://localhost:1/0")
        # Dead port -> connection error -> allow() returns True, no exception.
        assert limiter.allow("client-a") is True
        assert limiter.allow("client-a") is True

    def test_unlimited_when_zero(self):
        limiter = RedisRateLimiter(0, "redis://localhost:1/0")
        for i in range(50):
            assert limiter.allow("client-a", now=float(i)) is True


@needs_redis
class TestRedisRateLimiterLive:
    def test_blocks_over_limit(self):
        limiter = RedisRateLimiter(2, _REDIS_URL, prefix="conformal_rag:test:")
        key = "block-test"
        assert limiter.allow(key, now=0.0) is True
        assert limiter.allow(key, now=1.0) is True
        assert limiter.allow(key, now=2.0) is False
        limiter._get_client().delete("conformal_rag:test:" + key)

    def test_window_slides(self):
        limiter = RedisRateLimiter(2, _REDIS_URL, prefix="conformal_rag:test:")
        key = "slide-test"
        assert limiter.allow(key, now=0.0) is True
        assert limiter.allow(key, now=1.0) is True
        assert limiter.allow(key, now=2.0) is False
        # After the 60s window, the earliest hits age out.
        assert limiter.allow(key, now=61.0) is True
        limiter._get_client().delete("conformal_rag:test:" + key)

    def test_per_key_isolation(self):
        limiter = RedisRateLimiter(1, _REDIS_URL, prefix="conformal_rag:test:")
        assert limiter.allow("iso-a", now=0.0) is True
        assert limiter.allow("iso-b", now=0.0) is True
        assert limiter.allow("iso-a", now=1.0) is False
        client = limiter._get_client()
        client.delete("conformal_rag:test:iso-a")
        client.delete("conformal_rag:test:iso-b")


@needs_redis
class TestRedisRateLimitAPI:
    @pytest.mark.asyncio
    async def test_returns_429_over_limit(self):
        import uuid
        settings = Settings(rate_limit_per_minute=2, redis_url=_REDIS_URL)
        app = create_app(settings)
        # Unique client IP per run so leftover keys from prior runs don't
        # pre-fill the window.
        unique_ip = f"10.0.{uuid.uuid4().int % 256}.{uuid.uuid4().int % 256}"
        headers = {"x-forwarded-for": unique_ip}
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.get("/config", headers=headers)
            r2 = await client.get("/config", headers=headers)
            r3 = await client.get("/config", headers=headers)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 429
