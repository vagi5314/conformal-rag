"""Tests for configuration settings."""

from conformal_rag.config import Settings


class TestSettings:
    def test_defaults(self):
        settings = Settings()
        assert settings.default_alpha == 0.05
        assert settings.alpha_min == 0.001
        assert settings.alpha_max == 0.2

    def test_loads_from_env(self, monkeypatch):
        monkeypatch.setenv("CONFORMAL_RAG_ALPHA", "0.10")
        settings = Settings()
        assert settings.default_alpha == 0.10

    def test_handles_missing_env_gracefully(self):
        settings = Settings()
        assert settings.default_alpha == 0.05
