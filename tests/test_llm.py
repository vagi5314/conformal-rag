"""Tests for LLM multi-provider routing."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from conformal_rag.llm import LLMClient, LLMConfig


class TestLLMConfig:
    def test_default_config(self):
        config = LLMConfig()
        assert config.temperature == 0.0
        assert config.max_tokens == 10
        assert config.timeout == 30.0

    def test_custom_config(self):
        config = LLMConfig(temperature=0.5, max_tokens=100)
        assert config.temperature == 0.5
        assert config.max_tokens == 100


class TestLLMClient:
    def test_initializes_with_providers(self):
        client = LLMClient()
        assert len(client.providers) > 0

    def test_has_fallback_chain(self):
        client = LLMClient()
        assert client.fallback_chain is not None
