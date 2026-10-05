"""Tests for LLM multi-provider routing."""

from unittest.mock import patch
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

    def test_fallback_on_provider_failure(self):
        client = LLMClient()
        call_count = 0

        def mock_call(provider, prompt, temperature=0.0, max_tokens=10):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Provider 1 failed")
            return "success"

        with patch.object(client, '_call_provider', side_effect=mock_call):
            result = client.complete("test prompt")
            assert result == "success"
            assert call_count == 2
