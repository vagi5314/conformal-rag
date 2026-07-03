# tests/test_llm_judge.py
import pytest
from unittest.mock import patch, MagicMock
from conformal_rag.llm import LLMClient, LLMConfig


class TestLLMClient:
    def setup_method(self):
        self.config = LLMConfig()

    @patch("conformal_rag.llm.litellm")
    def test_complete_returns_string(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.85"
        mock_litellm.completion.return_value = mock_response

        client = LLMClient(self.config)
        result = client.complete("Test prompt")
        assert result == "0.85"

    @patch("conformal_rag.llm.litellm")
    def test_complete_uses_temperature_0(self, mock_litellm):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.9"
        mock_litellm.completion.return_value = mock_response

        client = LLMClient(self.config)
        client.complete("Test prompt")
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    @patch("conformal_rag.llm.litellm")
    def test_fallback_on_first_provider_failure(self, mock_litellm):
        mock_litellm.completion.side_effect = [
            Exception("Groq failed"),
            MagicMock(choices=[MagicMock()]),
        ]
        mock_litellm.completion.return_value.choices[0].message.content = "0.7"

        client = LLMClient(self.config)
        result = client.complete("Test prompt")
        assert mock_litellm.completion.call_count == 2

    @patch("conformal_rag.llm.litellm")
    def test_all_providers_fail_raises(self, mock_litellm):
        mock_litellm.completion.side_effect = Exception("All failed")

        client = LLMClient(self.config)
        with pytest.raises(RuntimeError):
            client.complete("Test prompt")
