"""LLM multi-provider routing via LiteLLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import httpx


@dataclass
class LLMConfig:
    """LLM configuration."""

    temperature: float = 0.0
    max_tokens: int = 10
    timeout: float = 30.0
    fallback_chain: List[str] = field(default_factory=lambda: [
        "groq/llama-3.3-70b-versatile",
        "openrouter/meta-llama/llama-3.1-8b-instruct:free",
        "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "gemini/gemini-2.5-flash",
    ])


class LLMClient:
    """Multi-provider LLM client with fallback routing."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.providers = self.config.fallback_chain
        self.fallback_chain = self.config.fallback_chain
        self._client = httpx.Client(timeout=self.config.timeout)

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 10,
    ) -> str:
        """Complete a prompt with automatic fallback.

        Args:
            prompt: The prompt to complete.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The completion text.

        Raises:
            RuntimeError: If all providers fail.
        """
        last_error = None
        for provider in self.providers:
            try:
                response = self._call_provider(provider, prompt, temperature, max_tokens)
                return response
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def _call_provider(
        self,
        provider: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 10,
    ) -> str:
        """Call a specific LLM provider.

        In production, this uses litellm.completion().
        For testing, we mock this method.
        """
        raise NotImplementedError("Use litellm in production")

    def close(self):
        """Close the HTTP client."""
        self._client.close()
