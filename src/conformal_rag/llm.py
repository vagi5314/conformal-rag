from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import litellm


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
    usage_callback: Optional[Callable[[str, int, int], None]] = None


class LLMClient:
    """Multi-provider LLM client with fallback routing."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.providers = self.config.fallback_chain
        self.fallback_chain = self.config.fallback_chain
        self.usage_callback = self.config.usage_callback

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 10,
    ) -> str:
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
        response = litellm.completion(
            model=provider,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if self.usage_callback is not None:
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.usage_callback(
                    provider,
                    int(getattr(usage, "prompt_tokens", 0) or 0),
                    int(getattr(usage, "completion_tokens", 0) or 0),
                )
        return response.choices[0].message.content.strip()

    def close(self):
        pass
