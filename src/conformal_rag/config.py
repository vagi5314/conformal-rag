"""Configuration settings for conformal-rag."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Alpha bounds
    alpha_min: float = 0.001
    alpha_max: float = 0.2
    default_alpha: float = 0.05

    # Calibration
    min_calibration_samples: int = 100
    recommended_calibration_samples: int = 500

    # Scoring
    nli_confidence_threshold: float = 0.7
    scoring_weights: tuple[float, float] = (0.6, 0.4)

    # LLM
    llm_temperature: float = 0.0
    llm_max_tokens: int = 10
    llm_timeout: float = 30.0
    llm_fallback_chain: List[str] = field(default_factory=lambda: [
        "groq/llama-3.3-70b-versatile",
        "openrouter/meta-llama/llama-3.1-8b-instruct:free",
        "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "gemini/gemini-2.5-flash",
    ])

    # Retrieval
    embedding_model: str = "intfloat/e5-base-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5

    # Online conformal
    online_window_size: int = 500
    online_eta: float = 0.01

    # Database
    postgres_url: str = ""
    redis_url: str = ""

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    def __post_init__(self) -> None:
        """Load overrides from environment variables."""
        self.postgres_url = os.environ.get("CONFORMAL_RAG_POSTGRES_URL", self.postgres_url)
        self.redis_url = os.environ.get("CONFORMAL_RAG_REDIS_URL", self.redis_url)
        self.qdrant_host = os.environ.get("CONFORMAL_RAG_QDRANT_HOST", self.qdrant_host)
        self.qdrant_port = int(os.environ.get("CONFORMAL_RAG_QDRANT_PORT", str(self.qdrant_port)))
        self.api_host = os.environ.get("CONFORMAL_RAG_API_HOST", self.api_host)
        self.api_port = int(os.environ.get("CONFORMAL_RAG_API_PORT", str(self.api_port)))
        self.default_alpha = float(os.environ.get("CONFORMAL_RAG_ALPHA", str(self.default_alpha)))

    @classmethod
    def from_env(cls) -> Settings:
        """Create settings from environment variables."""
        return cls()
