"""Configuration settings for conformal-rag.

Settings are loaded from environment variables prefixed with CONFORMAL_RAG_.
Security-sensitive features (API auth, rate limiting, CORS) are opt-in so
local development keeps working with zero configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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
    # Full Qdrant URL (e.g. http://localhost:6333). Empty = in-memory index;
    # Qdrant is opt-in because qdrant_host has a localhost default that cannot
    # serve as an "enabled" switch.
    qdrant_url: str = ""
    qdrant_collection: str = "docs"

    # Domain routing
    # Minimum cosine similarity to route a query to a calibrated domain;
    # below this the query falls back to "general".
    domain_route_margin: float = 0.55

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Security / production hardening (opt-in) ---------------------
    # When non-empty, every API route except /health requires this key in
    # the X-API-Key header. Empty string = auth disabled (local dev).
    api_key: str = ""
    # Max API requests per client IP per minute. 0 = unlimited.
    rate_limit_per_minute: int = 0
    # Comma-separated allowed origins for CORS. Empty = no CORS middleware
    # (the bundled frontend is same-origin and needs none).
    cors_origins: List[str] = field(default_factory=list)
    # Directory for persisting calibration + index state across restarts.
    # Empty string = persistence disabled (in-memory only).
    state_dir: str = ""
    # Log level for the package logger.
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Load overrides from environment variables."""
        self.postgres_url = os.environ.get("CONFORMAL_RAG_POSTGRES_URL", self.postgres_url)
        self.redis_url = os.environ.get("CONFORMAL_RAG_REDIS_URL", self.redis_url)
        self.qdrant_host = os.environ.get("CONFORMAL_RAG_QDRANT_HOST", self.qdrant_host)
        self.qdrant_port = int(os.environ.get("CONFORMAL_RAG_QDRANT_PORT", str(self.qdrant_port)))
        self.qdrant_url = os.environ.get("CONFORMAL_RAG_QDRANT_URL", self.qdrant_url)
        self.qdrant_collection = os.environ.get(
            "CONFORMAL_RAG_QDRANT_COLLECTION", self.qdrant_collection
        )
        self.domain_route_margin = float(
            os.environ.get("CONFORMAL_RAG_DOMAIN_ROUTE_MARGIN", str(self.domain_route_margin))
        )
        self.api_host = os.environ.get("CONFORMAL_RAG_API_HOST", self.api_host)
        self.api_port = int(os.environ.get("CONFORMAL_RAG_API_PORT", str(self.api_port)))
        self.default_alpha = float(os.environ.get("CONFORMAL_RAG_ALPHA", str(self.default_alpha)))

        self.api_key = os.environ.get("CONFORMAL_RAG_API_KEY", self.api_key)
        self.rate_limit_per_minute = int(
            os.environ.get("CONFORMAL_RAG_RATE_LIMIT_PER_MINUTE", str(self.rate_limit_per_minute))
        )
        cors_raw = os.environ.get("CONFORMAL_RAG_CORS_ORIGINS", "")
        if cors_raw.strip():
            self.cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
        self.state_dir = os.environ.get("CONFORMAL_RAG_STATE_DIR", self.state_dir)
        self.log_level = os.environ.get("CONFORMAL_RAG_LOG_LEVEL", self.log_level)

    @classmethod
    def from_env(cls) -> Settings:
        """Create settings from environment variables."""
        return cls()
