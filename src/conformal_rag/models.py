"""Pydantic schemas for conformal-rag."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Label(int, Enum):
    CORRECT = 0
    HALLUCINATION = 1


class Document(BaseModel):
    """Input document for indexing."""

    text: str
    metadata: Optional[dict] = None
    doc_id: Optional[str] = None
    source: Optional[str] = None
    domain: Optional[str] = None

    def model_post_init(self, __context: object) -> None:
        if self.doc_id is None:
            self.doc_id = str(uuid.uuid4())


class ChunkConfig(BaseModel):
    """Configuration for document chunking."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "intfloat/e5-base-v2"


class CalibrationSample(BaseModel):
    """Single calibration sample."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    context: str
    response: str
    label: Label
    domain: str = "default"
    conformal_score: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScoringBreakdown(BaseModel):
    """Breakdown of scoring components."""

    nli_score: float
    llm_judge_score: Optional[float] = None
    final_score: float
    nli_latency_ms: float
    llm_latency_ms: Optional[float] = None
    used_llm_judge: bool = False


class ConformalResult(BaseModel):
    """Result from conformal prediction with risk control."""

    query: str
    response: str
    risk_score: float = Field(ge=0.0)
    risk_bound: float = Field(ge=0.0)
    abstained: bool
    calibrated: bool = False
    confidence_interval: tuple[float, float]
    scoring_breakdown: ScoringBreakdown
    domain: str = "default"
    latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Prediction(BaseModel):
    """Stored prediction record."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    response: str
    risk_score: float = Field(ge=0.0)
    risk_bound: float = Field(ge=0.0)
    abstained: bool
    domain: str = "default"
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DriftStatus(BaseModel):
    """Status of distribution drift monitoring."""

    domain: str
    drifted: bool
    fdr: float
    threshold: float
    samples: int
    coverage: float
    recommendation: str


class CalibrationStats(BaseModel):
    """Statistics from a calibration run."""

    domain: str
    alpha: float
    n_samples: int
    n_hallucinations: int
    threshold: float
    estimated_fdr: float
    coverage: float
    weights: tuple[float, float] = (0.6, 0.4)
    warnings: list[str] = Field(default_factory=list)
