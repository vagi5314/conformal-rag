"""Conformal RAG — FDR-guaranteed hallucination bounds for RAG systems."""

from conformal_rag.models import (
    Document,
    CalibrationSample,
    Prediction,
    ConformalResult,
    DriftStatus,
    CalibrationStats,
    ScoringBreakdown,
    ChunkConfig,
    Label,
)
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.domain import DomainManager
from conformal_rag.client import ConformalRAG
from conformal_rag.monitoring import FDRMonitor
from conformal_rag.guards import clamp_alpha, sanitize_query

__version__ = "0.1.0"
__all__ = [
    "Document",
    "CalibrationSample",
    "Prediction",
    "ConformalResult",
    "DriftStatus",
    "CalibrationStats",
    "ScoringBreakdown",
    "ChunkConfig",
    "Label",
    "ConformalRAGEngine",
    "DomainManager",
    "ConformalRAG",
    "FDRMonitor",
    "clamp_alpha",
    "sanitize_query",
]
