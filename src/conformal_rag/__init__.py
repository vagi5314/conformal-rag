"""Conformal RAG — FDR-guaranteed hallucination bounds for RAG systems."""

from conformal_rag.models import (
    Document,
    CalibrationSample,
    Prediction,
    ConformalResult,
    DriftStatus,
    Label,
)

__version__ = "0.1.0"
__all__ = [
    "Document",
    "CalibrationSample",
    "Prediction",
    "ConformalResult",
    "DriftStatus",
    "Label",
]
