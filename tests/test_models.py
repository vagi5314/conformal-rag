"""Tests for Pydantic models."""

import pytest
from conformal_rag.models import (
    Document,
    CalibrationSample,
    Prediction,
    ConformalResult,
    DriftStatus,
    Label,
    ChunkConfig,
    ScoringBreakdown,
    CalibrationStats,
)


class TestDocument:
    def test_auto_generates_doc_id(self):
        doc = Document(text="test")
        assert doc.doc_id is not None
        assert len(doc.doc_id) == 36  # UUID format

    def test_preserves_provided_doc_id(self):
        doc = Document(text="test", doc_id="custom-id")
        assert doc.doc_id == "custom-id"

    def test_optional_fields_default_none(self):
        doc = Document(text="test")
        assert doc.metadata is None
        assert doc.source is None
        assert doc.domain is None


class TestChunkConfig:
    def test_defaults(self):
        config = ChunkConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.embedding_model == "intfloat/e5-base-v2"


class TestCalibrationSample:
    def test_creates_with_label(self):
        sample = CalibrationSample(
            query="What is Python?",
            context="Python is a language.",
            response="Python is a language.",
            label=Label.CORRECT,
        )
        assert sample.label == Label.CORRECT
        assert sample.domain == "default"

    def test_hallucination_label(self):
        sample = CalibrationSample(
            query="What is Python?",
            context="Python is a language.",
            response="Python is a snake.",
            label=Label.HALLUCINATION,
        )
        assert sample.label == Label.HALLUCINATION


class TestConformalResult:
    def test_creates_with_all_fields(self):
        result = ConformalResult(
            query="test",
            response="test",
            risk_score=0.02,
            risk_bound=0.05,
            abstained=False,
            confidence_interval=(0.01, 0.05),
            scoring_breakdown=ScoringBreakdown(
                nli_score=0.92,
                llm_judge_score=0.95,
                final_score=0.94,
                nli_latency_ms=100.0,
                llm_latency_ms=200.0,
                used_llm_judge=True,
            ),
        )
        assert result.abstained is False
        assert result.risk_score == 0.02


class TestLabel:
    def test_correct_is_zero(self):
        assert Label.CORRECT == 0

    def test_hallucination_is_one(self):
        assert Label.HALLUCINATION == 1


class TestPrediction:
    def test_creates_with_defaults(self):
        pred = Prediction(
            query="test",
            response="test",
            risk_score=0.1,
            risk_bound=0.05,
            abstained=False,
        )
        assert pred.domain == "default"
        assert pred.latency_ms == 0.0
        assert pred.id is not None


class TestCalibrationStats:
    def test_creates_with_defaults(self):
        stats = CalibrationStats(
            domain="finance",
            alpha=0.05,
            n_samples=100,
            n_hallucinations=5,
            threshold=0.45,
            estimated_fdr=0.008,
            coverage=0.85,
        )
        assert stats.weights == (0.6, 0.4)

    def test_custom_weights(self):
        stats = CalibrationStats(
            domain="finance",
            alpha=0.05,
            n_samples=100,
            n_hallucinations=5,
            threshold=0.45,
            estimated_fdr=0.008,
            coverage=0.85,
            weights=(0.7, 0.3),
        )
        assert stats.weights == (0.7, 0.3)


class TestDriftStatus:
    def test_creates_with_recommendation(self):
        status = DriftStatus(
            domain="finance",
            drifted=False,
            approved_wrong_rate=0.008,
            threshold=0.42,
            samples=1200,
            coverage=0.85,
            recommendation="No drift detected",
        )
        assert status.drifted is False


class TestModelRejection:
    def test_document_requires_text(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Document()

    def test_calibration_sample_rejects_invalid_label(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CalibrationSample(
                query="q", context="c", response="r", label="invalid"
            )

    def test_conformal_result_rejects_negative_risk(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ConformalResult(
                query="q", response="r", risk_score=-0.1,
                risk_bound=0.05, abstained=False,
                confidence_interval=(0.0, 0.05),
                scoring_breakdown=ScoringBreakdown(
                    nli_score=0.9, final_score=0.9, nli_latency_ms=100.0
                )
            )
