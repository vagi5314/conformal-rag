"""Tests for Conformal Risk Control engine."""

from unittest.mock import patch
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import ConformalResult, CalibrationStats, ScoringBreakdown


def _make_breakdown(final_score: float, nli_score: float = 0.5, nli_latency_ms: float = 10.0) -> ScoringBreakdown:
    return ScoringBreakdown(
        nli_score=nli_score,
        final_score=final_score,
        nli_latency_ms=nli_latency_ms,
    )


class TestConformalRAGEngine:
    def test_initializes_with_alpha(self):
        engine = ConformalRAGEngine(alpha=0.05)
        assert engine.alpha == 0.05

    def test_clamps_invalid_alpha(self):
        engine = ConformalRAGEngine(alpha=0.5)
        assert engine.alpha == 0.2

        engine = ConformalRAGEngine(alpha=0.0)
        assert engine.alpha == 0.001

    def test_calibrate_with_samples(self):
        engine = ConformalRAGEngine(alpha=0.05)
        with patch.object(engine.scoring, "score") as mock_score:
            mock_score.return_value = _make_breakdown(0.8)
            stats = engine.calibrate(
                queries=["q1", "q2", "q3"],
                contexts=["c1", "c2", "c3"],
                responses=["r1", "r2", "r3"],
                labels=[0, 0, 1],
            )
            assert isinstance(stats, CalibrationStats)
            assert stats.n_samples == 3

    def test_predict_with_threshold(self):
        engine = ConformalRAGEngine(alpha=0.05)
        engine._threshold = 0.45

        with patch.object(engine.scoring, "score") as mock_score:
            mock_score.return_value = _make_breakdown(0.8, nli_score=0.8, nli_latency_ms=100.0)
            result = engine.predict(
                query="test",
                context="test context",
                response="test response",
            )
            assert isinstance(result, ConformalResult)
            assert result.abstained is False

    def test_abstains_when_risk_too_high(self):
        engine = ConformalRAGEngine(alpha=0.05)
        engine._threshold = 0.45

        with patch.object(engine.scoring, "score") as mock_score:
            mock_score.return_value = _make_breakdown(0.2, nli_score=0.2, nli_latency_ms=100.0)
            result = engine.predict(
                query="test",
                context="test context",
                response="test response",
            )
            assert result.abstained is True

    def test_predict_without_calibration_uses_default(self):
        engine = ConformalRAGEngine(alpha=0.05)
        with patch.object(engine.scoring, "score") as mock_score:
            mock_score.return_value = _make_breakdown(0.8, nli_score=0.8, nli_latency_ms=100.0)
            result = engine.predict(
                query="test",
                context="test context",
                response="test response",
            )
            assert result.risk_bound == 0.05
