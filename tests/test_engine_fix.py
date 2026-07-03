import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import ScoringBreakdown
from conformal_rag.scoring import ScoringPipeline


class TestEngineCIFix:
    def setup_method(self):
        self.engine = ConformalRAGEngine(alpha=0.05)

    @patch.object(ScoringPipeline, 'score')
    def test_confidence_interval_uses_quantile(self, mock_scoring):
        mock_scoring.return_value = ScoringBreakdown(
            nli_score=0.9, final_score=0.9, nli_latency_ms=10.0, used_llm_judge=False
        )
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.engine._calibration_scores = scores
        self.engine._calibration_labels = labels
        self.engine._threshold = np.percentile(scores, 95)

        result = self.engine.predict("test", "context", "response")
        ci = result.confidence_interval
        assert ci[0] >= 0.0
        assert ci[1] <= 1.0
        assert ci[0] < ci[1]

    @patch.object(ScoringPipeline, 'score')
    def test_confidence_interval_empty_calibration(self, mock_scoring):
        mock_scoring.return_value = ScoringBreakdown(
            nli_score=0.9, final_score=0.9, nli_latency_ms=10.0, used_llm_judge=False
        )
        result = self.engine.predict("test", "context", "response")
        ci = result.confidence_interval
        assert ci == (0.0, 1.0)

    @patch.object(ScoringPipeline, 'score')
    def test_confidence_interval_not_proportion_ci(self, mock_scoring):
        mock_scoring.return_value = ScoringBreakdown(
            nli_score=0.9, final_score=0.9, nli_latency_ms=10.0, used_llm_judge=False
        )
        np.random.seed(42)
        scores = np.random.uniform(0.1, 0.9, 1000).tolist()
        labels = [0] * 1000
        self.engine._calibration_scores = scores
        self.engine._calibration_labels = labels
        self.engine._threshold = np.percentile(scores, 95)

        result = self.engine.predict("test", "context", "response")
        ci = result.confidence_interval
        assert ci[1] - ci[0] < 0.5
