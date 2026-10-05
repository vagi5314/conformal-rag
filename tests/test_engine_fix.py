"""Tests for the engine's conformal lower bound (confidence interval).

The CI is a one-sided split-conformal bound computed at calibration time:
with k = ceil((n+1)*alpha), lower = k-th smallest calibration score, and
exchangeability gives P(new score >= lower) >= 1 - alpha.
"""

import numpy as np
from unittest.mock import patch
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import ScoringBreakdown
from conformal_rag.scoring import ScoringPipeline


def _breakdown(score: float) -> ScoringBreakdown:
    return ScoringBreakdown(
        nli_score=score, final_score=score, nli_latency_ms=10.0, used_llm_judge=False
    )


class TestEngineCIFix:
    def setup_method(self):
        self.engine = ConformalRAGEngine(alpha=0.05)

    @patch.object(ScoringPipeline, "score")
    def test_confidence_interval_is_alpha_quantile(self, mock_scoring):
        # Calibration scores 0.1..0.95 with mixed labels; the CI lower bound
        # must sit near the alpha quantile (a LOWER bound, not the 95th pct).
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        mock_scoring.side_effect = [_breakdown(s) for s in scores]
        self.engine.calibrate(["q"] * 10, ["c"] * 10, ["r"] * 10, labels)

        with patch.object(ScoringPipeline, "score", return_value=_breakdown(0.9)):
            result = self.engine.predict("test", "context", "response")
        ci = result.confidence_interval
        assert ci[0] >= 0.0
        assert ci[1] == 1.0
        # k = ceil(11 * 0.05) = 1 -> lower = smallest calibration score
        assert abs(ci[0] - 0.1) < 1e-9

    @patch.object(ScoringPipeline, "score")
    def test_confidence_interval_empty_calibration(self, mock_scoring):
        mock_scoring.return_value = _breakdown(0.9)
        result = self.engine.predict("test", "context", "response")
        assert result.confidence_interval == (0.0, 1.0)
        assert result.calibrated is False

    @patch.object(ScoringPipeline, "score")
    def test_confidence_interval_covers_most_new_scores(self, mock_scoring):
        # On 1000 uniform scores the alpha-quantile lower bound should be low
        # enough that ~95% of calibration scores sit above it.
        rng = np.random.default_rng(42)
        scores = rng.uniform(0.1, 0.9, 1000).tolist()
        labels = [0] * 900 + [1] * 100
        mock_scoring.side_effect = [_breakdown(s) for s in scores]
        self.engine.calibrate(["q"] * 1000, ["c"] * 1000, ["r"] * 1000, labels)

        lower = self.engine._confidence_interval[0]
        covered = sum(1 for s in scores if s >= lower) / len(scores)
        assert covered >= 0.94
