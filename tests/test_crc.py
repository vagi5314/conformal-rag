"""Statistical tests for the CRC construction (Bates et al. 2021).

These test the actual guarantee, not just plumbing: the threshold must be
the (c_max+1)-th largest hallucination score, and on exchangeable data the
empirical rate of (approved AND wrong) must stay at or below alpha.
"""

import numpy as np
import pytest
from unittest.mock import patch

from conformal_rag.engine import ConformalRAGEngine, _ABSTAIN_ON_EVERYTHING
from conformal_rag.models import ScoringBreakdown
from conformal_rag.scoring import ScoringPipeline


def _breakdown(score: float) -> ScoringBreakdown:
    return ScoringBreakdown(
        nli_score=score, final_score=score, nli_latency_ms=1.0, used_llm_judge=False
    )


class TestCRCThreshold:
    def test_threshold_is_ranked_hallucination_score(self):
        # n=19, alpha=0.05 -> (n+1)*alpha = 1.0, c_max = 0:
        # threshold = the LARGEST hallucination score.
        scores = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45,
                  0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.3, 0.7, 0.95]
        labels = [0] * 16 + [1, 1, 1]  # hallucinations score 0.3, 0.7, 0.95
        t = ConformalRAGEngine._crc_threshold(scores, labels, alpha=0.05)
        assert t == pytest.approx(0.95)

    def test_more_data_allows_more_approved_errors(self):
        # n=99, alpha=0.05 -> c_max = floor(100*0.05) - 1 = 4:
        # threshold = 5th largest hallucination score.
        rng = np.random.default_rng(1)
        scores = rng.uniform(0.6, 0.99, 99).tolist()
        labels = [0] * 99
        halluc_scores = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]
        for i, hs in enumerate(halluc_scores):
            scores[i] = hs
            labels[i] = 1
        t = ConformalRAGEngine._crc_threshold(scores, labels, alpha=0.05)
        # hallucination scores sorted desc: [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]
        assert t == pytest.approx(0.1)  # index c_max = 4 -> 5th largest

    def test_enough_data_approves_everything(self):
        # c_max >= number of hallucinations -> threshold 0 (approve all).
        scores = [0.9] * 98 + [0.2, 0.1]
        labels = [0] * 98 + [1, 1]
        t = ConformalRAGEngine._crc_threshold(scores, labels, alpha=0.05)
        assert t == 0.0

    def test_too_few_samples_abstains_on_everything(self):
        # n=9, alpha=0.05 -> c_max = floor(10*0.05) - 1 = -1: cannot certify.
        scores = [0.9] * 8 + [0.2]
        labels = [0] * 8 + [1]
        t = ConformalRAGEngine._crc_threshold(scores, labels, alpha=0.05)
        assert t == _ABSTAIN_ON_EVERYTHING


class TestCRCGuarantee:
    """The actual finite-sample guarantee, checked empirically."""

    @patch.object(ScoringPipeline, "score")
    def test_approved_and_wrong_rate_below_alpha(self, mock_scoring):
        rng = np.random.default_rng(7)
        alpha = 0.05
        n_cal, n_test = 500, 2000

        def make(n):
            is_halluc = rng.random(n) < 0.15
            # faithful answers score high, hallucinations low (with noise)
            scores = np.where(
                is_halluc,
                rng.beta(2, 8, n),
                rng.beta(8, 2, n),
            )
            return scores.tolist(), is_halluc.astype(int).tolist()

        cal_scores, cal_labels = make(n_cal)
        test_scores, test_labels = make(n_test)

        mock_scoring.side_effect = [_breakdown(s) for s in cal_scores]
        engine = ConformalRAGEngine(alpha=alpha)
        engine.calibrate(["q"] * n_cal, ["c"] * n_cal, ["r"] * n_cal, cal_labels)

        approved_and_wrong = 0
        for s, lbl in zip(test_scores, test_labels):
            approved = s > engine._threshold
            if approved and lbl == 1:
                approved_and_wrong += 1

        rate = approved_and_wrong / n_test
        # The CRC guarantee: E[rate] <= alpha. Allow sampling slack.
        assert rate <= alpha + 0.02, f"guarantee violated: {rate:.4f} > {alpha}"

    @patch.object(ScoringPipeline, "score")
    def test_tie_abstains(self, mock_scoring):
        # Approving strictly above the threshold is what makes the guarantee
        # hold when calibration scores tie with test scores.
        scores = [0.9] * 30 + [0.4, 0.4, 0.4]
        labels = [0] * 30 + [1, 1, 1]
        mock_scoring.side_effect = [_breakdown(s) for s in scores]
        engine = ConformalRAGEngine(alpha=0.05)
        engine.calibrate(["q"] * 33, ["c"] * 33, ["r"] * 33, labels)

        with patch.object(ScoringPipeline, "score", return_value=_breakdown(engine._threshold)):
            result = engine.predict("q", "c", "r")
        assert result.abstained is True

    @patch.object(ScoringPipeline, "score")
    def test_calibrate_rejects_degenerate_labels(self, mock_scoring):
        engine = ConformalRAGEngine(alpha=0.05)
        with pytest.raises(ValueError, match="same"):
            engine.calibrate(["q"] * 5, ["c"] * 5, ["r"] * 5, [0, 0, 0, 0, 0])

    @patch.object(ScoringPipeline, "score")
    def test_small_calibration_warns_and_abstains(self, mock_scoring):
        scores = [0.9] * 9 + [0.2]
        labels = [0] * 9 + [1]
        mock_scoring.side_effect = [_breakdown(s) for s in scores]
        engine = ConformalRAGEngine(alpha=0.05)
        stats = engine.calibrate(["q"] * 10, ["c"] * 10, ["r"] * 10, labels)
        assert any("too few" in w for w in stats.warnings)
        with patch.object(ScoringPipeline, "score", return_value=_breakdown(0.99)):
            result = engine.predict("q", "c", "r")
        assert result.abstained is True

    @patch.object(ScoringPipeline, "score")
    def test_record_uses_same_approval_rule_as_predict(self, mock_scoring):
        scores = [0.9] * 30 + [0.4, 0.4, 0.4]
        labels = [0] * 30 + [1, 1, 1]
        mock_scoring.side_effect = [_breakdown(s) for s in scores]
        engine = ConformalRAGEngine(alpha=0.05)
        engine.calibrate(["q"] * 33, ["c"] * 33, ["r"] * 33, labels)

        # A faithful answer (score above threshold) recorded as correct must
        # NOT be logged as a predicted hallucination.
        with patch.object(ScoringPipeline, "score", return_value=_breakdown(0.95)):
            engine.record("q", "c", "r", actual_hallucination=False)
        m = engine.fdr_monitor.get_metrics()
        assert m["abstention_rate"] == 0.0

        # A low-score answer must be logged as a predicted hallucination.
        with patch.object(ScoringPipeline, "score", return_value=_breakdown(0.1)):
            engine.record("q", "c", "r", actual_hallucination=True)
        m = engine.fdr_monitor.get_metrics()
        assert m["abstention_rate"] == 0.5
