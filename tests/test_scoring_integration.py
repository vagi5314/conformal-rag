"""Integration tests for ScoringPipeline with NLI scorer and LLM judge."""

import pytest
from unittest.mock import patch, MagicMock
from conformal_rag.scoring import ScoringPipeline, NLIScorer
from conformal_rag.models import ScoringBreakdown


class TestScoringPipelineIntegration:
    def setup_method(self):
        self.pipeline = ScoringPipeline(use_llm_judge=False)

    def test_pipeline_returns_scoring_breakdown(self):
        with patch.object(self.pipeline.nli, "score", return_value=0.85):
            result = self.pipeline.score("query", "context", "response")
            assert isinstance(result, ScoringBreakdown)
            assert result.nli_score == 0.85
            assert result.final_score == 0.85
            assert result.used_llm_judge is False

    def test_pipeline_with_low_confidence_triggers_llm(self):
        pipeline = ScoringPipeline(use_llm_judge=True, nli_confidence_threshold=0.3)
        with patch.object(pipeline.nli, "score", return_value=0.5):
            with patch.object(pipeline.llm_judge, "score", return_value=0.9):
                result = pipeline.score("query", "context", "response")
                assert result.used_llm_judge is True
                assert result.llm_judge_score == 0.9

    def test_pipeline_nli_only_path_skips_llm(self):
        pipeline = ScoringPipeline(use_llm_judge=True, nli_confidence_threshold=0.1)
        with patch.object(pipeline.nli, "score", return_value=0.99):
            result = pipeline.score("query", "context", "response")
            assert result.used_llm_judge is False
            assert result.nli_score == 0.99
            assert result.final_score == 0.99

    def test_pipeline_weighted_ensemble(self):
        pipeline = ScoringPipeline(
            use_llm_judge=True, nli_confidence_threshold=0.3, weights=(0.6, 0.4)
        )
        with patch.object(pipeline.nli, "score", return_value=0.5):
            with patch.object(pipeline.llm_judge, "score", return_value=1.0):
                result = pipeline.score("query", "context", "response")
                expected = 0.6 * 1.0 + 0.4 * 0.5
                assert abs(result.final_score - expected) < 1e-6

    def test_pipeline_latency_fields_populated(self):
        with patch.object(self.pipeline.nli, "score", return_value=0.8):
            result = self.pipeline.score("query", "context", "response")
            assert result.nli_latency_ms >= 0
            assert result.llm_latency_ms is None

    def test_pipeline_handles_all_zero_nli_scores(self):
        with patch.object(self.pipeline.nli, "score", return_value=0.0):
            result = self.pipeline.score("query", "context", "response")
            assert result.nli_score == 0.0
            assert result.final_score == 0.0
            assert result.used_llm_judge is False
