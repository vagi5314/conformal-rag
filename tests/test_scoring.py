"""Tests for hybrid scoring pipeline."""

import pytest
from unittest.mock import patch, MagicMock
from conformal_rag.scoring import ScoringPipeline, NLIScorer, LLMJudge
from conformal_rag.models import ScoringBreakdown


class TestNLIScorer:
    def test_initializes_with_models(self):
        scorer = NLIScorer()
        assert scorer.primary_model is not None
        assert scorer.fallback_model is not None

    def test_scores_faithfulness(self):
        scorer = NLIScorer()
        # Mock the model to avoid downloading
        with patch.object(scorer, "_predict", return_value=0.9):
            score = scorer.score("query", "context", "response")
            assert 0.0 <= score <= 1.0


class TestLLMJudge:
    def test_initializes_with_config(self):
        judge = LLMJudge()
        assert judge.config.temperature == 0.0

    def test_builds_prompt(self):
        judge = LLMJudge()
        prompt = judge._build_prompt("What is Python?", "Python is a language.", "Python is a language.")
        assert "What is Python?" in prompt
        assert "Python is a language." in prompt


class TestScoringPipeline:
    def test_scores_with_nli_only(self):
        pipeline = ScoringPipeline(use_llm_judge=False)
        with patch.object(pipeline.nli, "score", return_value=0.85):
            result = pipeline.score("query", "context", "response")
            assert isinstance(result, ScoringBreakdown)
            assert result.used_llm_judge is False
            assert result.final_score == 0.85

    def test_scores_with_hybrid(self):
        pipeline = ScoringPipeline(use_llm_judge=True)
        with patch.object(pipeline.nli, "score", return_value=0.7), \
             patch.object(pipeline.llm_judge, "score", return_value=0.9):
            result = pipeline.score("query", "context", "response")
            assert result.used_llm_judge is True
            assert 0.0 <= result.final_score <= 1.0
