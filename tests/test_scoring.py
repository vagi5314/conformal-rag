"""Tests for hybrid scoring pipeline."""

import math

import torch
from unittest.mock import patch, MagicMock
from conformal_rag.scoring import ScoringPipeline, NLIScorer, LLMJudge
from conformal_rag.models import ScoringBreakdown


class _FakeTokenizer:
    """Forwards (context, claim) through so the fake model can see claims."""

    def __call__(self, context, claim, **kwargs):
        return {"claim": claim}


class _FakeModel:
    """Deterministic entailment probability derived from each claim's text.

    Produces 3-class logits whose softmax at the entailment index equals
    `ent_prob(claim)`, so batched and per-sample paths are directly comparable.
    """

    @staticmethod
    def ent_prob(claim: str) -> float:
        return (len(claim) % 9 + 1) / 10.0  # 0.1 .. 0.9, never 0 or 1

    def __call__(self, **inputs):
        claims = inputs["claim"]
        if isinstance(claims, str):
            claims = [claims]
        rows = []
        for claim in claims:
            p = self.ent_prob(claim)
            side = math.log((1.0 - p) / 2.0)
            rows.append([math.log(p), side, side])
        result = MagicMock()
        result.logits = torch.tensor(rows, dtype=torch.float32)
        return result


def _fake_nli_scorer() -> NLIScorer:
    scorer = NLIScorer()
    scorer._tokenizer = _FakeTokenizer()
    scorer._model = _FakeModel()
    scorer._entailment_idx = 0
    scorer._loaded = True
    return scorer


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

    def test_score_batch_matches_per_sample(self):
        scorer = _fake_nli_scorer()
        pairs = [
            ("ctx one", "The Eiffel Tower is in Paris. It is tall."),
            ("ctx two", "Python is a language."),
            ("ctx three", "Short. Another claim here for good measure."),
        ]
        batched = scorer.score_batch(pairs)
        per_sample = [scorer.score("q", c, r) for c, r in pairs]
        assert len(batched) == len(per_sample)
        for b, s in zip(batched, per_sample):
            assert abs(b - s) < 1e-6

    def test_score_batch_empty_decomposition_returns_neutral(self):
        scorer = _fake_nli_scorer()
        # Response decomposes to no claims (>10 char sentences only).
        batched = scorer.score_batch([("ctx", "tiny")])
        assert batched == [0.5]


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
