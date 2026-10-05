# tests/test_nli.py
from conformal_rag.scoring import NLIScorer


class TestNLIScorer:
    def setup_method(self):
        self.scorer = NLIScorer()

    def test_score_returns_float_between_0_and_1(self):
        score = self.scorer.score(
            query="What is Python?",
            context="Python is a programming language created by Guido van Rossum.",
            response="Python is a programming language."
        )
        assert 0.0 <= score <= 1.0

    def test_faithful_response_scores_high(self):
        score = self.scorer.score(
            query="What is Python?",
            context="Python is a programming language.",
            response="Python is a programming language."
        )
        assert score > 0.5

    def test_hallucination_scores_low(self):
        score = self.scorer.score(
            query="What is Python?",
            context="Python is a programming language.",
            response="Python is a type of snake found in Africa."
        )
        assert score < 0.8

    def test_decompose_splits_sentences(self):
        claims = self.scorer._decompose("Paris is the capital. France is in Europe.")
        assert len(claims) >= 2

    def test_score_claim_returns_float(self):
        score = self.scorer._score_claim(
            context="Python is a language.",
            claim="Python is a language."
        )
        assert 0.0 <= score <= 1.0
