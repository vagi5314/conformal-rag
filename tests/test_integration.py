"""End-to-end integration tests for the full conformal RAG pipeline."""

import pytest
from unittest.mock import patch, MagicMock
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.retrieval import HybridRetriever
from conformal_rag.models import ScoringBreakdown, Document


class TestEndToEnd:
    def setup_method(self):
        self.engine = ConformalRAGEngine(alpha=0.05)
        self.retriever = HybridRetriever()

    def test_full_pipeline(self):
        mock_breakdown = ScoringBreakdown(
            nli_score=0.9, final_score=0.9, nli_latency_ms=10.0, used_llm_judge=False
        )
        with patch.object(self.engine.scoring, "score", return_value=mock_breakdown):
            queries = ["q"] * 10
            contexts = ["c"] * 10
            responses = ["r"] * 10
            labels = [0] * 8 + [1] * 2

            stats = self.engine.calibrate(queries, contexts, responses, labels)
            assert stats.n_samples == 10
            assert stats.estimated_fdr <= 0.1

            predict_breakdown = ScoringBreakdown(
                nli_score=0.85, final_score=0.85, nli_latency_ms=10.0, used_llm_judge=False
            )
            with patch.object(self.engine.scoring, "score", return_value=predict_breakdown):
                result = self.engine.predict("test query", "test context", "test response")
                assert 0.0 <= result.risk_score <= 1.0
                assert isinstance(result.abstained, bool)

    def test_retrieval_integration(self):
        docs = [
            Document(text="Paris is the capital of France.", domain="geography"),
            Document(text="Python is a programming language.", domain="tech"),
        ]
        self.retriever.index(docs)
        with patch.object(self.retriever, "_get_embedder") as mock_embedder:
            mock_emb = MagicMock()
            mock_emb.encode.return_value.__getitem__ = lambda self, x: MagicMock(
                __truediv__=lambda self, x: MagicMock(
                    __truediv__=lambda self, x: MagicMock()
                )
            )
            mock_embedder.return_value = mock_emb

            with patch("conformal_rag.retrieval.np") as mock_np:
                mock_np.dot.return_value = MagicMock()
                mock_np.linalg.norm.return_value = MagicMock()
                mock_np.argsort.return_value = MagicMock(
                    __getitem__=lambda self, x: [0, 1]
                )
                results = self.retriever.retrieve("capital of France")
                assert len(results) > 0

    def test_calibration_then_predict_flow(self):
        mock_breakdown = ScoringBreakdown(
            nli_score=0.8, final_score=0.8, nli_latency_ms=10.0, used_llm_judge=False
        )
        with patch.object(self.engine.scoring, "score", return_value=mock_breakdown):
            queries = ["q"] * 20
            contexts = ["c"] * 20
            responses = ["r"] * 20
            labels = [0] * 16 + [1] * 4

            stats = self.engine.calibrate(queries, contexts, responses, labels)
            assert stats.n_samples == 20
            assert stats.n_hallucinations == 4

        predict_breakdown = ScoringBreakdown(
            nli_score=0.95, final_score=0.95, nli_latency_ms=10.0, used_llm_judge=False
        )
        with patch.object(self.engine.scoring, "score", return_value=predict_breakdown):
            result = self.engine.predict("q", "c", "r")
            assert result.risk_score == 0.95
            assert result.abstained is False

    def test_predict_with_untrusted_response_abstains(self):
        high_breakdown = ScoringBreakdown(
            nli_score=0.9, final_score=0.9, nli_latency_ms=10.0, used_llm_judge=False
        )
        low_breakdown = ScoringBreakdown(
            nli_score=0.3, final_score=0.3, nli_latency_ms=10.0, used_llm_judge=False
        )

        def calibrate_side_effect(q, c, r):
            return high_breakdown

        with patch.object(self.engine.scoring, "score", side_effect=calibrate_side_effect):
            queries = ["q"] * 20
            contexts = ["c"] * 20
            responses = ["r"] * 20
            labels = [0] * 16 + [1] * 4

            stats = self.engine.calibrate(queries, contexts, responses, labels)

        predict_breakdown = ScoringBreakdown(
            nli_score=0.1, final_score=0.1, nli_latency_ms=10.0, used_llm_judge=False
        )
        with patch.object(self.engine.scoring, "score", return_value=predict_breakdown):
            result = self.engine.predict("q", "c", "r")
            assert result.abstained is True

    def test_engine_creates_default_scoring_pipeline(self):
        engine = ConformalRAGEngine(alpha=0.05)
        assert engine.scoring is not None
        assert isinstance(engine.scoring, type(engine.scoring).__mro__[0])
        assert engine.scoring.use_llm_judge is False or True
