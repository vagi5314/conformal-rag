"""Tests for per-domain calibration manager."""

import numpy as np
from unittest.mock import patch, MagicMock
from conformal_rag.domain import DomainManager
from conformal_rag.models import CalibrationStats, ScoringBreakdown


def _breakdown(score: float = 0.9) -> ScoringBreakdown:
    return ScoringBreakdown(
        nli_score=score, final_score=score, nli_latency_ms=10.0, used_llm_judge=False
    )


class _FakeEmbedder:
    """Deterministic embedder: keyword overlap drives the vector direction."""

    KEYWORDS = ["revenue", "profit", "stock", "diagnosis", "patient", "treatment"]

    def encode(self, texts):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vectors = []
        for text in items:
            tokens = set(text.lower().split())
            vectors.append([1.0 if kw in tokens else 0.0 for kw in self.KEYWORDS])
        return np.asarray(vectors, dtype=np.float32)[0] if single else np.asarray(vectors, dtype=np.float32)


class TestDomainManager:
    def test_initializes_empty(self):
        dm = DomainManager()
        assert len(dm.domains) == 0

    def test_registers_domain(self):
        dm = DomainManager()
        dm.register_domain("finance", alpha=0.01)
        assert "finance" in dm.domains
        assert dm.domains["finance"].alpha == 0.01

    def test_calibrate_per_domain(self):
        dm = DomainManager()
        dm.register_domain("finance", alpha=0.01)
        with patch.object(dm.domains["finance"], "calibrate") as mock_cal:
            mock_cal.return_value = CalibrationStats(
                domain="finance",
                alpha=0.01,
                n_samples=100,
                n_hallucinations=5,
                threshold=0.45,
                estimated_fdr=0.008,
                coverage=0.85,
            )
            stats = dm.calibrate(
                domain="finance",
                queries=["q1"],
                contexts=["c1"],
                responses=["r1"],
                labels=[0],
            )
            assert stats.domain == "finance"

    def test_predict_auto_routes_domain(self):
        dm = DomainManager()
        dm.register_domain("finance", alpha=0.01)
        dm.register_domain("general", alpha=0.10)

        with patch.object(dm, "_detect_domain", return_value="finance"):
            with patch.object(dm.domains["finance"], "predict") as mock_pred:
                mock_pred.return_value = MagicMock(domain="finance", abstained=False)
                result = dm.predict(
                    query="What is P/E ratio?",
                    context="context",
                    response="response",
                )
                assert result.domain == "finance"

    def test_predict_records_to_domain_monitor(self):
        dm = DomainManager()
        dm.register_domain("finance", alpha=0.01)
        with patch.object(dm, "_detect_domain", return_value="finance"):
            with patch.object(dm.domains["finance"], "predict") as mock_pred:
                mock_pred.return_value = MagicMock(domain="finance", abstained=True)
                dm.predict(query="q", context="c", response="r")
        metrics = dm.monitors["finance"].get_metrics()
        assert metrics["total_predictions"] == 1
        assert metrics["abstention_rate"] == 1.0

    def test_drift_monitoring(self):
        dm = DomainManager()
        dm.register_domain("finance", alpha=0.01)
        status = dm.monitor_drift("finance")
        assert status.domain == "finance"


class TestKeywordRouting:
    def test_routes_finance(self):
        dm = DomainManager(routing="keyword")
        assert dm._detect_domain("What was the quarterly revenue?") == "finance"

    def test_routes_medical(self):
        dm = DomainManager(routing="keyword")
        assert dm._detect_domain("What is the diagnosis?") == "medical"

    def test_routes_legal(self):
        dm = DomainManager(routing="keyword")
        assert dm._detect_domain("Is the contract binding?") == "legal"

    def test_falls_back_to_general(self):
        dm = DomainManager(routing="keyword")
        assert dm._detect_domain("Tell me about penguins") == "general"


class TestEmbeddingRouting:
    def _calibrated_dm(self):
        dm = DomainManager(routing="embedding", route_margin=0.5)
        dm._embedder = _FakeEmbedder()
        for domain, query in [
            ("finance", "quarterly revenue and profit outlook"),
            ("medical", "patient diagnosis and treatment plan"),
        ]:
            dm.register_domain(domain)
            with patch.object(
                dm.domains[domain].scoring, "score", return_value=_breakdown(0.8)
            ):
                dm.calibrate(
                    domain=domain,
                    queries=[query],
                    contexts=["c"],
                    responses=["r"],
                    labels=[0, 1],
                )
        return dm

    def test_routes_similar_query_to_domain(self):
        dm = self._calibrated_dm()
        assert dm._detect_domain("revenue profit forecast") == "finance"
        assert dm._detect_domain("diagnosis for the patient") == "medical"

    def test_dissimilar_query_falls_back_to_general(self):
        dm = self._calibrated_dm()
        assert dm._detect_domain("tell me about penguins") == "general"

    def test_uncalibrated_domains_are_not_candidates(self):
        dm = self._calibrated_dm()
        dm.register_domain("legal")  # registered but never calibrated
        with patch.object(dm, "_detect_domain_keyword", return_value="legal") as kw:
            # Even if keyword fallback would say legal, embedding routing only
            # considers calibrated domains; a legal-ish query has no matching
            # centroid and lands on general.
            assert dm._detect_domain("contract liability") == "general"
            kw.assert_not_called()

    def test_no_calibrated_domains_returns_general(self):
        dm = DomainManager(routing="embedding")
        dm._embedder = _FakeEmbedder()
        dm.register_domain("finance")
        assert dm._detect_domain("revenue profit") == "general"

    def test_centroid_updates_after_recalibration(self):
        dm = self._calibrated_dm()
        embedder = dm._embedder
        before = dm._domain_centroid("finance", embedder).copy()
        with patch.object(
            dm.domains["finance"].scoring, "score", return_value=_breakdown(0.8)
        ):
            dm.calibrate(
                domain="finance",
                queries=["patient diagnosis treatment"],
                contexts=["c"],
                responses=["r"],
                labels=[0, 1],
            )
        after = dm._domain_centroid("finance", embedder)
        assert not np.allclose(before, after)

    def test_embedder_failure_falls_back_to_keyword(self):
        dm = self._calibrated_dm()
        with patch.object(
            dm, "_get_embedder", side_effect=RuntimeError("no torch")
        ):
            assert dm._detect_domain("What was the quarterly revenue?") == "finance"

    def test_routing_state_round_trip(self, tmp_path):
        dm = self._calibrated_dm()
        dm.save_state(tmp_path)

        fresh = DomainManager(routing="embedding", route_margin=0.5)
        fresh._embedder = _FakeEmbedder()
        loaded = fresh.load_state(tmp_path)
        assert "finance" in loaded and "medical" in loaded
        assert fresh._domain_queries["finance"] == ["quarterly revenue and profit outlook"]
        assert fresh._detect_domain("revenue profit forecast") == "finance"

    def test_load_state_without_routing_file_still_works(self, tmp_path):
        dm = self._calibrated_dm()
        dm.save_state(tmp_path)
        (tmp_path / "domain_routing.json").unlink()

        fresh = DomainManager()
        loaded = fresh.load_state(tmp_path)
        assert "finance" in loaded
        assert fresh._domain_queries["finance"] == []
