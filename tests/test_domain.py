"""Tests for per-domain calibration manager."""

import pytest
from unittest.mock import patch, MagicMock
from conformal_rag.domain import DomainManager
from conformal_rag.models import CalibrationStats


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
                mock_pred.return_value = MagicMock(domain="finance")
                result = dm.predict(
                    query="What is P/E ratio?",
                    context="context",
                    response="response",
                )
                assert result.domain == "finance"

    def test_drift_monitoring(self):
        dm = DomainManager()
        dm.register_domain("finance", alpha=0.01)
        status = dm.monitor_drift("finance")
        assert status.domain == "finance"
