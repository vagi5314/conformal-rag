"""Tests for FDR monitoring."""

import pytest
from conformal_rag.monitoring import FDRMonitor


class TestFDRMonitor:
    def test_initializes(self):
        monitor = FDRMonitor(alpha=0.05)
        assert monitor.alpha == 0.05

    def test_records_predictions(self):
        monitor = FDRMonitor(alpha=0.05)
        monitor.record(predicted_hallucination=True, actual_hallucination=False)
        monitor.record(predicted_hallucination=False, actual_hallucination=False)
        metrics = monitor.get_metrics()
        assert metrics["total_predictions"] == 2

    def test_computes_fdr(self):
        monitor = FDRMonitor(alpha=0.05)
        # 2 false positives out of 10 predicted hallucinations
        for _ in range(8):
            monitor.record(predicted_hallucination=True, actual_hallucination=True)
        for _ in range(2):
            monitor.record(predicted_hallucination=True, actual_hallucination=False)
        metrics = monitor.get_metrics()
        assert metrics["fdr"] == 0.2

    def test_detects_fdr_violation(self):
        monitor = FDRMonitor(alpha=0.05)
        # FDR way above alpha
        for _ in range(1):
            monitor.record(predicted_hallucination=True, actual_hallucination=True)
        for _ in range(9):
            monitor.record(predicted_hallucination=True, actual_hallucination=False)
        assert monitor.is_violating() is True

    def test_computes_coverage(self):
        monitor = FDRMonitor(alpha=0.05)
        for _ in range(8):
            monitor.record(predicted_hallucination=False, actual_hallucination=False)
        for _ in range(2):
            monitor.record(predicted_hallucination=True, actual_hallucination=False)
        metrics = monitor.get_metrics()
        assert metrics["coverage"] == 0.8
