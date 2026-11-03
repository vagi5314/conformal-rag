"""Tests for FDR monitoring."""

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

    def test_computes_false_alarm_rate(self):
        monitor = FDRMonitor(alpha=0.05)
        # 2 of the 10 flagged answers were actually fine
        for _ in range(8):
            monitor.record(predicted_hallucination=True, actual_hallucination=True)
        for _ in range(2):
            monitor.record(predicted_hallucination=True, actual_hallucination=False)
        metrics = monitor.get_metrics()
        assert metrics["false_alarm_rate"] == 0.2
        assert metrics["approved_wrong_rate"] == 0.0

    def test_approved_wrong_rate_is_the_guarantee_quantity(self):
        monitor = FDRMonitor(alpha=0.05)
        # 1 approved answer turned out wrong, out of 10 labeled observations
        monitor.record(predicted_hallucination=False, actual_hallucination=True)
        for _ in range(9):
            monitor.record(predicted_hallucination=False, actual_hallucination=False)
        assert monitor.get_metrics()["approved_wrong_rate"] == 0.1

    def test_detects_violation_of_approved_wrong_rate(self):
        monitor = FDRMonitor(alpha=0.05)
        for _ in range(3):
            monitor.record(predicted_hallucination=False, actual_hallucination=True)
        for _ in range(7):
            monitor.record(predicted_hallucination=False, actual_hallucination=False)
        assert monitor.is_violating() is True

    def test_abstaining_on_everything_never_violates(self):
        """The old monitor called this a violation; the guarantee says otherwise."""
        monitor = FDRMonitor(alpha=0.05)
        for _ in range(10):
            monitor.record(predicted_hallucination=True, actual_hallucination=False)
        assert monitor.is_violating() is False

    def test_computes_coverage(self):
        monitor = FDRMonitor(alpha=0.05)
        for _ in range(8):
            monitor.record(predicted_hallucination=False, actual_hallucination=False)
        for _ in range(2):
            monitor.record(predicted_hallucination=True, actual_hallucination=False)
        metrics = monitor.get_metrics()
        assert metrics["coverage"] == 0.8
