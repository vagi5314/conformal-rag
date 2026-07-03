# tests/test_online_fix.py
import pytest
from conformal_rag.online import OnlineConformal


class TestOnlineFDRFix:
    def setup_method(self):
        self.oc = OnlineConformal(alpha=0.05, window_size=100)

    def test_fdr_computed_correctly(self):
        for _ in range(80):
            self.oc.update(0.3, is_correct=True)
        for _ in range(20):
            self.oc.update(0.8, is_correct=False)

        fdr = self.oc._compute_fdr()
        assert fdr == 0.0

    def test_fdr_with_false_positives(self):
        for _ in range(50):
            self.oc.update(0.9, is_correct=True)
        for _ in range(50):
            self.oc.update(0.9, is_correct=False)

        fdr = self.oc._compute_fdr()
        assert abs(fdr - 0.5) < 0.01

    def test_threshold_adjusts_correctly(self):
        for _ in range(100):
            self.oc.update(0.9, is_correct=True)

        assert self.oc.threshold > 0.5
