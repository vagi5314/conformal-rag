"""Tests for online adaptive conformal."""

from conformal_rag.online import OnlineConformal


class TestOnlineConformal:
    def test_initializes_with_defaults(self):
        oc = OnlineConformal(alpha=0.05)
        assert oc.alpha == 0.05
        assert oc.window_size == 500
        assert oc.eta == 0.01

    def test_updates_threshold(self):
        oc = OnlineConformal(alpha=0.05, window_size=100)
        # Add 50 correct samples (score 0.2, label 0)
        for _ in range(50):
            oc.update(score=0.2, is_correct=True)
        # Threshold should be near 0.2
        assert 0.1 <= oc.threshold <= 0.6

    def test_abstains_on_high_risk(self):
        oc = OnlineConformal(alpha=0.05)
        oc._threshold = 0.45
        assert oc.should_abstain(0.5) is True
        assert oc.should_abstain(0.3) is False

    def test_computes_fdr(self):
        oc = OnlineConformal(alpha=0.05, window_size=100)
        # Add samples: some correct, some wrong
        for _ in range(10):
            oc.update(score=0.2, is_correct=True)  # Below threshold
        for _ in range(5):
            oc.update(score=0.8, is_correct=False)  # Above threshold, wrong
        fdr = oc._compute_fdr()
        assert 0.0 <= fdr <= 1.0

    def test_window_maxsize(self):
        oc = OnlineConformal(alpha=0.05, window_size=10)
        for i in range(20):
            oc.update(score=0.3, is_correct=True)
        assert len(oc.window) == 10
