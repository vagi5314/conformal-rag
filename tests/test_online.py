"""Tests for online adaptive conformal."""

import numpy as np
from conformal_rag.online import OnlineConformal


class TestOnlineConformal:
    def test_initializes_with_defaults(self):
        oc = OnlineConformal(alpha=0.05)
        assert oc.alpha == 0.05
        assert oc.window_size == 500
        assert oc.eta == 0.01

    def test_updates_threshold(self):
        oc = OnlineConformal(alpha=0.05, window_size=100, eta=0.05)
        for _ in range(50):
            oc.update(score=0.2, is_correct=True)
        stats = oc.get_stats()
        assert 0.0 <= stats["threshold"] <= 1.0

    def test_abstains_on_high_risk(self):
        oc = OnlineConformal(alpha=0.05)
        oc._threshold = 0.45
        assert oc.should_abstain(0.5) is True
        assert oc.should_abstain(0.3) is False

    def test_converges_under_exchangeable_stream(self):
        rng = np.random.default_rng(0)
        oc = OnlineConformal(alpha=0.1, window_size=200, eta=0.02, initial_level=0.9)
        n = 4000
        miscoverage = 0
        for _ in range(n):
            is_halluc = rng.random() < 0.1
            score = rng.normal(0.9 if is_halluc else 0.1, 0.1)
            oc.update(score, is_correct=not is_halluc)
            if score > oc.threshold and not is_halluc:
                miscoverage += 1
        rate = miscoverage / n
        assert abs(rate - 0.1) < 0.1

    def test_window_maxsize(self):
        oc = OnlineConformal(alpha=0.05, window_size=10)
        for i in range(20):
            oc.update(score=0.3, is_correct=True)
        assert len(oc.window) == 10
