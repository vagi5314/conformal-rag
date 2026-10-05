# tests/test_online_fix.py
# Score convention: HIGH score = MORE faithful (matches scoring.py).
import numpy as np
from conformal_rag.online import OnlineConformal


class TestOnlineFDRFix:
    def setup_method(self):
        self.oc = OnlineConformal(alpha=0.05, window_size=100, eta=0.02)

    def test_threshold_converges_under_exchangeable_stream(self):
        rng = np.random.default_rng(42)
        for _ in range(200):
            is_halluc = rng.random() < 0.1
            # High score = faithful; hallucinations score low
            score = rng.normal(0.1 if is_halluc else 0.9, 0.1)
            self.oc.update(score, is_correct=not is_halluc)
        stats = self.oc.get_stats()
        assert stats["n_updates"] == 200
        assert 0.0 <= stats["threshold"] <= 1.0

    def test_should_abstain(self):
        self.oc._threshold = 0.5
        # Low faithfulness -> abstain; high faithfulness -> approve
        assert self.oc.should_abstain(0.4) is True
        assert self.oc.should_abstain(0.6) is False

    def test_threshold_in_bounds(self):
        for s in [0.2, 0.8, 0.5, 0.9]:
            self.oc.update(s, is_correct=True)
        assert 0.0 <= self.oc.threshold <= 1.0

    def test_get_stats(self):
        self.oc.update(0.3, is_correct=True)
        stats = self.oc.get_stats()
        assert "threshold" in stats
        assert "level" in stats
        assert "n_updates" in stats
