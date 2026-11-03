"""Conformal Risk Control engine for RAG hallucination detection.

Score convention (must match scoring.py): HIGH score = MORE faithful.
The engine approves when score > threshold and abstains otherwise.

Guarantee (Bates et al. 2021, Conformal Risk Control, Theorem 1):
with the threshold computed by `_crc_threshold` over an exchangeable
calibration set, E[ P(next answer is approved AND wrong) ] <= alpha.
Equivalently, FDR <= alpha / coverage. The conditional claim
P(wrong | approved) <= alpha is not achievable in general for CRC
(see "When Can CRC Certify LLM Outputs?", 2026); docs state the
marginal guarantee only.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from conformal_rag.models import (
    ConformalResult,
    CalibrationStats,
)
from conformal_rag.scoring import ScoringPipeline
from conformal_rag.guards import (
    clamp_alpha,
    check_calibration_coverage,
    validate_label_distribution,
)
from conformal_rag.monitoring import FDRMonitor
from conformal_rag.online import OnlineConformal

logger = logging.getLogger(__name__)

# Scores are in [0, 1]; this is "above every possible score".
_ABSTAIN_ON_EVERYTHING = float(np.nextafter(1.0, 2.0))


class ConformalRAGEngine:
    """Core CRC engine with per-domain calibration.

    Provides the CRC guarantee: E[P(approved AND hallucinated)] <= alpha,
    once calibrated on a large enough exchangeable calibration set.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        scoring: Optional[ScoringPipeline] = None,
        weights: tuple[float, float] = (0.6, 0.4),
        fdr_monitor: Optional[FDRMonitor] = None,
        online: Optional[OnlineConformal] = None,
    ):
        self.alpha = clamp_alpha(alpha)
        self.scoring = scoring or ScoringPipeline(weights=weights)
        self.weights = weights
        self.fdr_monitor = fdr_monitor or FDRMonitor(alpha=self.alpha)
        self.online = online
        self._threshold: Optional[float] = None
        self._calibration_scores: List[float] = []
        self._calibration_labels: List[int] = []
        self._confidence_interval: tuple = (0.0, 1.0)
        self._domain: str = "default"

    @property
    def calibrated(self) -> bool:
        return self._threshold is not None

    def recalibrate_from_stored(self, alpha: float) -> None:
        """Recompute the CRC threshold for a new alpha from stored scores.

        Changing alpha does not require re-scoring: the calibration scores and
        labels are already in memory, so only the rank cutoff moves. This makes
        the tolerance control live instead of destroying the calibration.
        """
        if not self._calibration_scores:
            return
        self.alpha = clamp_alpha(alpha)
        self._threshold = self._crc_threshold(
            self._calibration_scores, self._calibration_labels, self.alpha
        )
        self._confidence_interval = self._conformal_lower_bound(
            self._calibration_scores, self.alpha
        )

    def calibration_stats(self) -> Optional[CalibrationStats]:
        """Rebuild CalibrationStats from stored calibration data.

        Used after `load_state()` restores a calibration from disk, where the
        original return value is long gone but the threshold, scores and labels
        are all still available.
        """
        if not self.calibrated or not self._calibration_scores:
            return None
        scores = self._calibration_scores
        labels = self._calibration_labels
        approved_idx = [i for i, s in enumerate(scores) if s > self._threshold]
        approved_hallucinations = sum(1 for i in approved_idx if labels[i] == 1)
        withheld_idx = [i for i in range(len(scores)) if i not in set(approved_idx)]
        return CalibrationStats(
            domain=self._domain,
            alpha=self.alpha,
            n_samples=len(labels),
            n_hallucinations=sum(labels),
            threshold=self._threshold,
            estimated_fdr=(
                approved_hallucinations / len(approved_idx) if approved_idx else 0.0
            ),
            approved_wrong_rate=(
                approved_hallucinations / len(scores) if scores else 0.0
            ),
            false_alarm_rate=(
                sum(1 for i in withheld_idx if labels[i] == 0) / len(withheld_idx)
                if withheld_idx
                else 0.0
            ),
            coverage=len(approved_idx) / len(scores) if scores else 1.0,
            min_samples_required=int(np.ceil(1.0 / self.alpha)) - 1,
            certifiable=self._threshold < _ABSTAIN_ON_EVERYTHING,
            weights=self.weights,
            warnings=[],
        )

    def calibrate(
        self,
        queries: List[str],
        contexts: List[str],
        responses: List[str],
        labels: List[int],
        domain: str = "default",
    ) -> CalibrationStats:
        """Calibrate on labeled data to compute the CRC threshold.

        Args:
            queries: List of queries.
            contexts: List of retrieved contexts.
            responses: List of LLM responses.
            labels: List of labels (0=correct, 1=hallucination).
            domain: Domain tag for per-domain calibration.

        Returns:
            CalibrationStats with threshold and in-sample estimates.

        Raises:
            ValueError: if the label set is degenerate (see guards).
        """
        valid, msg = validate_label_distribution(labels)
        if not valid:
            raise ValueError(f"Calibration rejected: {msg}")
        warnings = check_calibration_coverage(len(labels))

        self._domain = domain
        breakdowns = self.scoring.score_batch(queries, contexts, responses)
        scores = [b.final_score for b in breakdowns]

        self._calibration_scores = scores
        self._calibration_labels = list(labels)
        self._threshold = self._crc_threshold(scores, labels, self.alpha)
        self._confidence_interval = self._conformal_lower_bound(scores, self.alpha)

        if self._threshold >= _ABSTAIN_ON_EVERYTHING:
            min_n = int(np.ceil(1.0 / self.alpha)) - 1
            warnings.append(
                f"{len(labels)} samples is too few to certify alpha={self.alpha}. "
                f"The system will abstain on everything until calibrated with "
                f"at least {min_n} exchangeable samples."
            )
            logger.warning("Calibration with %d samples cannot certify alpha=%s", len(labels), self.alpha)

        # In-sample estimates under the approval rule (approve iff score > threshold).
        approved_idx = [i for i, s in enumerate(scores) if s > self._threshold]
        approved_hallucinations = sum(1 for i in approved_idx if labels[i] == 1)
        estimated_fdr = (
            approved_hallucinations / len(approved_idx) if approved_idx else 0.0
        )
        coverage = len(approved_idx) / len(scores) if scores else 1.0
        # The guarantee quantity: approved AND wrong, over all samples.
        approved_wrong_rate = (
            approved_hallucinations / len(scores) if scores else 0.0
        )
        withheld_idx = [i for i in range(len(scores)) if i not in set(approved_idx)]
        false_alarm_rate = (
            sum(1 for i in withheld_idx if labels[i] == 0) / len(withheld_idx)
            if withheld_idx
            else 0.0
        )

        return CalibrationStats(
            domain=domain,
            alpha=self.alpha,
            n_samples=len(labels),
            n_hallucinations=sum(labels),
            threshold=self._threshold,
            estimated_fdr=estimated_fdr,
            approved_wrong_rate=approved_wrong_rate,
            false_alarm_rate=false_alarm_rate,
            coverage=coverage,
            min_samples_required=int(np.ceil(1.0 / self.alpha)) - 1,
            certifiable=self._threshold < _ABSTAIN_ON_EVERYTHING,
            weights=self.weights,
            warnings=warnings,
        )

    @staticmethod
    def _crc_threshold(scores: List[float], labels: List[int], alpha: float) -> float:
        """Bates et al. (2021) CRC threshold for monotone 0/1 loss.

        Loss L_i(t) = 1[label_i = 1] * 1[score_i > t] is monotone non-increasing
        in t and bounded by B = 1. Theorem 1 gives the threshold as the infimum
        t with (#{approved calibration hallucinations} + 1) / (n + 1) <= alpha,
        which reduces to the (c_max + 1)-th largest hallucination score with
        c_max = floor((n+1)*alpha) - 1. Approving strictly above it keeps the
        finite-sample guarantee E[P(approved AND wrong)] <= alpha.
        """
        n = len(scores)
        hallucination_scores = sorted(
            (s for s, lbl in zip(scores, labels) if lbl == 1), reverse=True
        )
        m = len(hallucination_scores)
        c_max = int(np.floor((n + 1) * alpha)) - 1
        if c_max >= m:
            return 0.0  # even approving everything meets the bound
        if c_max < 0:
            return _ABSTAIN_ON_EVERYTHING  # not enough data to certify anything
        return hallucination_scores[c_max]

    @staticmethod
    def _conformal_lower_bound(scores: List[float], alpha: float) -> tuple:
        """One-sided split-conformal lower bound for a new score.

        With k = ceil((n+1)*alpha) and `lower` the k-th smallest calibration
        score, exchangeability gives P(new score >= lower) >= 1 - alpha.
        """
        if not scores:
            return (0.0, 1.0)
        n = len(scores)
        k = int(np.ceil((n + 1) * alpha))
        k = max(1, min(k, n))
        lower = float(sorted(scores)[k - 1])
        return (lower, 1.0)

    def predict(
        self,
        query: str,
        context: str,
        response: str,
        domain: Optional[str] = None,
    ) -> ConformalResult:
        """Predict with risk control.

        Args:
            query: The user query.
            context: Retrieved context.
            response: LLM response.
            domain: Domain tag (overrides default).

        Returns:
            ConformalResult with faithfulness score, abstention, and breakdown.
        """
        start_time = time.time()

        # Score the sample (high = faithful)
        breakdown = self.scoring.score(query, context, response)
        risk_score = breakdown.final_score

        # Uncalibrated engines use a plain default with NO guarantee.
        threshold = self._threshold if self._threshold is not None else 0.5
        risk_bound = self.alpha

        # Approve only strictly above the threshold (ties abstain — this is
        # what makes the CRC finite-sample guarantee hold with ties present).
        abstained = risk_score <= threshold

        latency_ms = (time.time() - start_time) * 1000

        return ConformalResult(
            query=query,
            response=response,
            risk_score=risk_score,
            risk_bound=risk_bound,
            abstained=abstained,
            calibrated=self.calibrated,
            threshold=threshold,
            alpha=self.alpha,
            confidence_interval=self._confidence_interval,
            scoring_breakdown=breakdown,
            domain=domain or self._domain,
            latency_ms=latency_ms,
        )

    def record(
        self,
        query: str,
        context: str,
        response: str,
        actual_hallucination: bool,
        domain: Optional[str] = None,
    ) -> None:
        """Feed a labeled observation into monitoring and (optional) online adaptation."""
        breakdown = self.scoring.score(query, context, response)
        score = breakdown.final_score
        threshold = self._threshold if self._threshold is not None else 0.5
        # Same approval rule as predict: abstain (flag) iff score <= threshold.
        predicted_hallucination = score <= threshold

        self.fdr_monitor.record(
            predicted_hallucination=predicted_hallucination,
            actual_hallucination=actual_hallucination,
        )
        if self.online is not None:
            self.online.update(score, is_correct=not actual_hallucination)
            self._threshold = self.online.threshold

    # ---------- persistence ----------

    def save_state(self, path: Union[str, Path]) -> None:
        """Persist calibration (threshold, scores, labels, CI) to a JSON file.

        Monitoring counters and the online ACI window are runtime state and
        are not persisted.
        """
        state = {
            "version": 1,
            "alpha": self.alpha,
            "domain": self._domain,
            "threshold": self._threshold,
            "calibration_scores": self._calibration_scores,
            "calibration_labels": self._calibration_labels,
            "confidence_interval": list(self._confidence_interval),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
        logger.info("Saved calibration state to %s", path)

    def load_state(self, path: Union[str, Path]) -> bool:
        """Restore calibration from a JSON file. Returns True if loaded."""
        path = Path(path)
        if not path.is_file():
            return False
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            self._threshold = state.get("threshold")
            self._calibration_scores = list(state.get("calibration_scores", []))
            self._calibration_labels = list(state.get("calibration_labels", []))
            ci = state.get("confidence_interval", [0.0, 1.0])
            self._confidence_interval = (float(ci[0]), float(ci[1]))
            self._domain = state.get("domain", self._domain)
            # alpha must travel with the state: per-domain tolerances differ
            # (medicine 0.05 vs finance 0.15), so a restored engine that kept
            # the constructor default would silently re-threshold on the next
            # calibration and lose the contrast the demo depends on.
            stored_alpha = state.get("alpha")
            if stored_alpha is not None:
                self.alpha = clamp_alpha(float(stored_alpha))
                if self.fdr_monitor is not None:
                    self.fdr_monitor.alpha = self.alpha
                if self.online is not None:
                    self.online.alpha = self.alpha
            logger.info(
                "Loaded calibration state from %s (threshold=%s, n=%d)",
                path, self._threshold, len(self._calibration_scores),
            )
            return True
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("Could not load calibration state from %s: %s", path, e)
            return False
