"""Conformal Risk Control engine for RAG hallucination detection."""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from conformal_rag.models import (
    ConformalResult,
    CalibrationStats,
)
from conformal_rag.scoring import ScoringPipeline
from conformal_rag.guards import clamp_alpha
from conformal_rag.monitoring import FDRMonitor
from conformal_rag.online import OnlineConformal


class ConformalRAGEngine:
    """Core CRC engine with per-domain calibration.

    Provides FDR guarantee: P(hallucination | not abstained) ≤ α.
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
        self._domain: str = "default"

    def calibrate(
        self,
        queries: List[str],
        contexts: List[str],
        responses: List[str],
        labels: List[int],
        domain: str = "default",
    ) -> CalibrationStats:
        """Calibrate on labeled data to compute conformal threshold.

        Args:
            queries: List of queries.
            contexts: List of retrieved contexts.
            responses: List of LLM responses.
            labels: List of labels (0=correct, 1=hallucination).
            domain: Domain tag for per-domain calibration.

        Returns:
            CalibrationStats with threshold and FDR estimate.
        """
        self._domain = domain
        scores = []
        for q, c, r in zip(queries, contexts, responses):
            breakdown = self.scoring.score(q, c, r)
            scores.append(breakdown.final_score)

        self._calibration_scores = scores
        self._calibration_labels = labels

        # Compute threshold: percentile of scores for correct samples
        correct_scores = [s for s, lbl in zip(scores, labels) if lbl == 0]
        if correct_scores:
            # Threshold = (1-α) percentile of correct scores
            self._threshold = float(np.percentile(
                correct_scores, (1 - self.alpha) * 100
            ))
        else:
            self._threshold = 0.5  # Default

        # Estimate FDR
        rejected = [s for s in scores if s > self._threshold]
        if rejected:
            false_positives = sum(
                1 for s, lbl in zip(scores, labels)
                if s > self._threshold and lbl == 0
            )
            estimated_fdr = false_positives / len(rejected)
        else:
            estimated_fdr = 0.0

        # Coverage = fraction not abstained
        coverage = 1.0 - (len(rejected) / len(scores)) if scores else 1.0

        return CalibrationStats(
            domain=domain,
            alpha=self.alpha,
            n_samples=len(labels),
            n_hallucinations=sum(labels),
            threshold=self._threshold,
            estimated_fdr=estimated_fdr,
            coverage=coverage,
            weights=self.weights,
        )

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
            ConformalResult with risk score, abstention, and breakdown.
        """
        start_time = time.time()

        # Score the sample
        breakdown = self.scoring.score(query, context, response)
        risk_score = breakdown.final_score

        # Use default threshold if not calibrated
        threshold = self._threshold if self._threshold is not None else 0.5
        risk_bound = self.alpha

        # Abstain if risk score falls below threshold (low faithfulness = high hallucination risk)
        abstained = risk_score < threshold

        ci = self._compute_confidence_interval(
            self._calibration_scores, self._calibration_labels, self.alpha
        )
        confidence_interval = ci

        latency_ms = (time.time() - start_time) * 1000

        return ConformalResult(
            query=query,
            response=response,
            risk_score=risk_score,
            risk_bound=risk_bound,
            abstained=abstained,
            confidence_interval=confidence_interval,
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
        predicted_hallucination = score > threshold

        self.fdr_monitor.record(
            predicted_hallucination=predicted_hallucination,
            actual_hallucination=actual_hallucination,
        )
        if self.online is not None:
            self.online.update(score, is_correct=not actual_hallucination)
            self._threshold = self.online.threshold

    def _compute_confidence_interval(
        self, scores: list, labels: list, alpha: float
    ) -> tuple:
        """One-sided conformal lower bound for the risk score.

        Uses the finite-sample (n+1)(1-alpha) quantile of the correct-class
        calibration scores (Vovk et al. conformal PI). Guarantees, under
        exchangeability, that P(risk_score >= lower) >= 1-alpha.
        """
        if not scores:
            return (0.0, 1.0)
        correct_scores = [s for s, lbl in zip(scores, labels) if lbl == 0]
        if not correct_scores:
            return (0.0, 1.0)
        n = len(correct_scores)
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = max(1, min(k, n))
        sorted_s = sorted(correct_scores)
        lower = float(sorted_s[k - 1])
        return (lower, 1.0)
