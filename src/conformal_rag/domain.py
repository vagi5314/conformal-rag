"""Per-domain calibration manager."""

from __future__ import annotations

from typing import Dict, List, Optional

from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import CalibrationStats, ConformalResult, DriftStatus
from conformal_rag.guards import clamp_alpha


class DomainManager:
    """Manages per-domain conformal calibration.

    Each domain (finance, legal, medical, general) gets its own
    calibration threshold and drift monitoring.
    """

    def __init__(self):
        self.domains: Dict[str, ConformalRAGEngine] = {}
        self._domain_drift: Dict[str, list] = {}

    def register_domain(self, domain: str, alpha: float = 0.05) -> None:
        """Register a new domain with its own alpha."""
        self.domains[domain] = ConformalRAGEngine(alpha=clamp_alpha(alpha))
        self._domain_drift[domain] = []

    def calibrate(
        self,
        domain: str,
        queries: List[str],
        contexts: List[str],
        responses: List[str],
        labels: List[int],
    ) -> CalibrationStats:
        """Calibrate a specific domain."""
        if domain not in self.domains:
            self.register_domain(domain)

        return self.domains[domain].calibrate(
            queries=queries,
            contexts=contexts,
            responses=responses,
            labels=labels,
            domain=domain,
        )

    def predict(
        self,
        query: str,
        context: str,
        response: str,
        domain: Optional[str] = None,
    ) -> ConformalResult:
        """Predict with auto-domain detection if not specified."""
        if domain is None:
            domain = self._detect_domain(query)

        if domain not in self.domains:
            self.register_domain(domain)

        return self.domains[domain].predict(
            query=query,
            context=context,
            response=response,
            domain=domain,
        )

    def _detect_domain(self, query: str) -> str:
        """Simple keyword-based domain detection.

        In production, use a classifier or embeddings.
        """
        query_lower = query.lower()

        finance_keywords = [
            "revenue", "profit", "stock", "earnings", "dividend",
            "market cap", "pe ratio", "balance sheet", "cash flow",
        ]
        if any(kw in query_lower for kw in finance_keywords):
            return "finance"

        medical_keywords = [
            "diagnosis", "treatment", "symptom", "patient", "clinical",
            "dosage", "side effect", "prognosis",
        ]
        if any(kw in query_lower for kw in medical_keywords):
            return "medical"

        legal_keywords = [
            "contract", "liability", "jurisdiction", "statute", "court",
            "litigation", "compliance", "regulation",
        ]
        if any(kw in query_lower for kw in legal_keywords):
            return "legal"

        return "general"

    def monitor_drift(self, domain: str) -> DriftStatus:
        """Monitor drift for a specific domain."""
        if domain not in self.domains:
            return DriftStatus(
                domain=domain,
                drifted=False,
                fdr=0.0,
                threshold=0.5,
                samples=0,
                coverage=1.0,
                recommendation="Domain not calibrated yet",
            )

        engine = self.domains[domain]
        stats = self._domain_drift.get(domain, [])

        # Simple drift detection: compare recent FDR to alpha
        recent_fdr = stats[-1] if stats else engine.alpha

        drifted = recent_fdr > engine.alpha * 1.5  # 50% above alpha

        return DriftStatus(
            domain=domain,
            drifted=drifted,
            fdr=recent_fdr,
            threshold=engine._threshold or 0.5,
            samples=len(engine._calibration_scores),
            coverage=1.0,
            recommendation="Re-calibrate" if drifted else "No drift detected",
        )
