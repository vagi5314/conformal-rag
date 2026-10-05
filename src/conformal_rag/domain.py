"""Per-domain calibration manager."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import CalibrationStats, ConformalResult, DriftStatus
from conformal_rag.guards import clamp_alpha
from conformal_rag.monitoring import FDRMonitor

logger = logging.getLogger(__name__)

# Routing-state caps: enough queries for a stable centroid, bounded file size.
_MAX_ROUTING_QUERIES = 100
_MAX_QUERY_CHARS = 400
_ROUTING_STATE = "domain_routing.json"


class DomainManager:
    """Manages per-domain conformal calibration.

    Each domain gets its own calibration threshold and drift monitoring.
    Queries without an explicit domain are routed to a calibrated domain by
    embedding similarity (default) or keyword matching; routing only ever
    picks among calibrated domains, so a misroute can at worst land on
    "general", never on a domain certified with the wrong threshold.
    """

    def __init__(
        self,
        routing: str = "embedding",
        route_margin: float = 0.55,
        embedding_model: str = "intfloat/e5-base-v2",
    ):
        self.routing = routing
        self.route_margin = route_margin
        self.embedding_model = embedding_model
        self.domains: Dict[str, ConformalRAGEngine] = {}
        self.monitors: Dict[str, FDRMonitor] = {}
        self._domain_drift: Dict[str, list] = {}
        self._domain_queries: Dict[str, List[str]] = {}
        self._centroids: Dict[str, np.ndarray] = {}
        self._embedder = None  # lazy; tests may inject a fake

    def register_domain(self, domain: str, alpha: float = 0.05) -> None:
        """Register a new domain with its own alpha."""
        a = clamp_alpha(alpha)
        self.domains[domain] = ConformalRAGEngine(alpha=a)
        self.monitors[domain] = FDRMonitor(alpha=a)
        self._domain_drift[domain] = []
        self._domain_queries.setdefault(domain, [])
        self._centroids.pop(domain, None)

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

        # The engine keeps only scores/labels; routing needs the queries to
        # build the domain's embedding centroid.
        self._domain_queries[domain] = [
            q[:_MAX_QUERY_CHARS] for q in queries[:_MAX_ROUTING_QUERIES]
        ]
        self._centroids.pop(domain, None)

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

        result = self.domains[domain].predict(
            query=query,
            context=context,
            response=response,
            domain=domain,
        )
        # Count the abstention decision; true labels arrive via record().
        monitor = self.monitors.get(domain)
        if monitor is not None:
            monitor.record_prediction(abstained=result.abstained)
        return result

    def record(
        self,
        domain: str,
        predicted_hallucination: bool,
        actual_hallucination: bool,
    ) -> None:
        """Record a labeled observation for drift monitoring."""
        if domain not in self.monitors:
            self.register_domain(domain)
        self.monitors[domain].record(
            predicted_hallucination=predicted_hallucination,
            actual_hallucination=actual_hallucination,
        )

    # ---------- routing ----------

    def _get_embedder(self):
        if self._embedder is None:
            from conformal_rag.retrieval import get_embedder
            self._embedder = get_embedder(self.embedding_model)
        return self._embedder

    def _domain_centroid(self, domain: str, embedder) -> np.ndarray:
        """Cached centroid of the domain's calibration-query embeddings.

        Falls back to embedding the domain name until it has been calibrated.
        """
        cached = self._centroids.get(domain)
        if cached is not None:
            return cached
        from conformal_rag.retrieval import encode_passages
        texts = self._domain_queries.get(domain) or [domain]
        emb = encode_passages(embedder, texts)
        centroid = emb.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        self._centroids[domain] = centroid
        return centroid

    def _detect_domain(self, query: str) -> str:
        """Route a query to a calibrated domain.

        Only calibrated domains are candidates: routing to an uncalibrated
        one would apply the unguaranteed default threshold. Queries too far
        from every centroid fall back to "general".
        """
        if self.routing == "keyword":
            return self._detect_domain_keyword(query)

        candidates = [d for d, engine in self.domains.items() if engine.calibrated]
        if not candidates:
            return "general"

        try:
            embedder = self._get_embedder()
        except Exception as e:
            logger.warning(
                "Embedder unavailable for domain routing (%s); keyword fallback", e
            )
            return self._detect_domain_keyword(query)

        from conformal_rag.retrieval import encode_query
        query_emb = encode_query(embedder, query)
        best, best_sim = "general", -1.0
        for domain in candidates:
            sim = float(np.dot(query_emb, self._domain_centroid(domain, embedder)))
            if sim > best_sim:
                best, best_sim = domain, sim
        if best_sim < self.route_margin:
            return "general"
        return best

    def _detect_domain_keyword(self, query: str) -> str:
        """Legacy keyword-based domain detection (fallback / opt-out)."""
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
        monitor = self.monitors.get(domain)
        if monitor is None:
            recent_fdr = engine.alpha
            samples = 0
        else:
            m = monitor.get_metrics()
            recent_fdr = m["fdr"]
            samples = m["total_predictions"]

        drifted = recent_fdr > engine.alpha * 1.5

        return DriftStatus(
            domain=domain,
            drifted=drifted,
            fdr=recent_fdr,
            threshold=engine._threshold or 0.5,
            samples=samples,
            coverage=1.0 - recent_fdr if samples else 1.0,
            recommendation="Re-calibrate" if drifted else "No drift detected",
        )

    # ---------- persistence ----------

    def save_state(self, directory: Union[str, Path]) -> None:
        """Persist every domain's calibration into `directory`."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, (name, engine) in enumerate(self.domains.items()):
            engine.save_state(directory / f"domain_{i:03d}.json")
        routing = {
            name: queries
            for name, queries in self._domain_queries.items()
            if queries
        }
        if routing:
            (directory / _ROUTING_STATE).write_text(
                json.dumps(routing), encoding="utf-8"
            )

    def load_state(self, directory: Union[str, Path]) -> List[str]:
        """Restore domain calibrations from `directory`. Returns loaded names."""
        directory = Path(directory)
        loaded: List[str] = []
        if not directory.is_dir():
            return loaded
        for f in sorted(directory.glob("domain_*.json")):
            try:
                name = json.loads(f.read_text(encoding="utf-8")).get("domain")
            except (json.JSONDecodeError, OSError):
                logger.warning("Skipping unreadable domain state file %s", f)
                continue
            if not name:
                continue
            if name not in self.domains:
                self.register_domain(name)
            if self.domains[name].load_state(f):
                loaded.append(name)
        routing_file = directory / _ROUTING_STATE
        if routing_file.is_file():
            try:
                routing = json.loads(routing_file.read_text(encoding="utf-8"))
                for name, queries in routing.items():
                    if name in self.domains and isinstance(queries, list):
                        self._domain_queries[name] = [
                            str(q)[:_MAX_QUERY_CHARS] for q in queries
                        ]
                self._centroids.clear()
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load routing state from %s: %s", routing_file, e)
        if loaded:
            logger.info("Loaded domain calibrations: %s", ", ".join(loaded))
        return loaded
