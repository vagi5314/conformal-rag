"""Python client SDK for conformal-rag."""

from __future__ import annotations

from typing import List, Optional

from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.domain import DomainManager
from conformal_rag.models import ConformalResult, CalibrationStats


class ConformalRAG:
    """Python client for conformal RAG.

    Usage:
        rag = ConformalRAG(alpha=0.05)
        rag.index(documents)
        result = rag.query("What is Python?")
        print(result.risk_score)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        vector_store: str = "qdrant",
        llm_provider: str = "groq",
        scoring: str = "hybrid",
    ):
        self.alpha = alpha
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.scoring = scoring

        self.engine = ConformalRAGEngine(alpha=alpha)
        self.domain_manager = DomainManager()
        self.domains = self.domain_manager.domains

    def index(self, documents: list, chunk_config: Optional[dict] = None) -> None:
        """Index documents for retrieval.

        Args:
            documents: List of Document objects or dicts.
            chunk_config: Optional chunking configuration.
        """
        pass

    def query(
        self,
        query: str,
        domain: Optional[str] = None,
        context: Optional[List[str]] = None,
    ) -> ConformalResult:
        """Query with risk control.

        Args:
            query: The user query.
            domain: Optional domain tag.
            context: Optional pre-retrieved context.

        Returns:
            ConformalResult with risk score and abstention.
        """
        if context is None:
            context = ["placeholder context"]

        return self.engine.predict(
            query=query,
            context="\n".join(context),
            response="",
            domain=domain,
        )

    def calibrate(
        self,
        domain: str = "default",
        alpha: Optional[float] = None,
        data: Optional[list] = None,
    ) -> CalibrationStats:
        """Calibrate on labeled data.

        Args:
            domain: Domain to calibrate.
            alpha: Optional alpha override.
            data: List of calibration samples.

        Returns:
            CalibrationStats.
        """
        if alpha is not None:
            self.engine.alpha = alpha

        if data is None:
            data = []

        return self.engine.calibrate(
            queries=[d.get("query", "") for d in data],
            contexts=[d.get("context", "") for d in data],
            responses=[d.get("response", "") for d in data],
            labels=[d.get("label", 0) for d in data],
            domain=domain,
        )

    def register_domain(self, domain: str, alpha: float = 0.05) -> None:
        """Register a new domain."""
        self.domain_manager.register_domain(domain, alpha)

    def monitor_drift(self, domain: str):
        """Monitor drift for a domain."""
        return self.domain_manager.monitor_drift(domain)
