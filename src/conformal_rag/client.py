"""Python client SDK for conformal-rag."""

from __future__ import annotations

from typing import List, Optional

from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.domain import DomainManager
from conformal_rag.models import ConformalResult, CalibrationStats, Document
from conformal_rag.retrieval import HybridRetriever


class ConformalRAG:
    """Python client for conformal RAG.

    The client retrieves context from an indexed corpus and risk-checks a
    *provided* LLM response. It does not generate responses — that is the
    caller's RAG pipeline.

    Usage:
        rag = ConformalRAG(alpha=0.05)
        rag.index(documents)
        context = rag.retrieve("What is Python?")
        response = my_llm(context)
        result = rag.check("What is Python?", response=response)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        vector_store: str = "qdrant",
        qdrant_url: Optional[str] = None,
        llm_provider: str = "groq",
        scoring: str = "hybrid",
    ):
        self.alpha = alpha
        self.vector_store = vector_store
        self.qdrant_url = qdrant_url
        self.llm_provider = llm_provider
        self.scoring = scoring

        self.engine = ConformalRAGEngine(alpha=alpha)
        self.domain_manager = DomainManager()
        self.domains = self.domain_manager.domains
        self.retriever: Optional[HybridRetriever] = None
        self._documents: List[Document] = []

    def index(self, documents: list, chunk_config: Optional[dict] = None) -> None:
        """Index documents for retrieval (in-memory by default, Qdrant if configured)."""
        docs: List[Document] = []
        for d in documents:
            if isinstance(d, Document):
                docs.append(d)
            elif isinstance(d, dict):
                docs.append(Document(text=d.get("text", ""), metadata=d.get("metadata")))
            elif isinstance(d, str):
                docs.append(Document(text=d))
            else:
                raise TypeError(f"Unsupported document type: {type(d)!r}")
        self._documents = docs
        self.retriever = HybridRetriever(qdrant_url=self.qdrant_url)
        self.retriever.index(docs)

    def retrieve(self, query: str, final_k: Optional[int] = None) -> List[str]:
        """Retrieve fused context for a query. Returns [] if not indexed."""
        if self.retriever is None:
            return []
        return self.retriever.retrieve(query)

    def check(
        self,
        query: str,
        response: str,
        context: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> ConformalResult:
        """Risk-check a generated response against retrieved (or provided) context."""
        if context is None:
            context = self.retrieve(query)
        context_str = "\n".join(context) if context else ""
        return self.engine.predict(
            query=query,
            context=context_str,
            response=response,
            domain=domain,
        )

    def query(
        self,
        query: str,
        response: Optional[str] = None,
        domain: Optional[str] = None,
        context: Optional[List[str]] = None,
    ) -> ConformalResult:
        """Honest convenience wrapper. Requires the LLM `response` to risk-check.

        If only a query is given, call retrieve() first, generate a response in
        your pipeline, then call check().
        """
        if response is None:
            raise ValueError(
                "ConformalRAG.query requires the generated `response` to risk-check. "
                "Pass response=... or use retrieve() then check()."
            )
        return self.check(query, response, context=context, domain=domain)

    def calibrate(self, domain: str = "default", alpha: Optional[float] = None,
                  data: Optional[list] = None) -> CalibrationStats:
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
        self.domain_manager.register_domain(domain, alpha)

    def monitor_drift(self, domain: str):
        return self.domain_manager.monitor_drift(domain)
