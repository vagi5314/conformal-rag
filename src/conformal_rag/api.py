"""FastAPI endpoints for conformal-rag."""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.domain import DomainManager
from conformal_rag.monitoring import FDRMonitor


class PredictRequest(BaseModel):
    query: str
    context: List[str]
    response: str
    model: Optional[str] = None
    alpha: Optional[float] = None
    domain: Optional[str] = None


class CalibrateRequest(BaseModel):
    data: List[dict]
    alpha: float = 0.05
    domain: str = "default"


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Conformal RAG",
        description="FDR-guaranteed hallucination bounds for RAG systems",
        version="0.1.0",
    )

    engine = ConformalRAGEngine()
    domain_manager = DomainManager()
    monitor = FDRMonitor()

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/metrics")
    async def metrics():
        return monitor.get_metrics()

    @app.post("/predict")
    async def predict(request: PredictRequest):
        result = engine.predict(
            query=request.query,
            context="\n".join(request.context),
            response=request.response,
            domain=request.domain,
        )
        return result.model_dump()

    @app.post("/calibrate")
    async def calibrate(request: CalibrateRequest):
        stats = engine.calibrate(
            queries=[d["query"] for d in request.data],
            contexts=[d["context"] for d in request.data],
            responses=[d["response"] for d in request.data],
            labels=[d["label"] for d in request.data],
            domain=request.domain,
        )
        return stats.model_dump()

    return app
