"""FastAPI endpoints for conformal-rag.

Serves the REST API and the static frontend (frontend/) from one process,
so the browser UI runs same-origin with no CORS configuration.

Handlers that call blocking model code (NLI scoring, embedding, calibration
loops) are plain `def`, so FastAPI runs them on the threadpool instead of
stalling the event loop.

Production hardening (all opt-in via environment, see config.py):
- CONFORMAL_RAG_API_KEY: require X-API-Key header on every route but /health
- CONFORMAL_RAG_RATE_LIMIT_PER_MINUTE: per-client-IP sliding window limit
- CONFORMAL_RAG_CORS_ORIGINS: comma-separated allowed origins
- CONFORMAL_RAG_STATE_DIR: persist calibration + index state across restarts
- CONFORMAL_RAG_LOG_LEVEL: package log level
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from conformal_rag.config import Settings
from conformal_rag.demo_corpus import DOMAINS as DEMO_DOMAINS
from conformal_rag.demo_corpus import MAIN_ALPHA, SHOWCASE
from conformal_rag.documents import (
    DocumentError,
    chunk_text,
    extract_text,
    generate_answer,
    is_exact_quote,
    optional_client,
    preview,
)
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.domain import DomainManager
from conformal_rag.models import CalibrationSample, Document
from conformal_rag.monitoring import FDRMonitor
from conformal_rag.online import OnlineConformal
from conformal_rag.retrieval import HybridRetriever
from conformal_rag.scoring import ScoringPipeline

logger = logging.getLogger(__name__)

SAMPLE_DOCUMENTS = [
    "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability and supports multiple programming paradigms.",
    "The capital of France is Paris. Paris is the most populous city in France, with a population of over 2 million people.",
    "Conformal prediction is a framework for constructing prediction sets with guaranteed coverage. It provides distribution-free uncertainty quantification.",
    "DeBERTa-v3 is a state-of-the-art transformer model for natural language understanding. It achieves top results on NLU benchmarks.",
    "FastAPI is a modern web framework for building APIs with Python. It provides automatic OpenAPI documentation and async support.",
    "The Eiffel Tower was built in 1889 for the World's Fair. It stands 330 meters tall and was the tallest structure in the world until 1930.",
    "Conformal Risk Control (CRC) extends conformal prediction to control the False Discovery Rate (FDR) in classification and sequence tasks.",
    "Sentence Transformers are used for computing dense embeddings of text. The e5-base-v2 model produces 768-dimensional embeddings.",
]

# Request size limits (defense against memory/DoS abuse).
_MAX_TEXT_CHARS = 20_000
_MAX_DOCS = 500
_MAX_CALIBRATION_SAMPLES = 200

# Upload limits. Documents live in memory and are never persisted, so the cap
# is about keeping one process healthy, not about storage.
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_MAX_UPLOADS = 8
_UPLOAD_FINAL_K = 5
# When the model declines, read more of the document before giving up: the
# answer is often one paragraph past the first pass, not absent.
_UPLOAD_RETRY_K = 10

# An uploaded document is outside the demo calibration distribution, so the
# threshold is the checker's, not a certified bound for that document.
_UPLOAD_CAVEAT = (
    "The tolerance is set on the demo calibration set, and this document sits "
    "outside it. Read the verdict as the checker's reading of this answer, not "
    "as a bound certified for this document."
)

# Routes exempt from API-key auth (liveness probes, static frontend).
_AUTH_EXEMPT_PATHS = {"/health", "/openapi.json", "/docs", "/redoc"}

# State file names inside settings.state_dir.
_ENGINE_STATE = "engine_state.json"
_INDEX_STATE = "index_state.json"
_DOMAIN_DIR = "domains"


class ConfigRequest(BaseModel):
    alpha: float = 0.05
    online: bool = True
    nli_model: Optional[str] = None
    use_llm_judge: Optional[bool] = None
    # Scope the tolerance change to one calibrated domain instead of the main
    # engine. Omit to change the main (general) tolerance.
    domain: Optional[str] = None


class IndexRequest(BaseModel):
    documents: List[str] = Field(max_length=_MAX_DOCS)


class RetrieveRequest(BaseModel):
    query: str = Field(max_length=_MAX_TEXT_CHARS)
    final_k: int = Field(default=5, ge=1, le=50)


class PredictRequest(BaseModel):
    query: str = Field(max_length=_MAX_TEXT_CHARS)
    context: List[str] = Field(max_length=_MAX_DOCS)
    response: str = Field(max_length=_MAX_TEXT_CHARS)
    model: Optional[str] = None
    alpha: Optional[float] = None
    domain: Optional[str] = None


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=_MAX_TEXT_CHARS)
    doc_id: str = Field(min_length=1, max_length=64)
    domain: Optional[str] = None


@dataclass
class UploadedDocument:
    """One upload, kept with its own passage index.

    The index is scoped to the upload so a question is answered only from that
    document, never from the demo corpus it happens to resemble.
    """

    doc_id: str
    filename: str
    kind: str
    pages: int
    characters: int
    passages: List[str]
    preview: str
    retriever: HybridRetriever


class CalibrateRequest(BaseModel):
    data: List[CalibrationSample] = Field(max_length=_MAX_CALIBRATION_SAMPLES)
    domain: str = "default"


class RecordRequest(BaseModel):
    query: str = Field(max_length=_MAX_TEXT_CHARS)
    context: str = Field(max_length=_MAX_TEXT_CHARS)
    response: str = Field(max_length=_MAX_TEXT_CHARS)
    actual_hallucination: bool


class DomainRequest(BaseModel):
    name: str = Field(max_length=200)
    alpha: float = 0.05


class FeedbackRequest(BaseModel):
    domain: Optional[str] = "default"
    predicted_hallucination: bool = False
    actual_hallucination: bool = False


def _build_engine(
    alpha: float = 0.05,
    online: bool = True,
    nli_model: str = "small",
    use_llm_judge: bool = True,
    fallback_chain: Optional[List[str]] = None,
) -> ConformalRAGEngine:
    """Build an engine with a wired FDR monitor and optional online ACI.

    The judge is only enabled when a provider key is actually present: with no
    key, attempting it burns seconds on four guaranteed failures and then
    degrades anyway. `ScoringPipeline` still degrades gracefully if the key is
    present but the provider is down.
    """
    judge_on = use_llm_judge and _judge_configured()
    return ConformalRAGEngine(
        alpha=alpha,
        scoring=ScoringPipeline(
            nli_model=nli_model,
            use_llm_judge=judge_on,
            fallback_chain=fallback_chain,
        ),
        fdr_monitor=FDRMonitor(alpha=alpha),
        online=OnlineConformal(alpha=alpha) if online else None,
    )


def _min_samples(alpha: float) -> int:
    """Smallest calibration set that can certify this alpha: ceil(1/alpha) - 1."""
    return max(1, int(-(-1.0 // alpha)) - 1)


def _judge_configured() -> bool:
    """Whether any provider key for the LLM-judge fallback chain is present."""
    return any(
        os.environ.get(name)
        for name in (
            "GROQ_API_KEY",
            "OPENROUTER_API_KEY",
            "TOGETHER_API_KEY",
            "TOGETHERAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        )
    )


def _build_domain_manager(settings: Settings) -> DomainManager:
    judge_on = settings.use_llm_judge and _judge_configured()
    return DomainManager(
        route_margin=settings.domain_route_margin,
        embedding_model=settings.embedding_model,
        nli_model=settings.nli_model,
        use_llm_judge=judge_on,
        fallback_chain=settings.llm_fallback_chain,
    )


def _build_retriever(settings: Settings) -> HybridRetriever:
    """Build the retriever; Qdrant is opt-in via CONFORMAL_RAG_QDRANT_URL."""
    return HybridRetriever(
        qdrant_url=settings.qdrant_url or None,
        collection=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
    )


def _is_qdrant_error(exc: Exception) -> bool:
    """Connection-level failures of the Qdrant HTTP client."""
    import httpx
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError))


class RateLimiter:
    """Fixed-window per-client rate limiter (in-memory, single process)."""

    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic() if now is None else now
        window = self._hits[key]
        while window and now - window[0] >= 60.0:
            window.popleft()
        if len(window) >= self.limit:
            return False
        window.append(now)
        return True


# Atomic sliding-window check: prune expired hits, count, and conditionally
# record the new hit. Runs server-side so concurrent instances stay consistent.
_REDIS_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now, member)
    redis.call('EXPIRE', key, math.ceil(window) + 1)
    return 1
end
return 0
"""


class RedisRateLimiter:
    """Sliding-window per-client rate limiter backed by Redis.

    Same interface as RateLimiter so backends are interchangeable. Shared
    state across instances makes the limit hold for multi-process deploys.
    Fails open: a Redis outage degrades to no rate limiting rather than
    taking the API down.
    """

    def __init__(
        self,
        limit_per_minute: int,
        redis_url: str,
        prefix: str = "conformal_rag:rl:",
        window: float = 60.0,
    ):
        self.limit = limit_per_minute
        self.redis_url = redis_url
        self.prefix = prefix
        self.window = window
        self._client = None
        self._script = None
        self._instance_id = uuid.uuid4().hex

    def _get_client(self):
        if self._client is None:
            import redis
            self._client = redis.Redis.from_url(
                self.redis_url, socket_connect_timeout=2, socket_timeout=2
            )
            self._script = self._client.register_script(_REDIS_SLIDING_WINDOW_LUA)
        return self._client

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        if self.limit <= 0:
            return True
        now = time.time() if now is None else now
        try:
            self._get_client()
            # Unique member per hit so repeated calls at the same timestamp
            # (and across instances) are all counted.
            member = f"{now}:{self._instance_id}:{uuid.uuid4().hex}"
            allowed = self._script(
                keys=[self.prefix + key],
                args=[now, self.window, self.limit, member],
            )
            return bool(int(allowed))
        except Exception as e:
            logger.warning("Rate limiter Redis error, failing open: %s", e)
            return True


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_frontend_path(path: str) -> bool:
    """Static frontend assets skip auth so the UI can load and show its
    own 'unauthorized' handling for API calls."""
    if path == "/":
        return True
    return path.startswith(("/css/", "/js/", "/fonts/", "/assets/", "/favicon"))


def _warm_models(app: FastAPI) -> None:
    """Load the NLI and embedding models in the background at startup.

    Model loads take 5-20s on CPU; warming them here means the first user
    request is not the one that pays for it.
    """
    try:
        app.state.engine.scoring.nli._load_model()
        logger.info("Warm-up: NLI model ready")
    except Exception as e:
        logger.warning("Warm-up: NLI model failed: %s", e)
    # The embedder load can hit a transient HF Hub lookup even when the weights
    # are cached locally; one retry is enough and keeps startup non-fatal.
    for attempt in (1, 2):
        try:
            app.state.retriever._get_embedder()
            logger.info("Warm-up: embedder ready")
            break
        except Exception as e:
            logger.warning("Warm-up: embedder attempt %d failed: %s", attempt, e)
            time.sleep(2)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    threading.Thread(target=_warm_models, args=(app,), daemon=True).start()
    yield


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create FastAPI application."""
    settings = settings or Settings.from_env()

    logging.getLogger("conformal_rag").setLevel(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    )

    app = FastAPI(
        title="Conformal RAG",
        description="Conformal Risk Control hallucination bounds for RAG systems",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.settings = settings

    app.state.engine = _build_engine(
        alpha=settings.default_alpha,
        nli_model=settings.nli_model,
        use_llm_judge=settings.use_llm_judge,
        fallback_chain=settings.llm_fallback_chain,
    )
    app.state.retriever = _build_retriever(settings)
    app.state.domain_manager = _build_domain_manager(settings)
    app.state.last_calibration = None
    # Uploaded documents are request-scoped working state: in memory only,
    # never written to the state directory, capped and evicted oldest first.
    app.state.uploads: dict[str, UploadedDocument] = {}
    app.state.uploads_lock = threading.Lock()

    # --- CORS (opt-in) --------------------------------------------------
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS enabled for origins: %s", settings.cors_origins)

    # --- rate limiting (opt-in) ------------------------------------------
    # Redis-backed when CONFORMAL_RAG_REDIS_URL is set (multi-instance safe),
    # in-memory otherwise (single process, zero-config local dev).
    if settings.redis_url:
        limiter = RedisRateLimiter(settings.rate_limit_per_minute, settings.redis_url)
        logger.info("Rate limiter backend: redis")
    else:
        limiter = RateLimiter(settings.rate_limit_per_minute)
        logger.info("Rate limiter backend: in-memory")
    app.state.rate_limiter = limiter

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if settings.rate_limit_per_minute > 0 and not _is_frontend_path(path):
            if not limiter.allow(_client_key(request)):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                )
        # OPTIONS passes through so CORS preflight isn't blocked by auth.
        if settings.api_key and request.method != "OPTIONS" \
                and path not in _AUTH_EXEMPT_PATHS \
                and not _is_frontend_path(path):
            provided = request.headers.get("x-api-key", "")
            if provided != settings.api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid X-API-Key header."},
                )
        response = await call_next(request)
        # The frontend is edited between demo runs; a cached app.js would run
        # old code against a new API. Revalidate every asset.
        if path.endswith((".js", ".css", ".html")) or path == "/":
            response.headers["Cache-Control"] = "no-cache"
        return response

    # --- persistence (opt-in) --------------------------------------------

    def _state_dir() -> Optional[Path]:
        if not settings.state_dir:
            return None
        d = Path(settings.state_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_state() -> None:
        d = _state_dir()
        if d is None:
            return
        try:
            app.state.engine.save_state(d / _ENGINE_STATE)
            app.state.retriever.save_state(d / _INDEX_STATE)
            app.state.domain_manager.save_state(d / _DOMAIN_DIR)
        except OSError as e:
            logger.warning("State save failed: %s", e)

    def _load_state() -> None:
        d = _state_dir()
        if d is None:
            return
        if app.state.engine.load_state(d / _ENGINE_STATE):
            logger.info("Restored engine calibration from %s", d / _ENGINE_STATE)
        if app.state.retriever.load_state(d / _INDEX_STATE):
            logger.info("Restored document index from %s", d / _INDEX_STATE)
        loaded = app.state.domain_manager.load_state(d / _DOMAIN_DIR)
        if loaded:
            logger.info("Restored domains: %s", ", ".join(loaded))

    _load_state()

    # --- meta ---------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/config")
    async def get_config():
        engine = app.state.engine
        return {
            "alpha": engine.alpha,
            "online": engine.online is not None,
            "calibrated": engine.calibrated,
            "documents": len(app.state.retriever._documents),
            "nli_model": settings.nli_model,
            "use_llm_judge": settings.use_llm_judge and _judge_configured(),
            "llm_judge_configured": _judge_configured(),
        }

    @app.post("/config")
    async def set_config(request: ConfigRequest):
        dm = app.state.domain_manager

        # A domain-scoped change re-thresholds only that domain, so the
        # per-domain tolerances (medicine 0.05 vs finance 0.15) survive.
        if request.domain is not None:
            if request.domain not in dm.domains:
                raise HTTPException(
                    status_code=404, detail=f"Unknown domain: {request.domain}"
                )
            domain_engine = dm.domains[request.domain]
            if not domain_engine.calibrated:
                raise HTTPException(
                    status_code=422,
                    detail=f"Domain '{request.domain}' is not calibrated yet",
                )
            domain_engine.recalibrate_from_stored(request.alpha)
            monitor = dm.monitors.get(request.domain)
            if monitor is not None:
                monitor.alpha = domain_engine.alpha
            _save_state()
            stats = domain_engine.calibration_stats()
            return {
                "domain": request.domain,
                "alpha": domain_engine.alpha,
                "calibrated": True,
                "threshold": domain_engine._threshold,
                "stats": stats.model_dump() if stats else None,
            }

        prev = app.state.engine
        prev_model = settings.nli_model
        if request.nli_model is not None:
            settings.nli_model = request.nli_model
        if request.use_llm_judge is not None:
            settings.use_llm_judge = request.use_llm_judge

        engine = _build_engine(
            alpha=request.alpha,
            online=request.online,
            nli_model=settings.nli_model,
            use_llm_judge=settings.use_llm_judge,
            fallback_chain=settings.llm_fallback_chain,
        )
        model_unchanged = settings.nli_model == prev_model
        if model_unchanged and prev.calibrated:
            # Re-threshold from the stored scores: changing alpha must not
            # throw away the calibration, and no model calls are needed.
            engine._calibration_scores = list(prev._calibration_scores)
            engine._calibration_labels = list(prev._calibration_labels)
            engine._domain = prev._domain
            engine.recalibrate_from_stored(request.alpha)

        app.state.engine = engine
        app.state.last_calibration = engine.calibration_stats()
        if not model_unchanged:
            app.state.domain_manager = _build_domain_manager(settings)
        _save_state()
        return {
            "alpha": engine.alpha,
            "online": engine.online is not None,
            "calibrated": engine.calibrated,
            "documents": len(app.state.retriever._documents),
            "nli_model": settings.nli_model,
            "use_llm_judge": settings.use_llm_judge and _judge_configured(),
            "llm_judge_configured": _judge_configured(),
        }

    @app.post("/reset")
    async def reset():
        app.state.engine = _build_engine(
            alpha=settings.default_alpha,
            nli_model=settings.nli_model,
            use_llm_judge=settings.use_llm_judge,
            fallback_chain=settings.llm_fallback_chain,
        )
        app.state.retriever = _build_retriever(settings)
        app.state.domain_manager = _build_domain_manager(settings)
        app.state.last_calibration = None
        _save_state()
        return {"status": "reset"}

    # --- retrieval ----------------------------------------------------

    @app.get("/samples")
    async def samples():
        return {"documents": SAMPLE_DOCUMENTS}

    @app.get("/demo")
    async def demo():
        """Showcase cases and domain presets for the UI's example buttons."""
        return {
            "alpha": MAIN_ALPHA,
            "showcase": SHOWCASE,
            "domains": [
                {"name": name, "alpha": spec["alpha"]}
                for name, spec in DEMO_DOMAINS.items()
            ],
        }

    @app.post("/index")
    def index(request: IndexRequest):
        lines = [d.strip() for d in request.documents if d.strip()]
        retriever = _build_retriever(settings)
        try:
            retriever.index([Document(text=line) for line in lines])
        except Exception as e:
            if _is_qdrant_error(e):
                raise HTTPException(status_code=503, detail="Vector database unavailable")
            raise
        app.state.retriever = retriever
        _save_state()
        return {"indexed": len(lines)}

    @app.get("/documents")
    async def documents():
        docs = app.state.retriever._documents
        return {"count": len(docs), "documents": docs}

    @app.post("/retrieve")
    def retrieve(request: RetrieveRequest):
        retriever = app.state.retriever
        retriever.final_k = request.final_k
        try:
            results = retriever.retrieve(request.query)
        except Exception as e:
            if _is_qdrant_error(e):
                raise HTTPException(status_code=503, detail="Vector database unavailable")
            raise
        return {"results": results}

    # --- uploaded documents --------------------------------------------

    @app.post("/documents/upload")
    async def upload_document(file: UploadFile = File(...)):
        """Take a PDF or text file, split it into passages, index it in memory.

        Returns a `doc_id` that `/ask` uses. Extraction and embedding are
        blocking, so they run off the event loop.
        """
        data = await file.read(_MAX_UPLOAD_BYTES + 1)
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"That file is larger than the {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
            )
        if not data:
            raise HTTPException(status_code=422, detail="The file is empty.")

        def build():
            parsed = extract_text(file.filename or "upload", data)
            chunks = chunk_text(parsed.text)
            if not chunks:
                raise DocumentError("No passages could be read from this file.")
            retriever = HybridRetriever(
                embedding_model=settings.embedding_model,
                final_k=_UPLOAD_FINAL_K,
            )
            retriever.index([Document(text=chunk) for chunk in chunks])
            return parsed, chunks, retriever

        try:
            parsed, chunks, retriever = await run_in_threadpool(build)
        except DocumentError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        doc_id = uuid.uuid4().hex[:12]
        upload = UploadedDocument(
            doc_id=doc_id,
            filename=parsed.filename,
            kind=parsed.kind,
            pages=parsed.pages,
            characters=len(parsed.text),
            passages=chunks,
            preview=preview(parsed.text),
            retriever=retriever,
        )
        store = app.state.uploads
        with app.state.uploads_lock:
            store[doc_id] = upload
            while len(store) > _MAX_UPLOADS:
                store.pop(next(iter(store)))

        return {
            "doc_id": doc_id,
            "filename": upload.filename,
            "kind": upload.kind,
            "pages": upload.pages,
            "characters": upload.characters,
            "passages": len(chunks),
            "preview": upload.preview,
        }

    @app.post("/ask")
    def ask(request: AskRequest):
        """Answer a question from one uploaded document, then check the answer.

        Retrieval, drafting and scoring all run in the threadpool. The verdict
        is the same `/predict` path, so the tolerance and threshold shown are
        the ones the selected topic actually carries.
        """
        upload = app.state.uploads.get(request.doc_id)
        if upload is None:
            raise HTTPException(
                status_code=404,
                detail="That document is no longer loaded. Upload it again.",
            )

        passages = upload.retriever.retrieve(request.query)
        client = optional_client(settings.use_llm_judge)
        answer, mode = generate_answer(request.query, passages, client=client)

        if mode == "none":
            # A decline on the first pass often means the answer sits just
            # outside the top passages. Read more before telling the user the
            # document does not cover the question.
            wider = upload.retriever.retrieve(request.query, k=_UPLOAD_RETRY_K)
            if len(wider) > len(passages):
                answer, mode = generate_answer(request.query, wider, client=client)
                passages = wider

        if not answer:
            return {
                "doc_id": upload.doc_id,
                "query": request.query,
                "answer": "",
                "mode": mode,
                # The closest passage is still shown, so a decline says what
                # the document does cover instead of nothing at all.
                "grounding": None,
                "passages": passages,
                "grounded_in": passages[0] if passages else "",
                "verdict": None,
                "domain": None,
                "note": (
                    "Nothing in this document states an answer to that question. "
                    "The closest passage it did find is shown below; try wording "
                    "closer to what the document says directly."
                ),
            }

        dm = app.state.domain_manager
        engine = app.state.engine
        context = "\n".join(passages) or answer
        grounding = "exact" if is_exact_quote(answer, context) else "paraphrase"
        # Uploads sit outside the demo calibration set, so auto-routing by
        # centroid would stamp a brand-new document with a topic from the demo
        # corpus (a statistics paper routes to "finance"). Use the broadest
        # calibrated domain unless the caller names one explicitly.
        target = request.domain or "general"
        target_engine = dm.domains.get(target)

        if target_engine is not None and target_engine.calibrated:
            result = dm.predict(
                query=request.query,
                context=context,
                response=answer,
                domain=target,
            )
        else:
            result = engine.predict(
                query=request.query,
                context=context,
                response=answer,
                domain=request.domain,
            )
            engine.fdr_monitor.record_prediction(abstained=result.abstained)

        return {
            "doc_id": upload.doc_id,
            "query": request.query,
            "answer": answer,
            "mode": mode,
            "grounding": grounding,
            "passages": passages,
            "grounded_in": passages[0] if passages else "",
            "verdict": result.model_dump(),
            "domain": result.domain,
            "note": _UPLOAD_CAVEAT,
        }

    # --- conformal engine ----------------------------------------------

    @app.post("/predict")
    def predict(request: PredictRequest):
        dm = app.state.domain_manager
        engine = app.state.engine
        context_str = "\n".join(request.context)

        # Route through the per-domain manager when it can apply a calibrated
        # threshold: an explicit registered domain, or auto-routing once at
        # least one registered domain is calibrated. Otherwise the main engine
        # handles it exactly as before.
        use_domain_manager = (
            request.domain is not None and request.domain in dm.domains
        ) or (
            request.domain is None
            and any(e.calibrated for e in dm.domains.values())
        )

        if use_domain_manager:
            result = dm.predict(
                query=request.query,
                context=context_str,
                response=request.response,
                domain=request.domain,
            )
        else:
            result = engine.predict(
                query=request.query,
                context=context_str,
                response=request.response,
                domain=request.domain,
            )
            # Count the abstention decision only; true labels arrive via /record.
            engine.fdr_monitor.record_prediction(abstained=result.abstained)
        return result.model_dump()

    @app.post("/calibrate")
    def calibrate(request: CalibrateRequest):
        engine = app.state.engine
        try:
            stats = engine.calibrate(
                queries=[d.query for d in request.data],
                contexts=[d.context for d in request.data],
                responses=[d.response for d in request.data],
                labels=[int(d.label) for d in request.data],
                domain=request.domain,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        app.state.last_calibration = stats
        _save_state()
        return stats.model_dump()

    @app.get("/calibration")
    async def calibration():
        stats = app.state.last_calibration or app.state.engine.calibration_stats()
        if stats is None:
            return {"calibrated": False}
        return {"calibrated": True, **stats.model_dump()}

    @app.get("/calibration/scores")
    async def calibration_scores(domain: Optional[str] = None):
        """Raw calibration scores and labels, for the score separation chart.

        The guarantee needs only aggregate statistics; the individual points
        are exposed so the UI can show where the threshold lands and how well
        faithful and hallucinated answers separate.
        """
        dm = app.state.domain_manager
        engine = dm.domains[domain] if domain and domain in dm.domains else app.state.engine
        scores = list(engine._calibration_scores)
        labels = list(engine._calibration_labels)
        return {
            "domain": domain or engine._domain,
            "alpha": engine.alpha,
            "threshold": engine._threshold,
            "calibrated": engine.calibrated,
            "points": [
                {"score": round(float(s), 4), "label": int(lbl)}
                for s, lbl in zip(scores, labels)
            ],
        }

    @app.post("/record")
    def record(request: RecordRequest):
        engine = app.state.engine
        engine.record(
            query=request.query,
            context=request.context,
            response=request.response,
            actual_hallucination=request.actual_hallucination,
        )
        return {
            "status": "recorded",
            "metrics": engine.fdr_monitor.get_metrics(),
            "online": (
                {**engine.online.get_stats(), "enabled": True}
                if engine.online is not None
                else {"enabled": False}
            ),
        }

    @app.get("/metrics")
    async def metrics():
        return app.state.engine.fdr_monitor.get_metrics()

    @app.get("/online")
    async def online():
        engine = app.state.engine
        if engine.online is None:
            return {"enabled": False}
        return {"enabled": True, **engine.online.get_stats()}

    # --- domains --------------------------------------------------------

    @app.get("/domains")
    async def list_domains():
        dm = app.state.domain_manager
        out = []
        for name, engine in dm.domains.items():
            monitor = dm.monitors.get(name)
            stats = engine.calibration_stats()
            out.append({
                "name": name,
                "alpha": engine.alpha,
                "threshold": engine._threshold,
                "calibrated": engine.calibrated,
                "n_samples": len(engine._calibration_scores),
                "min_samples_required": (
                    stats.min_samples_required if stats else _min_samples(engine.alpha)
                ),
                "certifiable": stats.certifiable if stats else False,
                "stats": stats.model_dump() if stats else None,
                "metrics": monitor.get_metrics() if monitor else FDRMonitor().get_metrics(),
                "drift": dm.monitor_drift(name).model_dump(),
            })
        return {"domains": out}

    @app.post("/domains")
    async def register_domain(request: DomainRequest):
        app.state.domain_manager.register_domain(request.name, alpha=request.alpha)
        _save_state()
        return {"status": "registered", "name": request.name}

    @app.post("/domains/{name}/calibrate")
    def calibrate_domain(name: str, request: CalibrateRequest):
        try:
            stats = app.state.domain_manager.calibrate(
                domain=name,
                queries=[d.query for d in request.data],
                contexts=[d.context for d in request.data],
                responses=[d.response for d in request.data],
                labels=[int(d.label) for d in request.data],
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        _save_state()
        return stats.model_dump()

    @app.get("/domains/{name}/drift")
    async def domain_drift(name: str):
        return app.state.domain_manager.monitor_drift(name).model_dump()

    @app.post("/feedback")
    async def feedback(request: FeedbackRequest):
        app.state.domain_manager.record(
            request.domain,
            predicted_hallucination=request.predicted_hallucination,
            actual_hallucination=request.actual_hallucination,
        )
        return {"status": "recorded"}

    # --- static frontend (mounted last so API routes take precedence) ---

    frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app
