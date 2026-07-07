FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml .
RUN mkdir -p src/conformal_rag && touch src/conformal_rag/__init__.py
RUN uv pip install --system --no-cache-dir -e .

COPY src/ src/
RUN uv pip install --system --no-cache-dir -e .

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HF_HOME=/tmp/hf_home

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ src/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "conformal_rag.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
