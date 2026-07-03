FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY . .

EXPOSE 8000

CMD ["uvicorn", "conformal_rag.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
