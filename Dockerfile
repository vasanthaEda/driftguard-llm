FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /data

EXPOSE 8000

# DRIFTGUARD_EMBEDDER_BACKEND/JUDGE_BACKEND/LLM_BACKEND default to the
# offline-safe deterministic backends (see app/config.py); set them to
# "openai" + OPENAI_API_KEY at deploy time to use real embeddings/LLM.
ENV DRIFTGUARD_DB_PATH=/data/driftguard.sqlite3

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3)" || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
