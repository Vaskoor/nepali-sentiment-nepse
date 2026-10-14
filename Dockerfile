# =============================================================================
# Nepali Sentiment NEPSE — Dockerfile
# Multi-stage: builder (heavy ML deps) -> slim runtime
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System build deps for numpy / lxml / torch CPU wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src

# Install into a virtualenv for clean copy into runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    NEPSE_ENV=production

LABEL org.opencontainers.image.title="nepali-sentiment-nepse" \
      org.opencontainers.image.description="Nepali financial sentiment analysis for NEPSE stock prediction (Jev + confidence-gated hybrid)" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/your-username/nepali-sentiment-nepse"

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy project
COPY --from=builder /build/src ./src
COPY api ./api
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY config ./config
COPY Makefile README.md ./

# Pre-create data dirs
RUN mkdir -p data/raw data/processed results models mlruns

# Healthcheck: hit the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).status==200 else 1)" || exit 1

EXPOSE 8000 8501

# Default: run the API. Override with `docker run ... streamlit run dashboard/app.py`
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
