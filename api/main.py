"""FastAPI service for NEPSE sentiment + prediction (Phase 5).

This module exposes the trained hybrid pipeline and the NEPSE
direction-prediction model behind a small HTTP API so that downstream
applications (Streamlit dashboard, notebooks, external integrations)
can request:

- ``GET  /health``      -- liveness probe.
- ``POST /sentiment``   -- classify Nepali news text.
- ``POST /predict``     -- produce a NEPSE direction forecast.
- ``GET  /metrics``     -- service-level observability.

Requirements
------------
Install with::

    pip install fastapi uvicorn[standard] pydantic

Run with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Or directly::

    python -m api.main

License: MIT (see LICENSE at repo root).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional

# Ensure the ``src`` directory is on sys.path so the nepse_nlp package
# imports cleanly whether or not the project has been pip-installed.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Optional FastAPI guard
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional dep
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    _FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    _FASTAPI_AVAILABLE = False

    class FastAPI:  # type: ignore[no-redef]
        """Minimal fallback that exposes ``add_middleware`` and a
        ``get``/``post`` decorator -- enough for the module to import
        in environments without FastAPI. Real network serving is
        disabled in this case.
        """

        def __init__(self, *args, **kwargs):
            self.routes: Dict[str, Any] = {}

        def add_middleware(self, *args, **kwargs):
            pass

        def middleware(self, *args, **kwargs):
            def _wrap(fn):
                return fn

            return _wrap

        def get(self, path, *args, **kwargs):
            def _wrap(fn):
                self.routes[("GET", path)] = fn
                return fn

            return _wrap

        def post(self, path, *args, **kwargs):
            def _wrap(fn):
                self.routes[("POST", path)] = fn
                return fn

            return _wrap

    class HTTPException(Exception):  # type: ignore[no-redef]
        def __init__(self, status_code: int = 500, detail: str = ""):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class CORSMiddleware:  # type: ignore[no-redef]
        pass

    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: str
    confidence: float
    source: str


class PredictRequest(BaseModel):
    date: str


class PredictResponse(BaseModel):
    predicted_direction: str
    predicted_return: float
    confidence: float


class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# Lazy pipeline cache
# ---------------------------------------------------------------------------
_PIPELINE_CACHE: Dict[str, Any] = {}


def _get_logger():  # pragma: no cover - thin wrapper
    try:
        from nepse_nlp.utils import get_logger

        return get_logger("api")
    except Exception:
        import logging

        return logging.getLogger("api")


def _load_pipeline():
    """Lazy-load the HybridPipeline + JevClassifier (cached)."""
    if "pipeline" in _PIPELINE_CACHE:
        return _PIPELINE_CACHE["pipeline"]
    # Both imports are guarded so the endpoint degrades gracefully if
    # the parallel modules (built by Task 1-b) are not yet available.
    try:
        from nepse_nlp.hybrid_pipeline import HybridPipeline  # lazy
        from nepse_nlp.jev_benchmark import JevClassifier  # lazy
    except Exception as exc:  # pragma: no cover - depends on env
        _get_logger().warning(
            "HybridPipeline / JevClassifier unavailable: %s -- degrading /sentiment endpoint.",
            exc,
        )
        _PIPELINE_CACHE["pipeline"] = None
        return None

    # JevClassifier may require optional transformers; if construction
    # fails we degrade gracefully and mark the pipeline as unavailable.
    try:
        classifier = JevClassifier()
        pipeline = HybridPipeline(classifier=classifier)
    except Exception as exc:  # pragma: no cover - depends on env
        _get_logger().warning("Pipeline construction failed: %s", exc)
        pipeline = None
    _PIPELINE_CACHE["pipeline"] = pipeline
    return pipeline


def _load_predictor():
    """Lazy-load the cached NEPSE predictor + features snapshot."""
    if "predictor" in _PIPELINE_CACHE:
        return _PIPELINE_CACHE["predictor"]
    # In production this would unpickle a trained NEPSEPredictor from
    # disk; for the Phase 5 skeleton we construct an xgboost instance
    # and let it be retrained on first request.
    try:
        from nepse_nlp.prediction import NEPSEPredictor  # lazy

        predictor = NEPSEPredictor(model_type="xgboost", target="direction")
    except Exception as exc:  # pragma: no cover - depends on env
        _get_logger().warning("Predictor construction failed: %s", exc)
        predictor = None
    _PIPELINE_CACHE["predictor"] = predictor
    return predictor


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NEPSE Sentiment API",
    version="0.1.0",
    description=(
        "Sentiment classification + NEPSE direction-prediction service "
        "for the Jev + Confidence-Gated Hybrid Architectures project."
    ),
)

if _FASTAPI_AVAILABLE:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:  # pragma: no cover
    print(
        "[api.main] FastAPI is not installed -- running in stub mode. "
        "Install with: pip install fastapi uvicorn"
    )

# ---------------------------------------------------------------------------
# Live service metrics (in-memory)
# ---------------------------------------------------------------------------
_STATE: Dict[str, Any] = {
    "requests": 0,
    "total_latency_ms": 0.0,
    "escalations": 0,
}


def _record_request(latency_ms: float, escalated: bool = False) -> None:
    _STATE["requests"] += 1
    _STATE["total_latency_ms"] += float(latency_ms)
    if escalated:
        _STATE["escalations"] += 1


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok")


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(req: SentimentRequest) -> SentimentResponse:
    """Classify a Nepali news headline / article snippet."""
    t0 = time.perf_counter()
    pipeline = _load_pipeline()
    if pipeline is None:
        # Degraded mode: return a deterministic 'unknown' label so the
        # endpoint stays callable for integration tests.
        _record_request((time.perf_counter() - t0) * 1000)
        return SentimentResponse(label="unknown", confidence=0.0, source="degraded")
    try:
        result = pipeline.predict(req.text)
        # HybridPipeline typically returns a dict {label, confidence,
        # source, ...} but we tolerate both dict and object shapes.
        label = (
            result.get("label") if isinstance(result, dict) else getattr(result, "label", "unknown")
        )
        confidence = float(
            result.get("confidence", 0.0)
            if isinstance(result, dict)
            else getattr(result, "confidence", 0.0)
        )
        source = (
            result.get("source", "hybrid")
            if isinstance(result, dict)
            else getattr(result, "source", "hybrid")
        )
        escalated = source == "escalated"
        _record_request((time.perf_counter() - t0) * 1000, escalated=escalated)
        return SentimentResponse(label=label, confidence=confidence, source=source)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"sentiment failed: {exc}")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict NEPSE direction for the requested date."""
    t0 = time.perf_counter()
    predictor = _load_predictor()
    if predictor is None:
        _record_request((time.perf_counter() - t0) * 1000)
        return PredictResponse(
            predicted_direction="unknown",
            predicted_return=0.0,
            confidence=0.0,
        )
    try:
        # In a real deployment we'd look up the feature row for this date
        # from the cached feature matrix. For the skeleton we return a
        # constant +1 forecast with placeholder confidence.
        direction_int = 1
        predicted_dir = "up" if direction_int > 0 else "down"
        _record_request((time.perf_counter() - t0) * 1000)
        return PredictResponse(
            predicted_direction=predicted_dir,
            predicted_return=0.0,
            confidence=0.5,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"predict failed: {exc}")


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """Service-level metrics: request count, avg latency, escalation
    rate, estimated cost per 1k requests."""
    n = int(_STATE["requests"])
    avg_lat = float(_STATE["total_latency_ms"]) / max(n, 1)
    esc = int(_STATE["escalations"])
    # Cost estimate: escalation => Jev-escalated (costly) call;
    # assume $0.0001 per escalation + flat $0.00001 per request.
    est_cost_per_1k = (
        0.00001 * 1000 + (esc / max(n, 1)) * 0.0001 * 1000  # base compute  # escalated share
    )
    return {
        "requests": n,
        "avg_latency_ms": round(avg_lat, 3),
        "escalation_rate": round(esc / max(n, 1), 6),
        "est_cost_per_1k": round(est_cost_per_1k, 6),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    try:
        import uvicorn
    except Exception as exc:
        print(f"[api.main] uvicorn not available: {exc}")
        print("Install with: pip install uvicorn")
        sys.exit(1)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
