from __future__ import annotations

import os
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.model_service import ModelService

app = FastAPI(
    title="MVTec Bottle Defect Detection API",
    version="1.0.0",
    description="Anomaly detection using the notebook's CNN Autoencoder and ResNet18 feature-distance models.",
)
# Configure allowed origins explicitly in production, e.g. via CORS_ORIGINS.
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

service = ModelService()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@app.get("/health")
def health():
    return {
        "status": "ok" if service.ready else "not_ready",
        "models_loaded": service.ready,
        "device": str(service.device),
        "detail": service.load_error,
    }


@app.get("/models")
def models():
    return {
        "available": ["autoencoder", "resnet", "both"],
        "thresholds_loaded": bool(service.thresholds),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("both"),
):
    if model not in {"both", "autoencoder", "resnet"}:
        raise HTTPException(status_code=422, detail="model must be 'both', 'autoencoder', or 'resnet'")
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPG, or WEBP image.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image upload exceeds the 10 MB limit.")
    try:
        return service.predict(data, model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Do not return internal stack traces to API clients.
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(exc).__name__}") from exc
