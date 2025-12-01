from fastapi import FastAPI
from pydantic import BaseModel
from app.metrics import REQUEST_COUNT, REQUEST_LATENCY
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import joblib
import time
from fastapi.responses import Response

app = FastAPI()

# Load trained model
model = joblib.load("models/baseline.joblib")
MODEL_VERSION = "v1.0"


# --- Request/Response Models ---
class PredictRequest(BaseModel):
    x1: float
    x2: float


class PredictResponse(BaseModel):
    score: float
    model_version: str


# --- Endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(data: PredictRequest):
    start = time.time()
    REQUEST_COUNT.inc()  # increment counter

    # Convert input to 2D list for sklearn
    features = [[data.x1, data.x2]]
    score = float(model.predict_proba(features)[0][1])

    REQUEST_LATENCY.observe(time.time() - start)

    return {
        "score": score,
        "model_version": MODEL_VERSION,
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
