
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import feature engineering functions from your pipeline
from datapipeline.bike_data_pipeline import add_time_features, add_lag_feature

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "lightgbm_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

LOGS_DIR = Path("logs")
PREDICTIONS_LOG = LOGS_DIR / "predictions.log"


def log_prediction(inputs: dict, prediction: int):
    LOGS_DIR.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "prediction": prediction,
    }
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

MODEL = None
PREPROCESSOR = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, PREPROCESSOR
    if MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
        PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
    yield


app = FastAPI(title="Bike Rental Demand API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRecord(BaseModel):
    datetime: str
    season: int
    holiday: int
    workingday: int
    weather: int
    temp: float
    atemp: float
    humidity: int
    windspeed: float


class PredictRequest(BaseModel):
    records: List[PredictRecord]
    round_to_int: Optional[bool] = True  # you requested integer output



@app.get("/logs")
def get_logs(limit: int = Query(default=50, le=500)):
    if not PREDICTIONS_LOG.exists():
        return {"entries": []}
    with open(PREDICTIONS_LOG, "r") as f:
        lines = f.readlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    return {"entries": entries[-limit:][::-1]}  # newest first


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "preprocessor_loaded": PREPROCESSOR is not None
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None or PREPROCESSOR is None:
        return {"error": "Model artifacts not loaded. Train model first.", "predictions": []}

    # Convert request -> DataFrame
    df = pd.DataFrame([r.dict() for r in req.records])

    # Feature engineering consistent with your pipeline inference logic
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = add_time_features(df)
    df = add_lag_feature(df, is_train=False)

    # Your pipeline expects X without datetime
    X = df.drop(columns=["datetime"])
    Xp = PREPROCESSOR.transform(X)

    preds = MODEL.predict(Xp)
    preds = np.maximum(preds, 0)

    if req.round_to_int:
        preds = np.round(preds).astype(int)

    for record, prediction in zip(req.records, preds.tolist()):
        log_prediction(record.dict(), prediction)

    return {"predictions": preds.tolist()}
