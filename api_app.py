
from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Import feature engineering functions from your pipeline
from datapipeline.bike_data_pipeline import add_time_features, add_lag_feature

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "lightgbm_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

app = FastAPI(title="Bike Rental Demand API", version="1.0")

MODEL = None
PREPROCESSOR = None


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


@app.on_event("startup")
def load_artifacts():
    global MODEL, PREPROCESSOR
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        MODEL, PREPROCESSOR = None, None
        return
    MODEL = joblib.load(MODEL_PATH)
    PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)


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

    return {"predictions": preds.tolist()}
