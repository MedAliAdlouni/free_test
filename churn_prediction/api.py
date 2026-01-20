"""FastAPI backend for churn prediction."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pickle
from pathlib import Path
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from churn_prediction.models import Preprocessor

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Load models
MODEL_DIR = Path(__file__).parent / 'saved'
models = {}
preprocessor = None


def load_models():
    """Load trained models."""
    global models, preprocessor
    try:
        with open(MODEL_DIR / 'preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open(MODEL_DIR / 'baseline.pkl', 'rb') as f:
            models['baseline'] = pickle.load(f)
        with open(MODEL_DIR / 'xgboost.pkl', 'rb') as f:
            models['xgboost'] = pickle.load(f)
    except FileNotFoundError:
        pass


@app.on_event("startup")
async def startup():
    load_models()


class PredictionRequest(BaseModel):
    acquisition_channel: str
    fiber_or_adsl: str
    has_retention: bool
    offer: str
    sub_offer: str
    recruit_year_month: str
    total_bill: float
    cancel_year_month: Optional[str] = None
    duration_month: Optional[str] = None


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    model_used: str


@app.get("/")
def root():
    return {"message": "Churn Prediction API", "models_loaded": len(models) > 0}


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(models) > 0}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, model: str = "xgboost"):
    """Predict churn for a customer."""
    if not models or preprocessor is None:
        raise HTTPException(503, "Models not loaded. Train models first.")
    if model not in models:
        raise HTTPException(400, f"Model '{model}' not available.")
    
    # Prepare input
    df = pd.DataFrame([{
        'acquisition_channel': request.acquisition_channel,
        'fiber_or_adsl': request.fiber_or_adsl,
        'has_retention': request.has_retention,
        'offer': request.offer,
        'sub_offer': request.sub_offer,
        'recruit_year_month': request.recruit_year_month,
        'total_bill': request.total_bill,
        'cancel_year_month': request.cancel_year_month or 'N/A',
        'duration_month': request.duration_month or 'N/A'
    }])
    
    # Predict
    X, _ = preprocessor.transform(df)
    m = models[model]
    pred = m.predict(X)[0]
    proba = m.predict_proba(X)[0][1]
    
    return PredictionResponse(
        churn_probability=float(proba),
        churn_prediction=int(pred),
        model_used=model
    )


@app.post("/predict/batch")
def predict_batch(requests: list[PredictionRequest], model: str = "xgboost"):
    """Batch prediction."""
    if not models or preprocessor is None:
        raise HTTPException(503, "Models not loaded.")
    if model not in models:
        raise HTTPException(400, f"Model '{model}' not available.")
    
    df = pd.DataFrame([{
        'acquisition_channel': r.acquisition_channel,
        'fiber_or_adsl': r.fiber_or_adsl,
        'has_retention': r.has_retention,
        'offer': r.offer,
        'sub_offer': r.sub_offer,
        'recruit_year_month': r.recruit_year_month,
        'total_bill': r.total_bill,
        'cancel_year_month': r.cancel_year_month or 'N/A',
        'duration_month': r.duration_month or 'N/A'
    } for r in requests])
    
    X, _ = preprocessor.transform(df)
    m = models[model]
    preds = m.predict(X)
    probas = m.predict_proba(X)[:, 1]
    
    return {
        "predictions": [
            {"churn_probability": float(p), "churn_prediction": int(pr)}
            for p, pr in zip(probas, preds)
        ]
    }

