
from pathlib import Path
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"

# Create the FastAPI application
app = FastAPI(
    title="Campus Food Demand Forecasting API",
    description="Predict food demand using a trained Random Forest model.",
    version="1.0.0",
)


# Define the expected input data
class DemandInput(BaseModel):
    center_id: str
    meal_id: str
    week_index: int = Field(ge=0)
    checkout_price: float = Field(gt=0)
    base_price: float = Field(gt=0)
    emailer_for_promotion: int = Field(ge=0, le=1)
    homepage_featured: int = Field(ge=0, le=1)
    discount_amount: float
    discount_percentage: float


class DemandOutput(BaseModel):
    predicted_orders: float


# Load the model once when the API starts
model = None


@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run the training script first."
        )

    model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Campus Food Demand Forecasting API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=DemandOutput)
def predict_demand(data: DemandInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    input_df = pd.DataFrame([data.model_dump()])

    try:
        prediction = model.predict(input_df)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return {"predicted_orders": max(0.0, float(prediction))}
