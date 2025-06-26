import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
import pandas as pd

app = FastAPI(title="Credit Risk Fraud Detection API")

# Define request model matching your feature columns
class CustomerFeatures(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    ProductCategory: str
    ChannelId: str
    PricingStrategy: str

# Resolve project root and model path robustly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.joblib')

# Load model once at startup
model = joblib.load(MODEL_PATH)

# Import and instantiate preprocessor pipeline from your data_processing module
# from data_processing import preprocess_pipeline
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing import preprocess_pipeline

preprocessor = preprocess_pipeline()

@app.post("/predict")
def predict(data: CustomerFeatures):
    try:
        # Convert input data to DataFrame for pipeline
        df = pd.DataFrame([data.dict()])

        # Ensure numeric columns are floats
        df[['Recency', 'Frequency', 'Monetary']] = df[['Recency', 'Frequency', 'Monetary']].astype(float)

        # Preprocess features
        X_processed = preprocessor.transform(df)

        # Predict fraud probability
        proba = model.predict_proba(X_processed)[:, 1][0]

        return {"fraud_probability": proba}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
