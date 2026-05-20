import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Any, Dict

from backend.model_utils import train_all_models, predict_one, STATIC_PLOTS_DIR
app = FastAPI(title="Insurance Claim API")

# Allow Streamlit frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved PNG plots
os.makedirs(STATIC_PLOTS_DIR, exist_ok=True)
app.mount("/plots", StaticFiles(directory=STATIC_PLOTS_DIR), name="plots")


class PredictRequest(BaseModel):
    age: Any
    gender: Any
    bmi: Any
    children: Any
    smoker: Any
    region: Any
    charges: Any


@app.get("/")
def root():
    return {"message": "FastAPI backend is running", "plots_endpoint": "/plots"}


@app.post("/train")
def train():
    try:
        meta = train_all_models()
        return meta
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = predict_one(req.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))