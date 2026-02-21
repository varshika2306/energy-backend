from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import random
import os

# -----------------------------
# App Initialization
# -----------------------------

app = FastAPI(title="AI Carbon Aware Backend")

# -----------------------------
# CORS (Allow frontend access)
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Input Schema
# -----------------------------

class EnergyInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    hour: int


# -----------------------------
# Health Check Route
# -----------------------------

@app.get("/")
def home():
    return {"message": "AI Carbon Aware Backend Running Successfully 🚀"}


# -----------------------------
# ML Prediction Endpoint
# -----------------------------

@app.post("/predict")
def predict(data: EnergyInput):
    # Simulated ML Logic (Temporary Deployment Version)
    solar_prediction = round(random.uniform(200, 400), 2)
    demand_prediction = round(random.uniform(300, 600), 2)

    carbon_saved = round(
        (solar_prediction * 0.2) - (demand_prediction * 0.05),
        2
    )

    return {
        "solar_prediction": solar_prediction,
        "demand_prediction": demand_prediction,
        "carbon_saved": carbon_saved,
        "status": "success"
    }
