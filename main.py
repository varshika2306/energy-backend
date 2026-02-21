from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
import os

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load models safely
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

solar_model = joblib.load(os.path.join(BASE_DIR, "models", "solar_model.pkl"))
demand_model = joblib.load(os.path.join(BASE_DIR, "models", "demand_model.pkl"))

# -----------------------------
# EXACT FEATURE ORDER (VERY IMPORTANT)
# -----------------------------

SOLAR_FEATURES = [
    'AMBIENT_TEMPERATURE',
    'MODULE_TEMPERATURE',
    'IRRADIATION',
    'hour',
    'day',
    'month'
]

DEMAND_FEATURES = [
    'hour',
    'dayofweek',
    'month',
    'year',
    'day'
]

CARBON_FACTOR = 0.82  # kg CO2 per kWh


# -----------------------------
# Input Schema
# -----------------------------

class EnergyInput(BaseModel):
    temperature: float
    irradiation: float
    hour: int
    day: int
    month: int
    battery_capacity: float
    current_battery_level: float


# -----------------------------
# Health Check
# -----------------------------

@app.get("/")
def root():
    return {"message": "AI Carbon-Aware Energy Backend is running 🚀"}


# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/predict")
def predict_energy(data: EnergyInput):

    try:
        # -------- Solar Prediction --------

        solar_input = pd.DataFrame([[
            data.temperature,
            data.temperature + 5,
            data.irradiation,
            data.hour,
            data.day,
            data.month
        ]], columns=SOLAR_FEATURES)

        predicted_solar = float(solar_model.predict(solar_input)[0])

        # -------- Demand Prediction --------

        now = datetime.now()

        demand_input = pd.DataFrame([[
            data.hour,
            now.weekday(),
            data.month,
            now.year,
            data.day
        ]], columns=DEMAND_FEATURES)

        predicted_demand = float(demand_model.predict(demand_input)[0])

        # -------- Gap Calculation --------

        gap = predicted_demand - predicted_solar

        # -------- Carbon Emission --------

        carbon_emission = max(gap, 0) * CARBON_FACTOR

        # -------- Sustainability Score --------

        if predicted_demand > 0:
            sustainability_score = min(
                max((predicted_solar / predicted_demand) * 100, 0), 100
            )
        else:
            sustainability_score = 0

        # -------- Battery Simulation --------

        new_battery_level = data.current_battery_level

        if gap < 0:
            # Surplus → charge battery
            new_battery_level += abs(gap)
        else:
            # Deficit → discharge battery
            new_battery_level -= gap

        new_battery_level = max(0, min(new_battery_level, data.battery_capacity))

        # -------- Recommendation --------

        if gap > 0:
            recommendation = "Shift heavy appliances to high solar hours."
        else:
            recommendation = "Store excess energy in battery."

        return {
            "predicted_solar": round(predicted_solar, 2),
            "predicted_demand": round(predicted_demand, 2),
            "gap": round(gap, 2),
            "carbon_emission": round(carbon_emission, 2),
            "sustainability_score": round(sustainability_score, 2),
            "battery_level": round(new_battery_level, 2),
            "recommendation": recommendation
        }

    except Exception as e:
        return {"error": str(e)}
