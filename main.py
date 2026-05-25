import pickle
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Literal

app = FastAPI(
    title="Laptop Price Prediction API",
    description="Estimează prețul unui laptop pe baza specificațiilor tehnice.",
    version="1.0.0",
)

with open("models/laptop_price_catboost.pth", "rb") as _f:
    _catboost = pickle.load(_f)

MODELS = {
    "ridge": joblib.load("models/ridge_model.pkl"),
    "random_forest": joblib.load("models/random_forest_model.pkl"),
    "gradient_boosting": joblib.load("models/gradient_boosting_model.pkl"),
    "catboost": _catboost,
}

MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "catboost": "CatBoost",
}

all_metrics = joblib.load("models/all_metrics.pkl")

import pickle
with open("models/laptop_price_metadata.pth", "rb") as f:
    _cb_meta = pickle.load(f)

all_metrics["catboost"] = {
    "MAE": round(_cb_meta["metrics"]["MAE"], 2),
    "RMSE": round(float(_cb_meta["metrics"]["RMSE"]), 2),
    "R2": round(float(_cb_meta["metrics"]["R²"]), 4),
}


class LaptopFeatures(BaseModel):
    Brand: str
    Processor: str
    RAM_GB: int
    Storage_GB: int
    Screen_Size_inches: float
    Graphics_Card: str
    Operating_System: str
    Weight_kg: float
    Battery_Life_hours: float
    Warranty_years: int
    model: Literal["ridge", "random_forest", "gradient_boosting", "catboost"] = "ridge"

    model_config = {
        "json_schema_extra": {
            "example": {
                "Brand": "Dell",
                "Processor": "Intel i7",
                "RAM_GB": 16,
                "Storage_GB": 512,
                "Screen_Size_inches": 15.6,
                "Graphics_Card": "NVIDIA RTX 3060",
                "Operating_System": "Windows 11",
                "Weight_kg": 1.8,
                "Battery_Life_hours": 8.0,
                "Warranty_years": 2,
                "model": "catboost",
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_price_usd: float
    model_used: str
    currency: str = "USD"


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/models")
def list_models():
    return {
        key: {
            "display_name": MODEL_DISPLAY_NAMES[key],
            "metrics": all_metrics[key],
        }
        for key in MODELS
    }


@app.get("/dataset-stats")
def dataset_stats():
    df = pd.read_csv("laptop Price Prediction Dataset.csv")
    return {
        "total_records": len(df),
        "price_stats": {
            "min": round(df["Price ($)"].min(), 2),
            "max": round(df["Price ($)"].max(), 2),
            "mean": round(df["Price ($)"].mean(), 2),
            "std": round(df["Price ($)"].std(), 2),
        },
        "brands": sorted(df["Brand"].unique().tolist()),
        "processors": sorted(df["Processor"].unique().tolist()),
        "graphics_cards": sorted(df["Graphics Card"].unique().tolist()),
        "operating_systems": sorted(df["Operating System"].unique().tolist()),
        "ram_options_gb": sorted(df["RAM (GB)"].unique().tolist()),
        "storage_options_gb": sorted(df["Storage (GB)"].unique().tolist()),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: LaptopFeatures):
    input_df = pd.DataFrame([{
        "Brand": features.Brand,
        "Processor": features.Processor,
        "RAM (GB)": features.RAM_GB,
        "Storage (GB)": features.Storage_GB,
        "Screen Size (inches)": features.Screen_Size_inches,
        "Graphics Card": features.Graphics_Card,
        "Operating System": features.Operating_System,
        "Weight (kg)": features.Weight_kg,
        "Battery Life (hours)": features.Battery_Life_hours,
        "Warranty (years)": features.Warranty_years,
    }])

    selected_model = MODELS[features.model]
    try:
        price = float(selected_model.predict(input_df)[0])
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictionResponse(
        predicted_price_usd=round(price, 2),
        model_used=MODEL_DISPLAY_NAMES[features.model],
    )
