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

# ── Încărcare modele ──────────────────────────────────────
MODELS = {
    "original": {
        "ridge":            joblib.load("models/original_ridge_regression.pth"),
        "random_forest":    joblib.load("models/original_random_forest.pth"),
        "gradient_boosting":joblib.load("models/original_gradient_boosting.pth"),
        "catboost":         joblib.load("models/original_catboost.pth"),
    },
    "realistic": {
        "ridge":            joblib.load("models/realistic_ridge.pth"),
        "random_forest":    joblib.load("models/realistic_rf.pth"),
        "gradient_boosting":joblib.load("models/realistic_gbr.pth"),
        "catboost":         joblib.load("models/realistic_catboost.pth"),
    },
}

MODEL_DISPLAY_NAMES = {
    "ridge":            "Ridge Regression",
    "random_forest":    "Random Forest",
    "gradient_boosting":"Gradient Boosting",
    "catboost":         "CatBoost",
}

# ── Metrici ───────────────────────────────────────────────
_meta_orig = joblib.load("models/original_metadata.pth")
_meta_real = joblib.load("models/realistic_metadata.pth")

METRICS = {
    "original": {
        "ridge":            _meta_orig["metrics"].get("Ridge Regression", {}),
        "random_forest":    _meta_orig["metrics"].get("Random Forest", {}),
        "gradient_boosting":_meta_orig["metrics"].get("Gradient Boosting", {}),
        "catboost":         _meta_orig["metrics"].get("CatBoost", {}),
    },
    "realistic": {
        "ridge":            _meta_real["metrics"].get("ridge", {}),
        "random_forest":    _meta_real["metrics"].get("rf", {}),
        "gradient_boosting":_meta_real["metrics"].get("gbr", {}),
        "catboost":         _meta_real["metrics"].get("catboost", {}),
    },
}


# ── Schemas ───────────────────────────────────────────────
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
    model:   Literal["ridge", "random_forest", "gradient_boosting", "catboost"] = "catboost"
    dataset: Literal["original", "realistic"] = "realistic"

    model_config = {
        "json_schema_extra": {
            "example": {
                "Brand": "Dell", "Processor": "Intel i7",
                "RAM_GB": 16, "Storage_GB": 512,
                "Screen_Size_inches": 15.6, "Graphics_Card": "NVIDIA RTX 3060",
                "Operating_System": "Windows 11", "Weight_kg": 1.8,
                "Battery_Life_hours": 8.0, "Warranty_years": 2,
                "model": "catboost", "dataset": "realistic",
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_price_usd: float
    model_used: str
    dataset: str
    currency: str = "USD"


# ── Endpoints ─────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/models")
def list_models():
    return {
        dataset: {
            key: {
                "display_name": MODEL_DISPLAY_NAMES[key],
                "metrics": METRICS[dataset][key],
            }
            for key in MODELS[dataset]
        }
        for dataset in MODELS
    }


@app.get("/dataset-stats")
def dataset_stats():
    df_orig = pd.read_csv("laptop Price Prediction Dataset.csv")
    df_real = pd.read_csv("laptop_price_realistic.csv")
    return {
        "original": {
            "total_records": len(df_orig),
            "price_stats": {
                "min":  round(df_orig["Price ($)"].min(), 2),
                "max":  round(df_orig["Price ($)"].max(), 2),
                "mean": round(df_orig["Price ($)"].mean(), 2),
                "std":  round(df_orig["Price ($)"].std(), 2),
            },
        },
        "realistic": {
            "total_records": len(df_real),
            "price_stats": {
                "min":  round(df_real["Price ($)"].min(), 2),
                "max":  round(df_real["Price ($)"].max(), 2),
                "mean": round(df_real["Price ($)"].mean(), 2),
                "std":  round(df_real["Price ($)"].std(), 2),
            },
        },
        "brands":           sorted(df_orig["Brand"].unique().tolist()),
        "processors":       sorted(df_orig["Processor"].unique().tolist()),
        "graphics_cards":   sorted(df_orig["Graphics Card"].unique().tolist()),
        "operating_systems":sorted(df_orig["Operating System"].unique().tolist()),
        "ram_options_gb":   sorted(df_orig["RAM (GB)"].unique().tolist()),
        "storage_options_gb":sorted(df_orig["Storage (GB)"].unique().tolist()),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: LaptopFeatures):
    input_df = pd.DataFrame([{
        "Brand":                features.Brand,
        "Processor":            features.Processor,
        "RAM (GB)":             features.RAM_GB,
        "Storage (GB)":         features.Storage_GB,
        "Screen Size (inches)": features.Screen_Size_inches,
        "Graphics Card":        features.Graphics_Card,
        "Operating System":     features.Operating_System,
        "Weight (kg)":          features.Weight_kg,
        "Battery Life (hours)": features.Battery_Life_hours,
        "Warranty (years)":     features.Warranty_years,
    }])

    selected_model = MODELS[features.dataset][features.model]
    try:
        price = float(selected_model.predict(input_df)[0])
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictionResponse(
        predicted_price_usd=round(price, 2),
        model_used=MODEL_DISPLAY_NAMES[features.model],
        dataset=features.dataset,
    )
