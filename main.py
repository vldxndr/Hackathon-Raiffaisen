"""
╔══════════════════════════════════════════════════════════════════╗
║  PARTEA 2 – Expunerea Modelului pentru Inferență (FastAPI)      ║
║  Proiect: Estimarea Prețului Laptopurilor                        ║
║                                                                  ║
║  Dataset-uri disponibile:                                        ║
║    - "original"  → dataset-ul original                           ║
║    - "realistic" → dataset-ul realistic                          ║
║                                                                  ║
║  Modele disponibile (pentru fiecare dataset):                    ║
║    - "ridge"             → Ridge Regression                      ║
║    - "random_forest"     → Random Forest                         ║
║    - "gradient_boosting" → Gradient Boosting                     ║
║    - "catboost"          → CatBoost Regressor                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── 1. IMPORTURI ──────────────────────────────────────────────────────────────
import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ── 2. CĂI FIȘIERE ───────────────────────────────────────────────────────────
MODEL_PATHS = {
    "original": {
        "ridge": "models/original_ridge_regression.pth",
        "random_forest": "models/original_random_forest.pth",
        "gradient_boosting": "models/original_gradient_boosting.pth",
        "catboost": "models/original_catboost.pth",
        "metadata": "models/original_metadata.pth",
    },
    "realistic": {
        "ridge": "models/realistic_ridge.pth",
        "random_forest": "models/realistic_rf.pth",
        "gradient_boosting": "models/realistic_gbr.pth",
        "catboost": "models/realistic_catboost.pth",
        "metadata": "models/realistic_metadata.pth",
    },
}

DATASET_PATHS = {
    "original": "laptop Price Prediction Dataset.csv",
    "realistic": "laptop_price_realistic.csv",
}


# ── 3. ENUM-URI — valori valide ──────────────────────────────────────────────
class DatasetName(str, Enum):
    original = "original"
    realistic = "realistic"


class ModelName(str, Enum):
    ridge = "ridge"
    random_forest = "random_forest"
    gradient_boosting = "gradient_boosting"
    catboost = "catboost"


MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "catboost": "CatBoost Regressor",
    # Aliases from realistic metadata keys
    "rf": "Random Forest",
    "gbr": "Gradient Boosting",
    "Ridge Regression": "Ridge Regression",
    "Random Forest": "Random Forest",
    "Gradient Boosting": "Gradient Boosting",
    "CatBoost": "CatBoost Regressor",
    "CatBoost Regressor": "CatBoost Regressor",
}


# ── 4. STATE GLOBAL ───────────────────────────────────────────────────────────
app_state: dict = {}


# ── 5. LIFESPAN — încarcă TOATE modelele la startup ──────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Pornire server - incarc modelele...")

    app_state["models"] = {}
    app_state["metadata"] = {}

    for dataset_name, paths in MODEL_PATHS.items():
        app_state["models"][dataset_name] = {}

        # Verifică existența fișierelor
        for key, path in paths.items():
            if not os.path.exists(path):
                print(f"[!] Fisierul '{path}' nu a fost gasit - skip {dataset_name}/{key}")
                continue

            if key == "metadata":
                app_state["metadata"][dataset_name] = joblib.load(path)
                print(f"  [OK] {dataset_name}/metadata incarcat")
            else:
                app_state["models"][dataset_name][key] = joblib.load(path)
                print(f"  [OK] {dataset_name}/{key} incarcat")

    total = sum(len(m) for m in app_state["models"].values())
    print(f"\n[+] Server gata! {total} modele incarcate.")

    yield

    app_state.clear()
    print("[x] Server oprit.")


# ── 6. APLICAȚIE ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Laptop Price Predictor API",
    description=(
        "Estimează prețul unui laptop pe baza specificațiilor tehnice.\n\n"
        "**Dataset-uri disponibile** (câmpul `dataset`):\n"
        "- `original` — dataset-ul original\n"
        "- `realistic` — dataset-ul realistic\n\n"
        "**Modele disponibile** (câmpul `model`):\n"
        "- `ridge` — Ridge Regression\n"
        "- `random_forest` — Random Forest\n"
        "- `gradient_boosting` — Gradient Boosting\n"
        "- `catboost` — CatBoost Regressor\n\n"
        "**Partea 2** din proiectul de internship Raiffeisen."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── 7. SCHEME PYDANTIC ───────────────────────────────────────────────────────
class LaptopInput(BaseModel):
    """Specificațiile laptopului + modelul și dataset-ul dorit pentru predicție."""

    # ── Parametri de selecție ─────────────────────────────────────────────────
    dataset: DatasetName = Field(
        default=DatasetName.original,
        description="Dataset-ul pe care a fost antrenat modelul: 'original' sau 'realistic'",
        examples=["original"],
    )
    model: ModelName = Field(
        default=ModelName.ridge,
        description="Modelul ML: 'ridge', 'random_forest', 'gradient_boosting', 'catboost'",
        examples=["catboost"],
    )

    # ── Specificațiile laptopului ─────────────────────────────────────────────
    Brand: str = Field(..., examples=["Dell"],
        description="Valori: Dell, Asus, HP, Acer, Lenovo, MSI, Apple, Razer")
    Processor: str = Field(..., examples=["Intel i7"],
        description="Valori: Intel i5, Intel i7, Intel i9, AMD Ryzen 5, AMD Ryzen 7, AMD Ryzen 9")
    RAM_GB: int = Field(..., alias="RAM (GB)", ge=4, le=128, examples=[16],
        description="Valori din dataset: 8, 16, 32, 64")
    Storage_GB: int = Field(..., alias="Storage (GB)", ge=64, le=4096, examples=[512],
        description="Valori din dataset: 256, 512, 1024, 2048")
    Screen_Size: float = Field(..., alias="Screen Size (inches)", ge=10.0, le=20.0, examples=[15.6])
    Graphics_Card: str = Field(..., alias="Graphics Card", examples=["NVIDIA RTX 3060"],
        description="Valori: Intel UHD, AMD Radeon, NVIDIA GTX 1650, NVIDIA RTX 3060, NVIDIA RTX 3070")
    Operating_System: str = Field(..., alias="Operating System", examples=["Windows 11"],
        description="Valori: Windows 10, Windows 11, macOS, Linux")
    Weight_kg: float = Field(..., alias="Weight (kg)", ge=0.5, le=6.0, examples=[1.8])
    Battery_Life: float = Field(..., alias="Battery Life (hours)", ge=1.0, le=30.0, examples=[8.0])
    Warranty_years: int = Field(..., alias="Warranty (years)", ge=1, le=5, examples=[2],
        description="Valori din dataset: 1, 2, 3")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    predicted_price_usd: float
    price_category: str
    model_used: str
    dataset_used: str
    input_received: dict


class CompareResponse(BaseModel):
    dataset_used: str
    predictions: dict
    best_model: str
    input_received: dict


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict


# ── 8. ENDPOINT-URI ───────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "message": "Laptop Price Predictor API — rulează!",
        "datasets": ["original", "realistic"],
        "models": ["ridge", "random_forest", "gradient_boosting", "catboost"],
        "endpoints": {
            "/docs": "Swagger UI",
            "/health": "status",
            "/predict": "POST predicție (un model)",
            "/compare": "POST comparare (toate modelele)",
            "/info": "GET metrici modele",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    loaded = {}
    for dataset_name, models in app_state.get("models", {}).items():
        loaded[dataset_name] = list(models.keys())

    total = sum(len(m) for m in loaded.values())
    return HealthResponse(
        status="ok" if total == 8 else "partial",
        models_loaded=loaded,
    )


@app.get("/info", tags=["Info"])
def info():
    """Metrici de performanta pentru toate modelele incarcate."""
    result = {}
    for dataset_name, meta in app_state.get("metadata", {}).items():
        if isinstance(meta, dict) and "metrics" in meta:
            metrics_data = meta["metrics"]
            models_metrics = {}
            for model_key, m in metrics_data.items():
                if isinstance(m, dict):
                    display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
                    models_metrics[display_name] = {
                        "MAE": round(m.get("MAE", 0), 2),
                        "RMSE": round(float(m.get("RMSE", 0)), 2),
                        "R2": round(float(m.get("R\u00b2", m.get("R2", 0))), 4),
                    }
            # Find best model by R2
            best_model = "N/A"
            if models_metrics:
                best_model = max(models_metrics, key=lambda k: models_metrics[k]["R2"])
            result[dataset_name] = {
                "best_model": best_model,
                "models": models_metrics,
            }
        else:
            result[dataset_name] = meta
    return {
        "datasets": list(MODEL_PATHS.keys()),
        "models_per_dataset": ["ridge", "random_forest", "gradient_boosting", "catboost"],
        "metadata": result,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predicție"])
def predict(laptop: LaptopInput):
    """
    Estimează prețul unui laptop cu un model specific.

    **Câmpul `dataset`** selectează setul de modele:
    - `"original"` → antrenat pe dataset-ul original
    - `"realistic"` → antrenat pe dataset-ul realistic

    **Câmpul `model`** selectează algoritmul:
    - `"ridge"` → Ridge Regression
    - `"random_forest"` → Random Forest
    - `"gradient_boosting"` → Gradient Boosting
    - `"catboost"` → CatBoost Regressor

    ### Exemplu:
    ```json
    {
      "dataset": "original",
      "model": "catboost",
      "Brand": "Dell",
      "Processor": "Intel i7",
      "RAM (GB)": 16,
      "Storage (GB)": 512,
      "Screen Size (inches)": 15.6,
      "Graphics Card": "NVIDIA RTX 3060",
      "Operating System": "Windows 11",
      "Weight (kg)": 1.8,
      "Battery Life (hours)": 8.0,
      "Warranty (years)": 2
    }
    ```
    """
    dataset_key = laptop.dataset.value
    model_key = laptop.model.value

    # Verifică dacă modelul e încărcat
    models = app_state.get("models", {}).get(dataset_key, {})
    if model_key not in models:
        available = list(models.keys()) if models else []
        raise HTTPException(
            status_code=503,
            detail=f"Modelul '{model_key}' din dataset-ul '{dataset_key}' nu e încărcat. "
                   f"Disponibile: {available}",
        )

    # Construim DataFrame
    input_dict = {
        "Brand": laptop.Brand,
        "Processor": laptop.Processor,
        "RAM (GB)": laptop.RAM_GB,
        "Storage (GB)": laptop.Storage_GB,
        "Screen Size (inches)": laptop.Screen_Size,
        "Graphics Card": laptop.Graphics_Card,
        "Operating System": laptop.Operating_System,
        "Weight (kg)": laptop.Weight_kg,
        "Battery Life (hours)": laptop.Battery_Life,
        "Warranty (years)": laptop.Warranty_years,
    }
    df_input = pd.DataFrame([input_dict])

    # Predicție
    try:
        predicted_price = float(models[model_key].predict(df_input)[0])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Eroare la predicție: {str(e)}")

    # Categorie de preț
    if predicted_price < 1000:
        category = "Budget"
    elif predicted_price < 2000:
        category = "Mid-Range"
    else:
        category = "Premium"

    return PredictionResponse(
        predicted_price_usd=round(predicted_price, 2),
        price_category=category,
        model_used=MODEL_DISPLAY_NAMES[model_key],
        dataset_used=dataset_key,
        input_received=input_dict,
    )


@app.post("/compare", response_model=CompareResponse, tags=["Predicție"])
def compare(laptop: LaptopInput):
    """
    Compară predicțiile TUTUROR modelelor disponibile pentru un dataset dat.

    Returnează prețul estimat de fiecare model + care model dă cel mai mare/mic preț.
    Câmpul `model` din request este ignorat — se folosesc toate modelele.
    """
    dataset_key = laptop.dataset.value
    models = app_state.get("models", {}).get(dataset_key, {})

    if not models:
        raise HTTPException(
            status_code=503,
            detail=f"Niciun model încărcat pentru dataset-ul '{dataset_key}'.",
        )

    # Construim DataFrame
    input_dict = {
        "Brand": laptop.Brand,
        "Processor": laptop.Processor,
        "RAM (GB)": laptop.RAM_GB,
        "Storage (GB)": laptop.Storage_GB,
        "Screen Size (inches)": laptop.Screen_Size,
        "Graphics Card": laptop.Graphics_Card,
        "Operating System": laptop.Operating_System,
        "Weight (kg)": laptop.Weight_kg,
        "Battery Life (hours)": laptop.Battery_Life,
        "Warranty (years)": laptop.Warranty_years,
    }
    df_input = pd.DataFrame([input_dict])

    # Predicții cu toate modelele
    predictions = {}
    for model_key, model_obj in models.items():
        try:
            price = float(model_obj.predict(df_input)[0])
            predictions[MODEL_DISPLAY_NAMES[model_key]] = round(price, 2)
        except Exception as e:
            predictions[MODEL_DISPLAY_NAMES[model_key]] = f"Eroare: {str(e)}"

    # Găsim cel mai bun model (cel mai aproape de medie)
    valid_prices = {k: v for k, v in predictions.items() if isinstance(v, float)}
    if valid_prices:
        avg = sum(valid_prices.values()) / len(valid_prices)
        best = min(valid_prices, key=lambda k: abs(valid_prices[k] - avg))
    else:
        best = "N/A"

    return CompareResponse(
        dataset_used=dataset_key,
        predictions=predictions,
        best_model=f"{best} (closest to ensemble average: ${avg:.2f})" if best != "N/A" else "N/A",
        input_received=input_dict,
    )
