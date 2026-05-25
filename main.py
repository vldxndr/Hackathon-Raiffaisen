"""
╔══════════════════════════════════════════════════════════════════╗
║  PARTEA 2 – Expunerea Modelului pentru Inferență (FastAPI)      ║
║  Proiect: Estimarea Prețului Laptopurilor                        ║
║                                                                  ║
║  Modele disponibile (parametrul "model" din request):           ║
║    - "ridge"    → Ridge Regression  (sklearn Pipeline)          ║
║    - "catboost" → CatBoost Regressor                            ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── 1. IMPORTURI ──────────────────────────────────────────────────────────────
import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from enum import Enum                               # pentru lista de modele valide

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ── 2. CĂI FIȘIERE ───────────────────────────────────────────────────────────
RIDGE_PATH    = "models/laptop_price_model.pth"      # sklearn Pipeline (Ridge)
CATBOOST_PATH = "models/laptop_price_catboost.pth"   # CatBoost direct
METADATA_PATH = "models/laptop_price_metadata.pth"   # metrici + coloane


# ── 3. ENUM — modele valide ───────────────────────────────────────────────────
# Enum = listă fixă de valori acceptate.
# Dacă cineva trimite "model": "ceva_invalid", FastAPI returnează 422 automat.
class ModelName(str, Enum):
    ridge    = "ridge"
    catboost = "catboost"


# ── 4. STATE GLOBAL ───────────────────────────────────────────────────────────
app_state: dict = {}


# ── 5. LIFESPAN — încarcă TOATE modelele la startup ──────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Pornire server — încarc modelele...")

    for path, label in [(RIDGE_PATH, "Ridge"), (CATBOOST_PATH, "CatBoost"), (METADATA_PATH, "Metadata")]:
        if not os.path.exists(path):
            raise RuntimeError(f"Fișierul '{path}' nu a fost găsit!")

    # Ridge — sklearn Pipeline complet (preprocesare + model)
    app_state["ridge"]    = joblib.load(RIDGE_PATH)

    # CatBoost — model direct, fără pipeline sklearn
    # CatBoost tratează nativ coloanele categoriale (nu are nevoie de OHE)
    app_state["catboost"] = joblib.load(CATBOOST_PATH)

    # Metadata (metrici, coloane) — folosite de /info
    app_state["metadata"] = joblib.load(METADATA_PATH)

    print("✅ Ridge Regression încărcat")
    print("✅ CatBoost Regressor încărcat")
    print("🚀 Server gata!")

    yield

    app_state.clear()
    print("🛑 Server oprit.")


# ── 6. APLICAȚIE ─────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Laptop Price Predictor API",
    description = (
        "Estimează prețul unui laptop pe baza specificațiilor tehnice.\n\n"
        "**Modele disponibile** (câmpul `model` din request body):\n"
        "- `ridge` — Ridge Regression (sklearn)\n"
        "- `catboost` — CatBoost Regressor\n\n"
        "**Partea 2** din proiectul de internship Raiffeisen."
    ),
    version  = "1.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── 7. SCHEME PYDANTIC ───────────────────────────────────────────────────────
class LaptopInput(BaseModel):
    """Specificațiile laptopului + modelul dorit pentru predicție."""

    # ── Parametrul NOU: alegerea modelului ──────────────────────────────────
    model: ModelName = Field(
        default     = ModelName.ridge,   # dacă nu specifici, folosește Ridge
        description = "Modelul de ML folosit pentru predicție: 'ridge' sau 'catboost'",
        example     = "ridge",
    )

    # ── Specificațiile laptopului (neschimbate) ──────────────────────────────
    Brand: str = Field(..., example="Dell",
        description="Valori: Dell, Asus, HP, Acer, Lenovo, MSI, Apple, Razer")
    Processor: str = Field(..., example="Intel i7",
        description="Valori: Intel i5, Intel i7, Intel i9, AMD Ryzen 5, AMD Ryzen 7, AMD Ryzen 9")
    RAM_GB: int = Field(..., alias="RAM (GB)", ge=4, le=128, example=16,
        description="Valori din dataset: 8, 16, 32, 64")
    Storage_GB: int = Field(..., alias="Storage (GB)", ge=64, le=4096, example=512,
        description="Valori din dataset: 256, 512, 1024, 2048")
    Screen_Size: float = Field(..., alias="Screen Size (inches)", ge=10.0, le=20.0, example=15.6)
    Graphics_Card: str = Field(..., alias="Graphics Card", example="NVIDIA RTX 3060",
        description="Valori: Intel UHD, AMD Radeon, NVIDIA GTX 1650, NVIDIA RTX 3060, NVIDIA RTX 3070")
    Operating_System: str = Field(..., alias="Operating System", example="Windows 11",
        description="Valori: Windows 10, Windows 11, macOS, Linux")
    Weight_kg: float = Field(..., alias="Weight (kg)", ge=0.5, le=6.0, example=1.8)
    Battery_Life: float = Field(..., alias="Battery Life (hours)", ge=1.0, le=30.0, example=8.0)
    Warranty_years: int = Field(..., alias="Warranty (years)", ge=1, le=5, example=2,
        description="Valori din dataset: 1, 2, 3")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    predicted_price_usd : float
    price_category      : str
    model_used          : str   # îți spune exact ce model a fost folosit
    input_received      : dict


class HealthResponse(BaseModel):
    status          : str
    models_loaded   : list[str]  # lista modelelor încărcate în memorie


# ── 8. ENDPOINT-URI ───────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "message"  : "Laptop Price Predictor API — rulează!",
        "modele"   : ["ridge", "catboost"],
        "endpoints": {"/docs": "Swagger UI", "/health": "status", "/predict": "POST predicție"},
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    loaded = [name for name in ["ridge", "catboost"] if name in app_state]
    return HealthResponse(
        status        = "ok" if len(loaded) == 2 else "partial",
        models_loaded = loaded,
    )


@app.get("/info", tags=["Info"])
def info():
    """Metrici și coloane pentru modelul Ridge (cel din metadata)."""
    if "metadata" not in app_state:
        raise HTTPException(status_code=503, detail="Metadatele nu sunt disponibile.")
    meta = app_state["metadata"]
    return {
        "modele_disponibile": ["ridge", "catboost"],
        "metadata_ridge": {
            "model_name" : meta["model_name"],
            "metrics"    : {
                "MAE" : round(meta["metrics"]["MAE"], 2),
                "RMSE": round(float(meta["metrics"]["RMSE"]), 2),
                "R2"  : round(float(meta["metrics"]["R²"]), 4),
            },
            "features": {
                "numerice"           : meta["num_cols"],
                "categoriale_OHE"    : meta["cat_ohe"],
                "categoriale_Ordinal": meta["cat_ord"],
            },
        },
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predicție"])
def predict(laptop: LaptopInput):
    """
    Estimează prețul unui laptop.

    **Câmpul `model`** controlează ce algoritm face predicția:
    - `"ridge"` → Ridge Regression cu sklearn pipeline
    - `"catboost"` → CatBoost Regressor (tratează nativ categoricele)

    ### Exemplu cu Ridge:
    ```json
    {
      "model": "ridge",
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
    # ── Construim DataFrame cu coloanele originale ───────────────────────────
    input_dict = {
        "Brand"                : laptop.Brand,
        "Processor"            : laptop.Processor,
        "RAM (GB)"             : laptop.RAM_GB,
        "Storage (GB)"         : laptop.Storage_GB,
        "Screen Size (inches)" : laptop.Screen_Size,
        "Graphics Card"        : laptop.Graphics_Card,
        "Operating System"     : laptop.Operating_System,
        "Weight (kg)"          : laptop.Weight_kg,
        "Battery Life (hours)" : laptop.Battery_Life,
        "Warranty (years)"     : laptop.Warranty_years,
    }
    df_input = pd.DataFrame([input_dict])

    # ── Predicție cu modelul ales ────────────────────────────────────────────
    try:
        if laptop.model == ModelName.ridge:
            # Ridge: sklearn Pipeline → .predict() standard
            if "ridge" not in app_state:
                raise HTTPException(status_code=503, detail="Modelul Ridge nu e încărcat.")
            predicted_price = float(app_state["ridge"].predict(df_input)[0])
            model_label = "Ridge Regression"

        elif laptop.model == ModelName.catboost:
            # CatBoost: model direct → .predict() pe DataFrame
            # Nu are nevoie de preprocesare manuală — tratează categoricele singur
            if "catboost" not in app_state:
                raise HTTPException(status_code=503, detail="Modelul CatBoost nu e încărcat.")
            predicted_price = float(app_state["catboost"].predict(df_input)[0])
            model_label = "CatBoost Regressor"

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Eroare la predicție: {str(e)}")

    # ── Categorie de preț ────────────────────────────────────────────────────
    if predicted_price < 1000:
        category = "Budget"
    elif predicted_price < 2000:
        category = "Mid-Range"
    else:
        category = "Premium"

    return PredictionResponse(
        predicted_price_usd = round(predicted_price, 2),
        price_category      = category,
        model_used          = model_label,
        input_received      = input_dict,
    )