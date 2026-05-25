"""
╔══════════════════════════════════════════════════════════════════╗
║  PARTEA 2 – Expunerea Modelului pentru Inferență (FastAPI)      ║
║  Proiect: Estimarea Prețului Laptopurilor                        ║
║                                                                  ║
║  Ce face acest fișier:                                           ║
║  - Încarcă modelul salvat în Partea 1 (Ridge Regression)         ║
║  - Expune 3 endpoint-uri HTTP:                                   ║
║      GET  /health  → verifică dacă serverul + modelul sunt ok    ║
║      GET  /info    → afișează metadate despre model              ║
║      POST /predict → primește specs laptop → returnează preț     ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── 1. IMPORTURI ──────────────────────────────────────────────────────────────
import os                             # pentru verificarea existenței fișierelor
import joblib                         # pentru încărcarea modelului sklearn
import pandas as pd                   # pentru construirea DataFrame-ului de input
from contextlib import asynccontextmanager  # pentru lifespan (startup/shutdown)

from fastapi import FastAPI, HTTPException  # framework-ul API
from fastapi.middleware.cors import CORSMiddleware  # permite request-uri din browser
from pydantic import BaseModel, Field       # validarea automată a request body


# ── 2. CĂI FIȘIERE ───────────────────────────────────────────────────────────
# Aceste fișiere au fost generate de train.py (Partea 1)
# Asigură-te că sunt în același folder cu main.py SAU ajustează calea
MODEL_PATH    = "models/laptop_price_model.pth"     # ← adăugat models/
METADATA_PATH = "models/laptop_price_metadata.pth"  # ← adăugat models/


# ── 3. STATE GLOBAL ───────────────────────────────────────────────────────────
# Stocăm modelul și metadatele în memorie după ce sunt încărcate o singură dată.
# Evităm să încărcăm de pe disk la fiecare request (lent și ineficient).
app_state: dict = {}


# ── 4. LIFESPAN (startup / shutdown) ─────────────────────────────────────────
# FastAPI apelează această funcție AUTOMAT la pornire și la oprire.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP: se execută ÎNAINTE să înceapă să primească request-uri ──
    print("⏳ Pornire server — încarc modelul...")

    # Verificăm că fișierele există înainte să încercăm să le deschidem
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Fișierul '{MODEL_PATH}' nu a fost găsit!\n"
            "Rulează mai întâi notebook-ul Partea 1 pentru a antrena și salva modelul."
        )
    if not os.path.exists(METADATA_PATH):
        raise RuntimeError(
            f"Fișierul '{METADATA_PATH}' nu a fost găsit!\n"
            "Rulează mai întâi notebook-ul Partea 1."
        )

    # Încărcăm pipeline-ul sklearn (include preprocesarea + modelul Ridge)
    app_state["model"]    = joblib.load(MODEL_PATH)
    app_state["metadata"] = joblib.load(METADATA_PATH)

    print(f"✅ Model încărcat: {app_state['metadata']['model_name']}")
    print(f"   MAE={app_state['metadata']['metrics']['MAE']:.2f}  "
          f"RMSE={app_state['metadata']['metrics']['RMSE']:.2f}  "
          f"R²={app_state['metadata']['metrics']['R²']:.4f}")

    yield  # ← serverul rulează între yield și ce urmează după

    # ── SHUTDOWN: se execută când serverul este oprit ──
    app_state.clear()
    print("🛑 Server oprit — resurse eliberate.")


# ── 5. INIȚIALIZARE APLICAȚIE ─────────────────────────────────────────────────
app = FastAPI(
    title       = "Laptop Price Predictor API",
    description = (
        "API pentru estimarea prețului unui laptop pe baza specificațiilor tehnice.\n\n"
        "**Partea 2** din proiectul de internship — expune modelul Ridge Regression "
        "antrenat în Partea 1 ca serviciu HTTP cu documentație interactivă."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — permite request-uri din orice origine (util pentru Swagger UI și testare)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── 6. SCHEME PYDANTIC (structura request/response) ───────────────────────────
# Pydantic validează AUTOMAT tipurile și valorile.
# Dacă cineva trimite "RAM (GB)": "abc", FastAPI returnează 422 Unprocessable Entity.

class LaptopInput(BaseModel):
    """
    Specificațiile unui laptop pentru care vrem să estimăm prețul.
    Câmpurile și valorile posibile sunt exact cele din dataset.
    """
    # alias= permite ca JSON-ul să folosească numele original cu spații și paranteze
    # (ex: "RAM (GB)": 16) în loc de RAM_GB: 16
    Brand: str = Field(
        ...,
        example     = "Dell",
        description = "Producătorul. Valori posibile: Dell, Asus, HP, Acer, Lenovo, MSI, Apple, Razer"
    )
    Processor: str = Field(
        ...,
        example     = "Intel i7",
        description = "Procesorul. Valori: Intel i5, Intel i7, Intel i9, AMD Ryzen 5, AMD Ryzen 7, AMD Ryzen 9"
    )
    RAM_GB: int = Field(
        ...,
        alias       = "RAM (GB)",
        ge          = 4,      # ge = greater or equal (minim 4 GB)
        le          = 128,    # le = less or equal (maxim 128 GB)
        example     = 16,
        description = "Memorie RAM în GB. Valori din dataset: 8, 16, 32, 64"
    )
    Storage_GB: int = Field(
        ...,
        alias       = "Storage (GB)",
        ge          = 64,
        le          = 4096,
        example     = 512,
        description = "Capacitate stocare în GB. Valori din dataset: 256, 512, 1024, 2048"
    )
    Screen_Size: float = Field(
        ...,
        alias       = "Screen Size (inches)",
        ge          = 10.0,
        le          = 20.0,
        example     = 15.6,
        description = "Diagonala ecranului în inch"
    )
    Graphics_Card: str = Field(
        ...,
        alias       = "Graphics Card",
        example     = "NVIDIA RTX 3060",
        description = "Placa video. Valori: Intel UHD, AMD Radeon, NVIDIA GTX 1650, NVIDIA RTX 3060, NVIDIA RTX 3070"
    )
    Operating_System: str = Field(
        ...,
        alias       = "Operating System",
        example     = "Windows 11",
        description = "Sistemul de operare. Valori: Windows 10, Windows 11, macOS, Linux"
    )
    Weight_kg: float = Field(
        ...,
        alias       = "Weight (kg)",
        ge          = 0.5,
        le          = 6.0,
        example     = 1.8,
        description = "Greutatea laptopului în kilograme"
    )
    Battery_Life: float = Field(
        ...,
        alias       = "Battery Life (hours)",
        ge          = 1.0,
        le          = 30.0,
        example     = 8.0,
        description = "Autonomia bateriei în ore"
    )
    Warranty_years: int = Field(
        ...,
        alias       = "Warranty (years)",
        ge          = 1,
        le          = 5,
        example     = 2,
        description = "Garanția în ani. Valori din dataset: 1, 2, 3"
    )

    # Această setare permite folosirea atât a numelui cu alias ("RAM (GB)")
    # cât și a numelui Python (RAM_GB) în request body
    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    """Răspunsul returnat de endpoint-ul /predict"""
    predicted_price_usd : float  # prețul estimat, rotunjit la 2 zecimale
    price_category      : str    # Budget / Mid-Range / Premium
    model_used          : str    # numele modelului care a făcut predicția
    input_received      : dict   # ecoul datelor de intrare, pentru verificare


class HealthResponse(BaseModel):
    """Răspunsul endpoint-ului /health"""
    status       : str   # "ok" sau "error"
    model_loaded : bool  # True dacă modelul e în memorie
    model_name   : str   # ex: "Ridge Regression"


# ── 7. ENDPOINT-URI ───────────────────────────────────────────────────────────

# ── GET / ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    """
    Endpoint de bază — confirmă că serverul rulează și indică celelalte rute.
    """
    return {
        "message"     : "Laptop Price Predictor API — rulează!",
        "endpoints"   : {
            "documentatie_interactiva" : "/docs",
            "health_check"             : "/health",
            "info_model"               : "/info",
            "predictie"                : "POST /predict",
        }
    }


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """
    Verifică dacă serverul și modelul sunt operaționale.
    Util pentru monitoring și pentru Docker health checks.
    """
    model_loaded = "model" in app_state
    return HealthResponse(
        status       = "ok" if model_loaded else "error",
        model_loaded = model_loaded,
        model_name   = app_state.get("metadata", {}).get("model_name", "N/A"),
    )


# ── GET /info ─────────────────────────────────────────────────────────────────
@app.get("/info", tags=["Info"])
def info():
    """
    Returnează metadate despre modelul antrenat:
    - numele modelului
    - metricile de performanță (MAE, RMSE, R²)
    - coloanele folosite la antrenare
    """
    if "metadata" not in app_state:
        raise HTTPException(status_code=503, detail="Metadatele nu sunt disponibile.")

    meta = app_state["metadata"]
    return {
        "model_name"   : meta["model_name"],
        "target_column": meta["target"],
        "metrics"      : {
            "MAE"  : round(meta["metrics"]["MAE"],  2),
            "RMSE" : round(float(meta["metrics"]["RMSE"]), 2),
            "R2"   : round(float(meta["metrics"]["R²"]),   4),
        },
        "features": {
            "numerice"          : meta["num_cols"],
            "categoriale_OHE"   : meta["cat_ohe"],
            "categoriale_Ordinal": meta["cat_ord"],
        },
        "nota": (
            "Dataset-ul este sintetic (prețuri generate aleatoriu), "
            "deci R² este aproape de 0. Arhitectura codului este identică "
            "cu cea pentru un dataset real."
        )
    }


# ── POST /predict ─────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Predicție"])
def predict(laptop: LaptopInput):
    """
    **Endpoint principal** — estimează prețul unui laptop.

    **Request body:** JSON cu specificațiile laptopului (vezi schema de mai jos).

    **Response:** prețul estimat în USD + categoria de preț.

    ### Exemplu request:
    ```json
    {
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
    # Verificăm că modelul e încărcat (protecție extra față de lifespan)
    if "model" not in app_state:
        raise HTTPException(
            status_code = 503,
            detail      = "Modelul nu este încărcat. Repornește serverul."
        )

    # ── Construim DataFrame-ul cu EXACT coloanele din Partea 1 ──────────────
    # Folosim numele originale ale coloanelor (cu spații și paranteze),
    # exact cum a fost antrenat modelul în notebook.
    input_dict = {
        "Brand"                 : laptop.Brand,
        "Processor"             : laptop.Processor,
        "RAM (GB)"              : laptop.RAM_GB,
        "Storage (GB)"          : laptop.Storage_GB,
        "Screen Size (inches)"  : laptop.Screen_Size,
        "Graphics Card"         : laptop.Graphics_Card,
        "Operating System"      : laptop.Operating_System,
        "Weight (kg)"           : laptop.Weight_kg,
        "Battery Life (hours)"  : laptop.Battery_Life,
        "Warranty (years)"      : laptop.Warranty_years,
    }

    df_input = pd.DataFrame([input_dict])

    # ── Predicție ────────────────────────────────────────────────────────────
    try:
        predicted_price = float(app_state["model"].predict(df_input)[0])
    except Exception as e:
        # Dacă pipeline-ul aruncă o eroare (ex: valoare necunoscută), o returnăm clar
        raise HTTPException(
            status_code = 422,
            detail      = f"Eroare la predicție: {str(e)}"
        )

    # ── Categorie de preț (bazată pe distribuția din dataset) ────────────────
    if predicted_price < 1000:
        category = "Budget"
    elif predicted_price < 2000:
        category = "Mid-Range"
    else:
        category = "Premium"

    # ── Răspuns ──────────────────────────────────────────────────────────────
    return PredictionResponse(
        predicted_price_usd = round(predicted_price, 2),
        price_category      = category,
        model_used          = app_state["metadata"]["model_name"],
        input_received      = input_dict,
    )