import pickle
import pickle
import joblib
import pandas as pd
from fastmcp import FastMCP

mcp = FastMCP("Laptop Price Predictor")

# ── încărcare modele ──────────────────────────────────────────────────────────
with open("models/laptop_price_catboost.pth", "rb") as _f:
    _catboost = pickle.load(_f)

MODELS = {
    "ridge": joblib.load("models/ridge_model.pkl"),
    "random_forest": joblib.load("models/random_forest_model.pkl"),
    "gradient_boosting": joblib.load("models/gradient_boosting_model.pkl"),
    "catboost": _catboost,
}

with open("models/all_metrics.pkl", "rb") as f:
    ALL_METRICS = pickle.load(f)

with open("models/laptop_price_metadata.pth", "rb") as f:
    _cb_meta = pickle.load(f)

ALL_METRICS["catboost"] = {
    "MAE": round(_cb_meta["metrics"]["MAE"], 2),
    "RMSE": round(float(_cb_meta["metrics"]["RMSE"]), 2),
    "R2": round(float(_cb_meta["metrics"]["R²"]), 4),
}

MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "catboost": "CatBoost",
}


def _build_input(brand, processor, ram_gb, storage_gb, screen_size,
                 graphics_card, operating_system, weight_kg,
                 battery_life_hours, warranty_years) -> pd.DataFrame:
    return pd.DataFrame([{
        "Brand": brand,
        "Processor": processor,
        "RAM (GB)": ram_gb,
        "Storage (GB)": storage_gb,
        "Screen Size (inches)": screen_size,
        "Graphics Card": graphics_card,
        "Operating System": operating_system,
        "Weight (kg)": weight_kg,
        "Battery Life (hours)": battery_life_hours,
        "Warranty (years)": warranty_years,
    }])


# ── TOOL: predicție preț ─────────────────────────────────────────────────────
@mcp.tool()
def predict_laptop_price(
    brand: str,
    processor: str,
    ram_gb: int,
    storage_gb: int,
    screen_size_inches: float,
    graphics_card: str,
    operating_system: str,
    weight_kg: float,
    battery_life_hours: float,
    warranty_years: int,
    model: str = "catboost",
) -> dict:
    """
    Estimează prețul unui laptop pe baza specificațiilor tehnice.

    Args:
        brand: Producătorul laptopului. Valori: Apple, Acer, Asus, Dell, HP, Lenovo, MSI, Razer
        processor: Tipul de procesor. Valori: Intel i5, Intel i7, Intel i9, AMD Ryzen 5, AMD Ryzen 7, AMD Ryzen 9
        ram_gb: Memorie RAM în GB. Valori: 8, 16, 32, 64
        storage_gb: Stocare internă în GB. Valori: 256, 512, 1024, 2048
        screen_size_inches: Diagonala ecranului în inch. Valori: 13.3, 14.0, 15.6, 17.0
        graphics_card: Placa video. Valori: Intel UHD, AMD Radeon, NVIDIA GTX 1650, NVIDIA RTX 3060, NVIDIA RTX 3070
        operating_system: Sistemul de operare. Valori: Linux, macOS, Windows 10, Windows 11
        weight_kg: Greutatea în kg (ex: 1.8)
        battery_life_hours: Autonomia bateriei în ore (ex: 8.0)
        warranty_years: Perioada de garanție. Valori: 1, 2, 3
        model: Modelul ML de folosit. Valori: ridge, random_forest, gradient_boosting, catboost
    """
    if model not in MODELS:
        return {"error": f"Model necunoscut: '{model}'. Alege din: {list(MODELS.keys())}"}

    input_df = _build_input(
        brand, processor, ram_gb, storage_gb, screen_size_inches,
        graphics_card, operating_system, weight_kg, battery_life_hours, warranty_years,
    )

    price = float(MODELS[model].predict(input_df)[0])
    return {
        "predicted_price_usd": round(price, 2),
        "model_used": MODEL_DISPLAY_NAMES[model],
        "currency": "USD",
    }


# ── RESOURCE: statistici dataset ─────────────────────────────────────────────
@mcp.resource("dataset://stats")
def dataset_stats() -> str:
    """Statistici generale despre dataset-ul de laptopuri: prețuri, branduri, procesoare disponibile."""
    df = pd.read_csv("laptop Price Prediction Dataset.csv")
    stats = {
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
    return str(stats)


# ── RESOURCE: metrici modele ──────────────────────────────────────────────────
@mcp.resource("models://metrics")
def model_metrics() -> str:
    """Metricile de performanță (MAE, RMSE, R²) pentru fiecare model ML disponibil."""
    lines = []
    for key, metrics in ALL_METRICS.items():
        lines.append(f"{MODEL_DISPLAY_NAMES[key]}: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, R²={metrics['R2']}")
    return "\n".join(lines)


# ── PROMPT: interpretare preț ─────────────────────────────────────────────────
@mcp.prompt()
def interpret_price(predicted_price: str, brand: str, specs: str) -> str:
    """Template de prompt pentru interpretarea unui preț estimat de model."""
    return f"""Modelul ML a estimat prețul laptopului {brand} la {predicted_price} USD.

Specificații: {specs}

Statistici piață din dataset:
- Preț minim: $501.53
- Preț maxim: $2999.68
- Preț mediu: $1780.33

Analizează dacă acest preț este corect față de medie, dacă laptopul oferă valoare bună pentru bani,
și sugerează cum ar putea fi promovat pe o platformă de e-commerce."""


if __name__ == "__main__":
    mcp.run(transport="stdio")