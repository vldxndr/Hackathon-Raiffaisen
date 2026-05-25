import joblib
import pandas as pd
from fastmcp import FastMCP

mcp = FastMCP("Laptop Price Predictor")

MODELS = {
    "original": {
        "ridge":             joblib.load("models/original_ridge_regression.pth"),
        "random_forest":     joblib.load("models/original_random_forest.pth"),
        "gradient_boosting": joblib.load("models/original_gradient_boosting.pth"),
        "catboost":          joblib.load("models/original_catboost.pth"),
    },
    "realistic": {
        "ridge":             joblib.load("models/realistic_ridge.pth"),
        "random_forest":     joblib.load("models/realistic_rf.pth"),
        "gradient_boosting": joblib.load("models/realistic_gbr.pth"),
        "catboost":          joblib.load("models/realistic_catboost.pth"),
    },
}

MODEL_DISPLAY_NAMES = {
    "ridge":             "Ridge Regression",
    "random_forest":     "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "catboost":          "CatBoost",
}

_meta_orig = joblib.load("models/original_metadata.pth")
_meta_real = joblib.load("models/realistic_metadata.pth")

METRICS = {
    "original": {
        "ridge":             _meta_orig["metrics"].get("Ridge Regression", {}),
        "random_forest":     _meta_orig["metrics"].get("Random Forest", {}),
        "gradient_boosting": _meta_orig["metrics"].get("Gradient Boosting", {}),
        "catboost":          _meta_orig["metrics"].get("CatBoost", {}),
    },
    "realistic": {
        "ridge":             _meta_real["metrics"].get("ridge", {}),
        "random_forest":     _meta_real["metrics"].get("rf", {}),
        "gradient_boosting": _meta_real["metrics"].get("gbr", {}),
        "catboost":          _meta_real["metrics"].get("catboost", {}),
    },
}


def _build_df(brand, processor, ram_gb, storage_gb, screen_size,
              graphics_card, operating_system, weight_kg,
              battery_life_hours, warranty_years) -> pd.DataFrame:
    return pd.DataFrame([{
        "Brand":                brand,
        "Processor":            processor,
        "RAM (GB)":             ram_gb,
        "Storage (GB)":         storage_gb,
        "Screen Size (inches)": screen_size,
        "Graphics Card":        graphics_card,
        "Operating System":     operating_system,
        "Weight (kg)":          weight_kg,
        "Battery Life (hours)": battery_life_hours,
        "Warranty (years)":     warranty_years,
    }])


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
    dataset: str = "realistic",
) -> dict:
    """
    Estimează prețul unui laptop pe baza specificațiilor tehnice.

    Args:
        brand: Producătorul. Valori: Apple, Acer, Asus, Dell, HP, Lenovo, MSI, Razer
        processor: Tipul de procesor. Valori: Intel i5, Intel i7, Intel i9, AMD Ryzen 5, AMD Ryzen 7, AMD Ryzen 9
        ram_gb: RAM în GB. Valori: 8, 16, 32, 64
        storage_gb: Stocare în GB. Valori: 256, 512, 1024, 2048
        screen_size_inches: Diagonala în inch. Valori: 13.3, 14.0, 15.6, 17.0
        graphics_card: Placa video. Valori: Intel UHD, AMD Radeon, NVIDIA GTX 1650, NVIDIA RTX 3060, NVIDIA RTX 3070
        operating_system: SO. Valori: Linux, macOS, Windows 10, Windows 11
        weight_kg: Greutate în kg (ex: 1.8)
        battery_life_hours: Autonomie în ore (ex: 8.0)
        warranty_years: Garanție în ani. Valori: 1, 2, 3
        model: Model ML. Valori: ridge, random_forest, gradient_boosting, catboost
        dataset: Set de date. Valori: original (date brute), realistic (prețuri calibrate pe piața reală)
    """
    if dataset not in MODELS:
        return {"error": f"Dataset necunoscut: '{dataset}'. Alege 'original' sau 'realistic'."}
    if model not in MODELS[dataset]:
        return {"error": f"Model necunoscut: '{model}'. Alege din: {list(MODELS[dataset].keys())}"}

    input_df = _build_df(
        brand, processor, ram_gb, storage_gb, screen_size_inches,
        graphics_card, operating_system, weight_kg, battery_life_hours, warranty_years,
    )
    price = float(MODELS[dataset][model].predict(input_df)[0])
    return {
        "predicted_price_usd": round(price, 2),
        "model_used": MODEL_DISPLAY_NAMES[model],
        "dataset": dataset,
        "currency": "USD",
    }


@mcp.tool()
def search_laptops_in_budget(
    budget_usd: float,
    dataset: str = "realistic",
    model: str = "catboost",
) -> dict:
    """
    Caută configurații de laptop care se încadrează într-un buget dat.
    Testează combinații de specificații și returnează top 5 opțiuni ordonate după preț.

    Args:
        budget_usd: Bugetul maxim în USD (ex: 1200)
        dataset: Set de date: 'realistic' sau 'original'
        model: Model ML folosit pentru estimare: ridge, random_forest, gradient_boosting, catboost
    """
    if dataset not in MODELS or model not in MODELS[dataset]:
        return {"error": "Dataset sau model invalid."}

    brands     = ["Apple", "Acer", "Asus", "Dell", "HP", "Lenovo", "MSI", "Razer"]
    processors = ["Intel i5", "Intel i7", "AMD Ryzen 5", "AMD Ryzen 7"]
    gpus       = ["Intel UHD", "AMD Radeon", "NVIDIA GTX 1650", "NVIDIA RTX 3060", "NVIDIA RTX 3070"]
    ram_opts   = [8, 16, 32]
    storage_opts = [256, 512, 1024]

    results = []
    for brand in brands:
        for processor in processors:
            for gpu in gpus:
                for ram in ram_opts:
                    for storage in storage_opts:
                        row = _build_df(brand, processor, ram, storage,
                                        15.6, gpu, "Windows 11", 1.8, 8.0, 2)
                        price = float(MODELS[dataset][model].predict(row)[0])
                        if price <= budget_usd:
                            results.append({
                                "brand": brand,
                                "processor": processor,
                                "ram_gb": ram,
                                "storage_gb": storage,
                                "graphics_card": gpu,
                                "estimated_price_usd": round(price, 2),
                            })

    results.sort(key=lambda x: x["estimated_price_usd"], reverse=True)
    top5 = results[:5]

    return {
        "budget_usd": budget_usd,
        "dataset": dataset,
        "model_used": MODEL_DISPLAY_NAMES[model],
        "options_found": len(results),
        "top_5": top5,
    }


@mcp.resource("dataset://stats")
def dataset_stats() -> str:
    """Statistici despre dataset-ul de laptopuri: prețuri, branduri, procesoare disponibile."""
    df = pd.read_csv("laptop Price Prediction Dataset.csv")
    df_r = pd.read_csv("laptop_price_realistic.csv")
    stats = {
        "original": {
            "total_records": len(df),
            "price_min":  round(df["Price ($)"].min(), 2),
            "price_max":  round(df["Price ($)"].max(), 2),
            "price_mean": round(df["Price ($)"].mean(), 2),
        },
        "realistic": {
            "total_records": len(df_r),
            "price_min":  round(df_r["Price ($)"].min(), 2),
            "price_max":  round(df_r["Price ($)"].max(), 2),
            "price_mean": round(df_r["Price ($)"].mean(), 2),
        },
        "brands":            sorted(df["Brand"].unique().tolist()),
        "processors":        sorted(df["Processor"].unique().tolist()),
        "graphics_cards":    sorted(df["Graphics Card"].unique().tolist()),
        "operating_systems": sorted(df["Operating System"].unique().tolist()),
    }
    import json
    return json.dumps(stats, indent=2)


@mcp.resource("models://metrics")
def model_metrics() -> str:
    """Metricile de performanță (MAE, RMSE, R²) pentru fiecare model ML pe ambele dataset-uri."""
    import json
    out = {}
    for ds in ("original", "realistic"):
        out[ds] = {}
        for key in MODELS[ds]:
            m = METRICS[ds][key]
            out[ds][MODEL_DISPLAY_NAMES[key]] = {
                "MAE":  round(float(m.get("MAE", 0)), 2),
                "RMSE": round(float(m.get("RMSE", 0)), 2),
                "R2":   round(float(m.get("R²", m.get("R2", 0))), 4),
            }
    return json.dumps(out, indent=2)


@mcp.prompt()
def interpret_price(predicted_price: str, brand: str, specs: str) -> str:
    """Template de prompt pentru interpretarea unui preț estimat de model."""
    return (
        f"Modelul ML a estimat prețul laptopului {brand} la {predicted_price} USD.\n\n"
        f"Specificații: {specs}\n\n"
        "Statistici piață (dataset realist):\n"
        "- Preț minim: ~$600\n"
        "- Preț maxim: ~$4500\n"
        "- Preț mediu: ~$1500\n\n"
        "Analizează dacă prețul este competitiv față de medie, "
        "dacă laptopul oferă valoare bună pentru bani, "
        "și sugerează cum ar putea fi promovat pe o platformă de e-commerce."
    )


if __name__ == "__main__":
    import os
    port = int(os.getenv("MCP_PORT", "8001"))
    mcp.run(transport="sse", host="0.0.0.0", port=port)
