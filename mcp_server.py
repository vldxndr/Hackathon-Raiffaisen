"""
╔══════════════════════════════════════════════════════════════════╗
║  PARTEA 3 – Server FastMCP                                       ║
║  Proiect: Estimarea Prețului Laptopurilor                        ║
║                                                                  ║
║  Tools:                                                          ║
║    - predict_laptop_price   → predicție cu un model ales          ║
║    - compare_all_models     → comparare predicții toate modelele  ║
║    - get_model_metrics      → metrici performanță modele          ║
║    - get_dataset_stats      → statistici dataset                  ║
║                                                                  ║
║  Prompts:                                                        ║
║    - interpret_price        → analiză preț estimat                ║
║    - recommend_laptop       → recomandare laptop pe buget         ║
║                                                                  ║
║  Resources:                                                      ║
║    - dataset://stats        → statistici generale                 ║
║    - models://metrics       → metrici modele                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pickle
import joblib
import pandas as pd
from fastmcp import FastMCP

# ── Inițializare server MCP ───────────────────────────────────────────────────
mcp = FastMCP("Laptop Price Predictor")

# ── Încărcare modele ──────────────────────────────────────────────────────────
MODELS = {
    "original": {
        "ridge": joblib.load("models/original_ridge_regression.pth"),
        "random_forest": joblib.load("models/original_random_forest.pth"),
        "gradient_boosting": joblib.load("models/original_gradient_boosting.pth"),
        "catboost": joblib.load("models/original_catboost.pth"),
    },
    "realistic": {
        "ridge": joblib.load("models/realistic_ridge.pth"),
        "random_forest": joblib.load("models/realistic_rf.pth"),
        "gradient_boosting": joblib.load("models/realistic_gbr.pth"),
        "catboost": joblib.load("models/realistic_catboost.pth"),
    },
}

# ── Încărcare metadata ────────────────────────────────────────────────────────
METADATA = {
    "original": joblib.load("models/original_metadata.pth"),
    "realistic": joblib.load("models/realistic_metadata.pth"),
}

MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "catboost": "CatBoost",
}

DATASET_PATHS = {
    "original": "laptop Price Prediction Dataset.csv",
    "realistic": "laptop_price_realistic.csv",
}


def _build_input(brand, processor, ram_gb, storage_gb, screen_size,
                 graphics_card, operating_system, weight_kg,
                 battery_life_hours, warranty_years) -> pd.DataFrame:
    """Construiește un DataFrame cu o singură înregistrare pentru predicție."""
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


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1: Predicție preț laptop
# ══════════════════════════════════════════════════════════════════════════════
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
    Predict the price of a laptop based on its technical specifications.

    Uses a trained ML model to estimate the market price in USD.

    Args:
        brand: Laptop manufacturer. Values: Apple, Acer, Asus, Dell, HP, Lenovo, MSI, Razer
        processor: CPU type. Values: Intel i5, Intel i7, Intel i9, AMD Ryzen 5, AMD Ryzen 7, AMD Ryzen 9
        ram_gb: RAM in GB. Values: 8, 16, 32, 64
        storage_gb: Internal storage in GB. Values: 256, 512, 1024, 2048
        screen_size_inches: Screen diagonal in inches. Values: 13.3, 14.0, 15.6, 17.0
        graphics_card: GPU. Values: Intel UHD, AMD Radeon, NVIDIA GTX 1650, NVIDIA RTX 3060, NVIDIA RTX 3070
        operating_system: OS. Values: Linux, macOS, Windows 10, Windows 11
        weight_kg: Weight in kg (e.g., 1.8)
        battery_life_hours: Battery life in hours (e.g., 8.0)
        warranty_years: Warranty period. Values: 1, 2, 3
        model: ML model to use. Values: ridge, random_forest, gradient_boosting, catboost
        dataset: Which trained model set to use. Values: original, realistic
    """
    if dataset not in MODELS:
        return {"error": f"Unknown dataset: '{dataset}'. Choose from: {list(MODELS.keys())}"}
    if model not in MODELS[dataset]:
        return {"error": f"Unknown model: '{model}'. Choose from: {list(MODELS[dataset].keys())}"}

    input_df = _build_input(
        brand, processor, ram_gb, storage_gb, screen_size_inches,
        graphics_card, operating_system, weight_kg, battery_life_hours, warranty_years,
    )

    price = float(MODELS[dataset][model].predict(input_df)[0])

    # Price category
    if price < 1000:
        category = "Budget"
    elif price < 2000:
        category = "Mid-Range"
    else:
        category = "Premium"

    return {
        "predicted_price_usd": round(price, 2),
        "price_category": category,
        "model_used": MODEL_DISPLAY_NAMES[model],
        "dataset_used": dataset,
        "currency": "USD",
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2: Comparare toate modelele
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def compare_all_models(
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
    dataset: str = "realistic",
) -> dict:
    """
    Compare price predictions from ALL available ML models for the same laptop specs.

    Returns predictions from Ridge, Random Forest, Gradient Boosting, and CatBoost,
    plus the average price and which model is closest to the ensemble average.

    Args:
        brand: Laptop manufacturer. Values: Apple, Acer, Asus, Dell, HP, Lenovo, MSI, Razer
        processor: CPU type. Values: Intel i5, Intel i7, Intel i9, AMD Ryzen 5, AMD Ryzen 7, AMD Ryzen 9
        ram_gb: RAM in GB. Values: 8, 16, 32, 64
        storage_gb: Internal storage in GB. Values: 256, 512, 1024, 2048
        screen_size_inches: Screen diagonal in inches. Values: 13.3, 14.0, 15.6, 17.0
        graphics_card: GPU. Values: Intel UHD, AMD Radeon, NVIDIA GTX 1650, NVIDIA RTX 3060, NVIDIA RTX 3070
        operating_system: OS. Values: Linux, macOS, Windows 10, Windows 11
        weight_kg: Weight in kg (e.g., 1.8)
        battery_life_hours: Battery life in hours (e.g., 8.0)
        warranty_years: Warranty period. Values: 1, 2, 3
        dataset: Which trained model set to use. Values: original, realistic
    """
    if dataset not in MODELS:
        return {"error": f"Unknown dataset: '{dataset}'. Choose from: {list(MODELS.keys())}"}

    input_df = _build_input(
        brand, processor, ram_gb, storage_gb, screen_size_inches,
        graphics_card, operating_system, weight_kg, battery_life_hours, warranty_years,
    )

    predictions = {}
    for model_key, model_obj in MODELS[dataset].items():
        try:
            price = float(model_obj.predict(input_df)[0])
            predictions[MODEL_DISPLAY_NAMES[model_key]] = round(price, 2)
        except Exception as e:
            predictions[MODEL_DISPLAY_NAMES[model_key]] = f"Error: {str(e)}"

    # Calculate ensemble average
    valid_prices = {k: v for k, v in predictions.items() if isinstance(v, float)}
    if valid_prices:
        avg = sum(valid_prices.values()) / len(valid_prices)
        best = min(valid_prices, key=lambda k: abs(valid_prices[k] - avg))
    else:
        avg = 0
        best = "N/A"

    return {
        "dataset_used": dataset,
        "predictions": predictions,
        "ensemble_average_usd": round(avg, 2),
        "most_reliable_model": best,
        "currency": "USD",
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3: Metrici modele
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def get_model_metrics(dataset: str = "realistic") -> dict:
    """
    Get performance metrics (MAE, RMSE, R²) for all ML models on a given dataset.

    Use this to understand which model performs best before making predictions.

    Args:
        dataset: Which dataset's metrics to show. Values: original, realistic
    """
    if dataset not in METADATA:
        return {"error": f"Unknown dataset: '{dataset}'. Choose from: {list(METADATA.keys())}"}

    meta = METADATA[dataset]
    metrics = meta.get("metrics", meta)

    result = {}
    for key, m in metrics.items():
        display_name = MODEL_DISPLAY_NAMES.get(key, key)
        if isinstance(m, dict):
            result[display_name] = {
                "MAE": round(m.get("MAE", 0), 2),
                "RMSE": round(float(m.get("RMSE", 0)), 2),
                "R2": round(float(m.get("R²", m.get("R2", 0))), 4),
            }

    return {
        "dataset": dataset,
        "metrics": result,
        "best_model": max(result, key=lambda k: result[k]["R2"]) if result else "N/A",
        "explanation": {
            "MAE": "Mean Absolute Error - average prediction error in USD",
            "RMSE": "Root Mean Squared Error - penalizes large errors more",
            "R2": "R-squared - 1.0 is perfect, 0 means no better than average",
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4: Statistici dataset
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def get_dataset_stats(dataset: str = "realistic") -> dict:
    """
    Get general statistics about the laptop pricing dataset.

    Returns price distribution, available brands, processors, and other categorical values.

    Args:
        dataset: Which dataset to analyze. Values: original, realistic
    """
    if dataset not in DATASET_PATHS:
        return {"error": f"Unknown dataset: '{dataset}'. Choose from: {list(DATASET_PATHS.keys())}"}

    df = pd.read_csv(DATASET_PATHS[dataset])

    return {
        "dataset": dataset,
        "total_records": len(df),
        "price_stats": {
            "min_usd": round(df["Price ($)"].min(), 2),
            "max_usd": round(df["Price ($)"].max(), 2),
            "mean_usd": round(df["Price ($)"].mean(), 2),
            "median_usd": round(df["Price ($)"].median(), 2),
            "std_usd": round(df["Price ($)"].std(), 2),
        },
        "available_values": {
            "brands": sorted(df["Brand"].unique().tolist()),
            "processors": sorted(df["Processor"].unique().tolist()),
            "graphics_cards": sorted(df["Graphics Card"].unique().tolist()),
            "operating_systems": sorted(df["Operating System"].unique().tolist()),
            "ram_options_gb": sorted(df["RAM (GB)"].unique().tolist()),
            "storage_options_gb": sorted(df["Storage (GB)"].unique().tolist()),
            "screen_sizes_inches": sorted(df["Screen Size (inches)"].unique().tolist()),
            "warranty_years": sorted(df["Warranty (years)"].unique().tolist()),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# RESOURCE: Statistici dataset
# ══════════════════════════════════════════════════════════════════════════════
@mcp.resource("dataset://stats")
def dataset_stats_resource() -> str:
    """General statistics about the laptop pricing datasets (original and realistic)."""
    lines = []
    for ds_name, path in DATASET_PATHS.items():
        df = pd.read_csv(path)
        lines.append(f"=== Dataset: {ds_name} ===")
        lines.append(f"  Records: {len(df)}")
        lines.append(f"  Price range: ${df['Price ($)'].min():.0f} – ${df['Price ($)'].max():.0f}")
        lines.append(f"  Mean price: ${df['Price ($)'].mean():.0f}")
        lines.append(f"  Brands: {', '.join(sorted(df['Brand'].unique()))}")
        lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# RESOURCE: Metrici modele
# ══════════════════════════════════════════════════════════════════════════════
@mcp.resource("models://metrics")
def model_metrics_resource() -> str:
    """Performance metrics (MAE, RMSE, R²) for all available ML models."""
    lines = []
    for ds_name, meta in METADATA.items():
        lines.append(f"=== Dataset: {ds_name} ===")
        metrics = meta.get("metrics", meta)
        for key, m in metrics.items():
            if isinstance(m, dict):
                display = MODEL_DISPLAY_NAMES.get(key, key)
                mae = round(m.get("MAE", 0), 2)
                rmse = round(float(m.get("RMSE", 0)), 2)
                r2 = round(float(m.get("R²", m.get("R2", 0))), 4)
                lines.append(f"  {display}: MAE=${mae}, RMSE=${rmse}, R²={r2}")
        lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT 1: Interpretare preț
# ══════════════════════════════════════════════════════════════════════════════
@mcp.prompt()
def interpret_price(predicted_price: str, brand: str, specs: str) -> str:
    """
    Template prompt for interpreting a predicted laptop price.
    Use after calling predict_laptop_price to get a detailed market analysis.
    """
    # Load realistic dataset stats for dynamic context
    df = pd.read_csv(DATASET_PATHS["realistic"])
    price_min = df["Price ($)"].min()
    price_max = df["Price ($)"].max()
    price_mean = df["Price ($)"].mean()
    price_std = df["Price ($)"].std()

    brand_avg = df.groupby("Brand")["Price ($)"].mean().to_dict()
    brand_price = brand_avg.get(brand, price_mean)

    return f"""You are a laptop pricing analyst and e-commerce consultant.

A Machine Learning model predicted the price of a **{brand}** laptop at **${predicted_price} USD**.

**Laptop Specifications:**
{specs}

**Market Context (from our dataset of {len(df)} laptops):**
- Overall price range: ${price_min:.0f} – ${price_max:.0f}
- Market average: ${price_mean:.0f} (std: ${price_std:.0f})
- Average price for {brand}: ${brand_price:.0f}

**Please provide a structured analysis:**

1. **Price Assessment** — Is ${predicted_price} above, below, or at market average? How does it compare to the {brand} brand average?

2. **Value Analysis** — Given the specifications, does this laptop offer good value for money? Consider RAM, storage, GPU, and processor tier.

3. **Market Positioning** — Classify as Budget (<$800), Mid-Range ($800–$1500), or Premium (>$1500). Is this positioning correct for the specs?

4. **Competitive Pricing Strategy** — Suggest an optimal e-commerce listing price considering:
   - A 5-15% retail margin
   - Competitor positioning
   - Psychological pricing (e.g., $1,499 vs $1,500)

5. **Marketing Recommendations** — Provide 2-3 compelling selling points and a suggested product title for the e-commerce listing."""


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT 2: Recomandare laptop pe buget
# ══════════════════════════════════════════════════════════════════════════════
@mcp.prompt()
def recommend_laptop(budget: str, use_case: str) -> str:
    """
    Template prompt for recommending a laptop configuration within a budget.
    Helps users find the best specs for their needs and price range.
    """
    return f"""You are a laptop purchasing advisor for an e-commerce platform.

**Customer Request:**
- Budget: ${budget} USD
- Primary use case: {use_case}

**Available Options in Our Catalog:**
- Brands: Apple, Acer, Asus, Dell, HP, Lenovo, MSI, Razer
- Processors: Intel i5, Intel i7, Intel i9, AMD Ryzen 5, AMD Ryzen 7, AMD Ryzen 9
- RAM: 8GB, 16GB, 32GB, 64GB
- Storage: 256GB, 512GB, 1024GB, 2048GB
- Screen: 13.3", 14.0", 15.6", 17.0"
- GPU: Intel UHD, AMD Radeon, NVIDIA GTX 1650, NVIDIA RTX 3060, NVIDIA RTX 3070
- OS: Linux, macOS, Windows 10, Windows 11

**Instructions:**
1. Recommend 2-3 laptop configurations that fit the budget and use case.
2. For each recommendation, use the `predict_laptop_price` tool to verify the predicted price is within budget.
3. Explain WHY each configuration suits the use case.
4. Rank them by value-for-money.
5. Mention any trade-offs (e.g., "more RAM but smaller screen")."""


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mcp.run(transport="stdio")
