"""
Generează laptop_price_realistic.csv cu prețuri realiste bazate pe specificații,
antrenează Ridge / Random Forest / Gradient Boosting / CatBoost și salvează
câte un .pth separat pentru fiecare model.
"""

import numpy as np
import pandas as pd
import joblib, os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

# ─────────────────────────────────────────────
# 1. ÎNCĂRCARE DATE ORIGINALE
# ─────────────────────────────────────────────
df = pd.read_csv("laptop Price Prediction Dataset.csv")

# ─────────────────────────────────────────────
# 2. FORMULĂ PREȚURI REALISTE
# ─────────────────────────────────────────────
BRAND_BASE = {
    "Apple":  1500,
    "Razer":  1300,
    "MSI":    1100,
    "Dell":    900,
    "HP":      850,
    "Asus":    800,
    "Lenovo":  750,
    "Acer":    650,
}

PROCESSOR_PREMIUM = {
    "Intel i9":    400,
    "AMD Ryzen 9": 300,
    "Intel i7":    250,
    "AMD Ryzen 7": 200,
    "AMD Ryzen 5": 100,
    "Intel i5":     50,
}

GPU_PREMIUM = {
    "NVIDIA RTX 3070": 500,
    "NVIDIA RTX 3060": 350,
    "NVIDIA GTX 1650": 150,
    "AMD Radeon":      100,
    "Intel UHD":         0,
}

np.random.seed(42)

price = (
    df["Brand"].map(BRAND_BASE)
    + df["Processor"].map(PROCESSOR_PREMIUM)
    + df["RAM (GB)"] * 8
    + df["Storage (GB)"] * 0.15
    + df["Graphics Card"].map(GPU_PREMIUM)
    + (df["Screen Size (inches)"] - 13) * 20
    - (df["Weight (kg)"] - 1.0) * 40
    + df["Battery Life (hours)"] * 15
    + df["Warranty (years)"] * 75
    + np.random.normal(0, 150, len(df))
)

price = price.clip(400, 4500).round(2)

df_realistic = df.copy()
df_realistic["Price ($)"] = price

df_realistic.to_csv("laptop_price_realistic.csv", index=False)
print(f"CSV salvat: laptop_price_realistic.csv")
print(f"  Preț min: ${price.min():.0f}  |  max: ${price.max():.0f}  |  medie: ${price.mean():.0f}")

# ─────────────────────────────────────────────
# 3. PREGĂTIRE DATE PENTRU ANTRENARE
# ─────────────────────────────────────────────
df_realistic = df_realistic.drop(columns=["Model"])

GPU_ORDER = [["Intel UHD", "AMD Radeon", "NVIDIA GTX 1650", "NVIDIA RTX 3060", "NVIDIA RTX 3070"]]
cat_ohe   = ["Brand", "Processor", "Operating System"]
cat_ord   = ["Graphics Card"]
num_cols  = ["RAM (GB)", "Storage (GB)", "Screen Size (inches)",
             "Weight (kg)", "Battery Life (hours)", "Warranty (years)"]
target    = "Price ($)"

X = df_realistic.drop(columns=[target])
y = df_realistic[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ("num",     StandardScaler(),                                             num_cols),
    ("cat_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False),  cat_ohe),
    ("cat_ord", OrdinalEncoder(categories=GPU_ORDER),                         cat_ord),
])

# ─────────────────────────────────────────────
# 4. ANTRENARE + SALVARE MODELE
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

sklearn_models = {
    "ridge":    Ridge(alpha=1.0),
    "rf":       RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "gbr":      GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=5, random_state=42),
}

print("\n── Modele antrenate pe date REALISTE ──")
for key, model in sklearn_models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    path = f"models/realistic_{key}.pth"
    joblib.dump(pipe, path)
    print(f"  {key.upper():<10s}  MAE={mae:6.0f}$  RMSE={rmse:6.0f}$  R²={r2:.4f}  →  {path}")

# CatBoost
cat_features = ["Brand", "Processor", "Graphics Card", "Operating System"]
cb = CatBoostRegressor(
    iterations=500, learning_rate=0.05, depth=6,
    loss_function="RMSE", cat_features=cat_features,
    random_seed=42, verbose=0, allow_writing_files=False,
)
cb.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
y_pred_cb = cb.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred_cb)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_cb))
r2   = r2_score(y_test, y_pred_cb)

joblib.dump(cb, "models/realistic_catboost.pth")
print(f"  {'CATBOOST':<10s}  MAE={mae:6.0f}$  RMSE={rmse:6.0f}$  R²={r2:.4f}  →  models/realistic_catboost.pth")

# ─────────────────────────────────────────────
# 5. VERIFICARE PREȚURI PER BRAND
# ─────────────────────────────────────────────
print("\n── Preț mediu per Brand (dataset realist) ──")
print(df_realistic.groupby("Brand")["Price ($)"].mean().sort_values(ascending=False).round(0).to_string())
