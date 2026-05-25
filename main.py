import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Literal, List, Optional
from dotenv import load_dotenv
import boto3
from fastmcp import Client as MCPClient

load_dotenv()

app = FastAPI(
    title="Laptop Price Prediction API",
    description="Estimează prețul unui laptop pe baza specificațiilor tehnice.",
    version="2.0.0",
)

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


# ── Schemas ───────────────────────────────────────────────────────────────────

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


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    reply: str


# ── Static & routes ───────────────────────────────────────────────────────────

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
    df = pd.read_csv("laptop Price Prediction Dataset.csv")
    price_stats = {
        "min":  round(df["Price ($)"].min(), 2),
        "max":  round(df["Price ($)"].max(), 2),
        "mean": round(df["Price ($)"].mean(), 2),
        "std":  round(df["Price ($)"].std(), 2),
    }
    return {
        "original":          {"total_records": len(df), "price_stats": price_stats},
        "realistic":         {"total_records": len(df), "price_stats": price_stats},
        "brands":            sorted(df["Brand"].unique().tolist()),
        "processors":        sorted(df["Processor"].unique().tolist()),
        "graphics_cards":    sorted(df["Graphics Card"].unique().tolist()),
        "operating_systems": sorted(df["Operating System"].unique().tolist()),
        "ram_options_gb":    sorted(df["RAM (GB)"].unique().tolist()),
        "storage_options_gb":sorted(df["Storage (GB)"].unique().tolist()),
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


# ── Chatbot ───────────────────────────────────────────────────────────────────

MCP_URL = os.getenv("MCP_URL", "http://localhost:8001/sse")

CHAT_TOOLS = [
    {
        "name": "predict_laptop_price",
        "description": (
            "Estimează prețul unui laptop pe baza specificațiilor tehnice. "
            "Folosește dataset='realistic' pentru prețuri calibrate pe piața reală."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "brand":              {"type": "string", "enum": ["Apple", "Acer", "Asus", "Dell", "HP", "Lenovo", "MSI", "Razer"]},
                "processor":          {"type": "string", "enum": ["Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 5", "AMD Ryzen 7", "AMD Ryzen 9"]},
                "ram_gb":             {"type": "integer", "enum": [8, 16, 32, 64]},
                "storage_gb":         {"type": "integer", "enum": [256, 512, 1024, 2048]},
                "screen_size_inches": {"type": "number",  "enum": [13.3, 14.0, 15.6, 17.0]},
                "graphics_card":      {"type": "string",  "enum": ["Intel UHD", "AMD Radeon", "NVIDIA GTX 1650", "NVIDIA RTX 3060", "NVIDIA RTX 3070"]},
                "operating_system":   {"type": "string",  "enum": ["Linux", "macOS", "Windows 10", "Windows 11"]},
                "weight_kg":          {"type": "number",  "description": "Greutate în kg, ex: 1.8"},
                "battery_life_hours": {"type": "number",  "description": "Autonomie baterie în ore, ex: 8.0"},
                "warranty_years":     {"type": "integer", "enum": [1, 2, 3]},
                "model":              {"type": "string",  "enum": ["ridge", "random_forest", "gradient_boosting", "catboost"]},
                "dataset":            {"type": "string",  "enum": ["original", "realistic"]},
            },
            "required": ["brand", "processor", "ram_gb", "storage_gb", "screen_size_inches",
                         "graphics_card", "operating_system", "weight_kg", "battery_life_hours", "warranty_years"],
        },
    },
    {
        "name": "search_laptops_in_budget",
        "description": (
            "Caută configurații de laptop care se încadrează într-un buget dat. "
            "Returnează top 5 opțiuni ordonate descrescător după preț."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "budget_usd": {"type": "number", "description": "Bugetul maxim în USD, ex: 1200"},
                "dataset":    {"type": "string", "enum": ["original", "realistic"]},
                "model":      {"type": "string", "enum": ["ridge", "random_forest", "gradient_boosting", "catboost"]},
            },
            "required": ["budget_usd"],
        },
    },
]

SYSTEM_PROMPT = """Ești un asistent specializat în laptopuri pentru Raiffeisen Bank. Ajuți utilizatorii să găsească laptopul potrivit bugetului și nevoilor lor.
Ai acces la un model ML antrenat pe două seturi de date: 'realistic' (prețuri calibrate pe piața reală) și 'original' (datele brute).
Implicit folosește 'realistic' și modelul 'catboost' pentru recomandări. Răspunde în română, concis și util.
Când prezinți mai multe opțiuni, folosește un tabel markdown clar cu coloane: Brand, Procesor, RAM, GPU, Preț estimat."""


async def _call_mcp_tool(name: str, inputs: dict) -> str:
    """Apelează un tool din mcp_server.py via FastMCP Client (HTTP/SSE)."""
    try:
        async with MCPClient(MCP_URL) as client:
            result = await client.call_tool(name, inputs)
            if isinstance(result, list) and result:
                return getattr(result[0], "text", str(result[0]))
            return str(result)
    except Exception as e:
        return json.dumps({"error": f"MCP server indisponibil: {str(e)}"})


def _call_bedrock(messages: list, system: str) -> dict:
    token    = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
    model_id = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-opus-4-6-v1")
    region   = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

    if not token or token == "pune_token_ul_aici":
        raise HTTPException(status_code=503, detail="Token Bedrock lipsă. Completează AWS_BEARER_TOKEN_BEDROCK în fișierul .env")

    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = token
    client = boto3.client("bedrock-runtime", region_name=region)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "system": system,
        "messages": messages,
        "tools": CHAT_TOOLS,
    })
    resp = client.invoke_model(modelId=model_id, body=body, contentType="application/json", accept="application/json")
    return json.loads(resp["body"].read())


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in (req.history or [])]
    messages.append({"role": "user", "content": req.message})

    try:
        for _ in range(8):
            data        = _call_bedrock(messages, SYSTEM_PROMPT)
            stop_reason = data.get("stop_reason")
            content     = data.get("content", [])

            text = next((b["text"] for b in content if b.get("type") == "text"), "")

            if stop_reason in ("end_turn", "max_tokens"):
                return ChatResponse(reply=text)

            if stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": content})
                tool_results = []
                for block in content:
                    if block.get("type") == "tool_use":
                        # apelează mcp_server.py via HTTP, nu local
                        result = await _call_mcp_tool(block["name"], block["input"])
                        tool_results.append({"type": "tool_result", "tool_use_id": block["id"], "content": result})
                messages.append({"role": "user", "content": tool_results})
                continue

            if text:
                return ChatResponse(reply=text)

        return ChatResponse(reply="Nu am putut genera un răspuns complet. Încearcă să reformulezi întrebarea.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare: {str(e)}")