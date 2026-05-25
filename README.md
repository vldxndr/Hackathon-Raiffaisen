# 🎯 Laptop Price Predictor — Proiect Hackathon Raiffeisen

## Arhitectura Proiectului

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT MCP (Claude Desktop)                    │
│         "Analizează un laptop Dell i7 16GB RTX 3060"            │
└──────────────────────────────┬──────────────────────────────────┘
                               │ MCP Protocol (stdio)
┌──────────────────────────────▼──────────────────────────────────┐
│                    SERVER FastMCP (mcp_server.py)                 │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │   4 Tools   │  │   2 Resources    │  │    2 Prompts      │  │
│  │  predict    │  │  dataset://stats │  │  interpret_price  │  │
│  │  compare    │  │  models://metrics│  │  recommend_laptop │  │
│  │  metrics    │  │                  │  │                   │  │
│  │  stats      │  │                  │  │                   │  │
│  └──────┬──────┘  └────────┬─────────┘  └───────────────────┘  │
│         │                  │                                     │
│  ┌──────▼──────────────────▼─────────────────────────────────┐  │
│  │              ML Models (joblib/pickle)                      │  │
│  │  ┌──────────────────┐    ┌──────────────────────────────┐ │  │
│  │  │  Original Set    │    │  Realistic Set               │ │  │
│  │  │  • Ridge         │    │  • Ridge        R²=0.8726    │ │  │
│  │  │  • Random Forest │    │  • Random Forest R²=0.8160   │ │  │
│  │  │  • Grad. Boost.  │    │  • Grad. Boost.  R²=0.8595  │ │  │
│  │  │  • CatBoost      │    │  • CatBoost      R²=0.8744  │ │  │
│  │  └──────────────────┘    └──────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    FastAPI Server (main.py)                       │
│  POST /predict  │  POST /compare  │  GET /info  │  GET /health  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Cerințe Acoperite

| Cerință | Status | Fișier |
|---------|--------|--------|
| **Partea 1** — Modelarea Datelor (ML) | ✅ | `part1_regresie.ipynb` |
| **Partea 2** — FastAPI pentru Inferență | ✅ | `main.py` |
| **Partea 3** — Server FastMCP | ✅ | `mcp_server.py` |
| **Partea 4** — Integrare Client MCP | ✅ | Claude Desktop / Cursor |

---

## 🧠 Partea 1: Machine Learning — Detalii Tehnice

### Problema aleasă
**Opțiunea B: Regresie** — Estimarea prețului de piață al unui laptop pe baza specificațiilor tehnice.

### Dataset
- **3000 înregistrări**, 12 coloane
- **Target:** `Price ($)` — valoare numerică continuă
- **Features:** Brand, Processor, RAM, Storage, Screen Size, Graphics Card, OS, Weight, Battery Life, Warranty

### Preprocesare (sklearn Pipeline)
```
ColumnTransformer:
├── Numerice (StandardScaler): Weight, Battery Life, Screen Size
├── Categoriale OHE (OneHotEncoder): Brand, Processor, OS
└── Categoriale Ordinal (OrdinalEncoder): Graphics Card
    (ordine: Intel UHD < AMD Radeon < GTX 1650 < RTX 3060 < RTX 3070)
```

### Modele Antrenate

| Model | Tip | Hiperparametri | Avantaje |
|-------|-----|----------------|----------|
| **Ridge Regression** | Liniar regularizat | α=1.0 | Baseline simplu, interpretabil |
| **Random Forest** | Ansamblu (Bagging) | 200 arbori | Robust la outliers, nu overfittează ușor |
| **Gradient Boosting** | Ansamblu (Boosting) | 300 iterații, lr=0.05, depth=5 | Captează relații neliniare |
| **CatBoost** | Gradient Boosting | 500 iterații, lr=0.05, depth=6 | Tratează nativ categoricele, cel mai performant |

### Rezultate — Dataset Realistic (cel relevant)

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Ridge Regression | $124 | $156 | 0.8726 |
| Random Forest | $150 | $187 | 0.8160 |
| Gradient Boosting | $128 | $163 | 0.8595 |
| **CatBoost** ✔ | **$122** | **$155** | **0.8744** |

> **CatBoost** este cel mai bun model cu R²=0.87 — explică 87% din variația prețului.

### Observație importantă despre dataset-ul original
Dataset-ul original are prețuri generate aleatoriu (R² ≈ 0, corelații < 0.02). Am creat un **dataset realist** unde prețul corelează logic cu specificațiile (RAM mai mare → preț mai mare, GPU mai bun → preț mai mare). Ambele seturi de modele sunt disponibile pentru comparație.

### Validare
- **Train/Test Split:** 80/20, random_state=42
- **Cross-Validation:** 5-Fold KFold pe dataset-ul realist
- **Metrici:** MAE, RMSE, R² pe test set

---

## 🚀 Partea 2: FastAPI — Endpoint-uri

### Rulare
```bash
cd Hackathon-Raiffaisen
uvicorn main:app --reload --port 8000
```

### Endpoint-uri disponibile

| Metoda | Endpoint | Descriere |
|--------|----------|-----------|
| GET | `/` | Info general + lista endpoint-uri |
| GET | `/health` | Status server + modele încărcate |
| GET | `/info` | Metrici performanță modele |
| POST | `/predict` | Predicție preț cu un model ales |
| POST | `/compare` | Comparare predicții toate modelele |

### Exemplu Request `/predict`
```json
{
  "dataset": "realistic",
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

### Exemplu Response
```json
{
  "predicted_price_usd": 1456.78,
  "price_category": "Mid-Range",
  "model_used": "CatBoost Regressor",
  "dataset_used": "realistic",
  "input_received": { ... }
}
```

---

## 🔌 Partea 3: Server FastMCP

### Rulare
```bash
cd Hackathon-Raiffaisen
python mcp_server.py
```

### Tools (apelabile de LLM)

| Tool | Descriere |
|------|-----------|
| `predict_laptop_price` | Predicție preț cu model + dataset la alegere |
| `compare_all_models` | Compară predicțiile tuturor celor 4 modele |
| `get_model_metrics` | Metrici MAE/RMSE/R² pentru toate modelele |
| `get_dataset_stats` | Statistici complete despre dataset |

### Resources (context static pentru LLM)

| Resource URI | Descriere |
|--------------|-----------|
| `dataset://stats` | Statistici generale ambele dataset-uri |
| `models://metrics` | Metrici performanță toate modelele |

### Prompts (template-uri predefinite)

| Prompt | Descriere |
|--------|-----------|
| `interpret_price` | Analiză detaliată a unui preț estimat (market positioning, value analysis, marketing) |
| `recommend_laptop` | Recomandare configurație pe baza bugetului și use-case-ului |

---

## 🖥️ Partea 4: Integrare Client MCP

### Configurare Claude Desktop

Adaugă în `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "laptop-price-predictor": {
      "command": "python",
      "args": ["C:/cale/catre/Hackathon-Raiffaisen/mcp_server.py"],
      "env": {}
    }
  }
}
```

### Exemplu de interacțiune
**User:** "Am un laptop Dell cu Intel i7, 16GB RAM, SSD 512GB, ecran 15.6 inch, RTX 3060, Windows 11, 1.8kg, 8h baterie, 2 ani garanție. Cât ar trebui să coste și cum îl promovez?"

**LLM (via MCP):**
1. Apelează `predict_laptop_price` → obține $1,456
2. Apelează `get_dataset_stats` → obține context piață
3. Folosește prompt-ul `interpret_price` → generează analiză completă

---

## 🛠️ Tehnologii Folosite

| Categorie | Tehnologie |
|-----------|-----------|
| ML Training | scikit-learn, CatBoost, pandas, numpy |
| API Backend | FastAPI, Pydantic, uvicorn |
| MCP Server | FastMCP (Python SDK oficial) |
| Serializare modele | joblib, pickle |
| Vizualizare | matplotlib, seaborn |
| Client MCP | Claude Desktop |

---

## 📁 Structura Fișierelor

```
Hackathon-Raiffaisen/
├── main.py                              # FastAPI server (Partea 2)
├── mcp_server.py                        # FastMCP server (Partea 3)
├── part1_regresie.ipynb                 # Notebook ML (Partea 1)
├── laptop Price Prediction Dataset.csv  # Dataset original
├── laptop_price_realistic.csv           # Dataset realist (generat)
├── models/
│   ├── original_ridge_regression.pth    # Ridge pe dataset original
│   ├── original_random_forest.pth       # RF pe dataset original
│   ├── original_gradient_boosting.pth   # GBR pe dataset original
│   ├── original_catboost.pth            # CatBoost pe dataset original
│   ├── original_metadata.pth            # Metrici dataset original
│   ├── realistic_ridge.pth             # Ridge pe dataset realist
│   ├── realistic_rf.pth                # RF pe dataset realist
│   ├── realistic_gbr.pth              # GBR pe dataset realist
│   ├── realistic_catboost.pth          # CatBoost pe dataset realist
│   └── realistic_metadata.pth          # Metrici dataset realist
├── photos/                              # Grafice EDA
│   ├── price_distribution.png
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   └── ...
└── README.md                            # Acest fișier
```

---

## 🎤 Puncte Cheie pentru Prezentare

1. **Pipeline complet end-to-end:** De la date brute → EDA → antrenare → API → MCP → interacțiune LLM
2. **4 modele comparate:** Ridge (baseline liniar) → Random Forest (bagging) → Gradient Boosting → CatBoost (SOTA)
3. **Două dataset-uri:** Original (demonstrează că pipeline-ul e corect chiar pe date fără semnal) + Realist (demonstrează performanță reală R²=0.87)
4. **MCP cu 4 tools:** Nu doar predicție, ci și comparare modele, metrici, statistici
5. **Prompts inteligente:** Template-uri care ghidează LLM-ul să facă analiză de piață, nu doar să returneze un număr
6. **Selecție model la runtime:** Utilizatorul poate alege algoritmul și dataset-ul — transparență totală
