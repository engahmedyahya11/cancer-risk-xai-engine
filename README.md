# Cancer Risk XAI Engine

End-to-end Cancer Risk Scoring system for health insurance underwriting with full SHAP-based explainability and a Gradio web UI.

## 🧠 Overview

This project simulates a cancer risk intelligence tool used by health insurance underwriting teams.  
The app takes anonymized patient-like features as input and outputs:

- A risk tier (Low / Medium / High / Critical).
- An estimated annual cost band (EGP).
- Class probabilities for each risk tier.
- A SHAP-based feature impact chart explaining the decision.

All data is synthetic. **No real patient data is used.**

## ⚙️ Tech Stack

- Python
- XGBoost, scikit-learn
- SHAP (Explainable AI)
- Pandas, NumPy
- Gradio (web UI)
- Matplotlib

## 🏗️ Project Structure

```text
cancer-risk-xai-engine/
├── app.py              # Gradio interface (UI + inference)
├── model.py            # RiskModel class (train / predict / encode)
├── explainer.py        # SHAPExplainer class + plots
├── config.py           # Feature definitions, risk tiers, cost bands
├── data_generator.py   # Synthetic cancer patient data
├── risk_model.pkl      # Trained XGBoost model (optional pre-trained)
├── logo.png
├── logo_transparent.png
├── requirements.txt
└── Procfile            # Render deployment command
```

## 🚀 Running Locally

```bash
# 1) Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# 2) Install dependencies
pip install -r requirements.txt

# 3) Launch the app
python app.py
```

Then open: `http://localhost:7860`

## 🌐 Deployment

This repository is designed to be deployed as a **Python Web Service on Render**:

- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`

Gradio is configured to bind to `0.0.0.0` and read the port from the `PORT` environment variable (required by Render).[web:4][web:8]

## ⚠️ Disclaimer

This tool is **for research and educational purposes only**:

- It does **not** provide medical diagnosis or treatment advice.
- It is intended to illustrate Explainable AI techniques in healthcare risk modeling.
- Any real-world use would require clinical validation and regulatory review.
