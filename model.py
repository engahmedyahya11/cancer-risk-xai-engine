
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data_generator import generate_data
from config import RISK_TIERS, COST_BANDS, CANCER_TYPE_MAP, TREATMENT_MAP

FEATURE_COLS = [
    "age","stage","bmi","prior_surgeries","comorbidities",
    "performance_status","smoking","diabetes","family_history",
    "metastasis","cancer_type","treatment_type","gender"
]

class RiskModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        )
        self.X_train = None

    def train(self):
        df = generate_data(1200)
        X = df[FEATURE_COLS]
        y = df["risk"]
        self.X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(self.X_train, y_train)
        preds = self.model.predict(X_test)
        print("=== Model Evaluation ===")
        print(classification_report(y_test, preds,
              target_names=[RISK_TIERS[i]["label"] for i in range(4)]))
        joblib.dump(self.model, "risk_model.pkl")
        return self.X_train

    def load(self):
        self.model = joblib.load("risk_model.pkl")

    def encode_input(self, raw: dict) -> pd.DataFrame:
        row = raw.copy()
        row["cancer_type"]    = CANCER_TYPE_MAP[raw["cancer_type"]]
        row["treatment_type"] = TREATMENT_MAP[raw["treatment_type"]]
        row["smoking"]        = int(raw["smoking"])
        row["diabetes"]       = int(raw["diabetes"])
        row["family_history"] = int(raw["family_history"])
        row["metastasis"]     = int(raw["metastasis"])
        row["gender"]         = 1 if raw["gender"] == "Male" else 0
        return pd.DataFrame([row])[FEATURE_COLS]

    def predict(self, raw: dict):
        X = self.encode_input(raw)
        tier  = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]
        return tier, proba

    def get_tier_info(self, tier: int):
        return RISK_TIERS[tier], COST_BANDS[tier]
