
import numpy as np
import pandas as pd
from config import CANCER_TYPE_MAP, TREATMENT_MAP

def generate_data(n=1200, seed=42):
    np.random.seed(seed)
    N = n

    # ── Gender (0=Female, 1=Male) ──────────────────────────────────────────
    gender = np.random.binomial(1, 0.45, N)   # 45% male, 55% female

    age              = np.random.normal(52, 14, N).clip(18, 90)
    stage            = np.random.choice([1,2,3,4], N, p=[0.25,0.35,0.25,0.15])
    bmi              = np.random.normal(26, 5, N).clip(15, 45)
    prior_surgeries  = np.random.poisson(0.8, N).clip(0, 5)
    comorbidities    = np.random.poisson(1.2, N).clip(0, 6)
    performance      = np.random.choice([0,1,2,3,4], N, p=[0.2,0.35,0.25,0.12,0.08])
    smoking          = np.random.binomial(1, 0.35, N)
    diabetes         = np.random.binomial(1, 0.25, N)
    family_history   = np.random.binomial(1, 0.30, N)
    metastasis       = (stage == 4).astype(int) | np.random.binomial(1, 0.05, N)
    metastasis       = metastasis.clip(0, 1)
    cancer_type      = np.random.choice(list(CANCER_TYPE_MAP.values()), N)
    treatment_type   = np.random.choice(list(TREATMENT_MAP.values()), N)

    # ── Fix impossible combos in training data ─────────────────────────────
    # ست معاهاش Prostate
    female_mask = gender == 0
    prostate_code = CANCER_TYPE_MAP["Prostate"]
    breast_code   = CANCER_TYPE_MAP["Breast"]
    lung_code     = CANCER_TYPE_MAP["Lung"]

    cancer_type = cancer_type.copy()
    cancer_type[female_mask & (cancer_type == prostate_code)] = breast_code

    # ── Risk score ─────────────────────────────────────────────────────────
    score = (
        (stage - 1) * 2.5
        + metastasis * 3.5
        + performance * 0.8
        + smoking * 1.2
        + diabetes * 0.9
        + family_history * 0.6
        + comorbidities * 0.4
        + (age - 18) / 30
        + prior_surgeries * 0.3
        # راجل عنده Breast Cancer → risk أعلى
        + ((gender == 1) & (cancer_type == breast_code)).astype(float) * 1.5
        + np.random.normal(0, 0.8, N)
    )

    thresholds = np.percentile(score, [30, 60, 85])
    risk = np.digitize(score, thresholds).clip(0, 3)

    df = pd.DataFrame({
        "age": age.astype(int),
        "stage": stage,
        "bmi": bmi.round(1),
        "prior_surgeries": prior_surgeries,
        "comorbidities": comorbidities,
        "performance_status": performance,
        "smoking": smoking,
        "diabetes": diabetes,
        "family_history": family_history,
        "metastasis": metastasis,
        "cancer_type": cancer_type,
        "treatment_type": treatment_type,
        "gender": gender,
        "risk": risk,
    })
    return df
