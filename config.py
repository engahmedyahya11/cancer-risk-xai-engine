
FEATURES = [
    {"key": "age",                "label": "Patient Age",          "type": "slider",   "min": 18,  "max": 90,  "default": 45},
    {"key": "stage",              "label": "Cancer Stage",         "type": "slider",   "min": 1,   "max": 4,   "default": 2},
    {"key": "bmi",                "label": "BMI",                  "type": "slider",   "min": 15,  "max": 45,  "default": 25},
    {"key": "prior_surgeries",    "label": "Prior Surgeries",      "type": "slider",   "min": 0,   "max": 5,   "default": 0},
    {"key": "comorbidities",      "label": "Comorbidities Count",  "type": "slider",   "min": 0,   "max": 6,   "default": 1},
    {"key": "performance_status", "label": "Performance Status",   "type": "slider",   "min": 0,   "max": 4,   "default": 1},
    {"key": "smoking",            "label": "Smoking History",      "type": "checkbox", "default": False},
    {"key": "diabetes",           "label": "Diabetes",             "type": "checkbox", "default": False},
    {"key": "family_history",     "label": "Family History",       "type": "checkbox", "default": False},
    {"key": "metastasis",         "label": "Metastasis Present",   "type": "checkbox", "default": False},
    {"key": "cancer_type",        "label": "Cancer Type",          "type": "dropdown",
     "choices": ["Breast","Lung","Colon","Liver","Leukemia","Lymphoma","Prostate","Thyroid"],
     "default": "Breast"},
    {"key": "treatment_type",     "label": "Treatment Type",       "type": "dropdown",
     "choices": ["Surgery","Chemotherapy","Radiation","Immunotherapy","Palliative"],
     "default": "Surgery"},
]

CANCER_TYPE_MAP  = {"Breast":0,"Lung":1,"Colon":2,"Liver":3,"Leukemia":4,"Lymphoma":5,"Prostate":6,"Thyroid":7}
TREATMENT_MAP    = {"Surgery":0,"Chemotherapy":1,"Radiation":2,"Immunotherapy":3,"Palliative":4}

RISK_TIERS = {
    0: {"label": "Low Risk",      "color": "#22c55e", "badge": "🟢"},
    1: {"label": "Medium Risk",   "color": "#eab308", "badge": "🟡"},
    2: {"label": "High Risk",     "color": "#f97316", "badge": "🟠"},
    3: {"label": "Critical Risk", "color": "#ef4444", "badge": "🔴"},
}

COST_BANDS = {
    0: "EGP 15,000 – 40,000 / year",
    1: "EGP 40,000 – 90,000 / year",
    2: "EGP 90,000 – 180,000 / year",
    3: "EGP 180,000+ / year",
}

# ── Medical validation rules ───────────────────────────────────────────────────

# سرطانات مستحيلة حسب الجنس
MALE_IMPOSSIBLE   = []          # مفيش سرطان مستحيل 100% في الراجل من اللي عندنا
FEMALE_IMPOSSIBLE = ["Prostate"]

# سرطانات نادرة جداً وأخطر لو جت في الجنس ده
RARE_HIGH_RISK = {
    "Male":   ["Breast"],       # راجل عنده Breast Cancer → أخطر بدرجة
    "Female": []
}

# Stage vs Metastasis
# Stage 1 أو 2 مع Metastasis = تناقض طبي
METASTASIS_WARNING_STAGES = [1, 2]

# BMI شديد الانخفاض مع مؤشرات صحة طبيعية = غير منطقي
BMI_CACHEXIA_THRESHOLD = 16.0

# عمر صغير مع Performance Status عالي = نادر، بس ممكن
YOUNG_AGE_THRESHOLD       = 30
HIGH_PERFORMANCE_STATUS   = 3

AMC_BLUE   = "#1a6fb5"
AMC_ORANGE = "#f7941d"
