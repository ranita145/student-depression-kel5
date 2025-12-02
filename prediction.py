import pandas as pd
import joblib

# Load pipeline (preprocess + RandomForest)
model = joblib.load("rf_depression_model.sav")

FEATURES = [
    "Gender",
    "Age",
    "Academic Pressure",
    "Study Satisfaction",
    "Sleep Duration",
    "Work/Study Hours",
    "Financial Stress",
    "Family History of Mental Illness",
    "Have you ever had suicidal thoughts ?"
]

def predict_depression(input_dict):
    """
    input_dict: dict {nama_fitur: nilai}
    return: label (0/1), prob kelas 1 (depresi)
    """
    data = {k: [input_dict[k]] for k in FEATURES}
    df = pd.DataFrame(data)
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]
    return int(pred), float(proba)