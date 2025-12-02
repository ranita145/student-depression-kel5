import pandas as pd
import joblib
import requests
from io import BytesIO

FILE_ID = "1xtI6w3HYyormFkK10Zy95IVgi1HXJV7s"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def load_model():
    """
    Download model .sav dari Google Drive, lalu load dengan joblib.
    Fungsi ini bisa dipanggil dari app Streamlit.
    """
    response = requests.get(MODEL_URL)
    response.raise_for_status()  # kalau gagal download, akan error jelas
    model = joblib.load(BytesIO(response.content))
    return model

model = load_model()

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
