import streamlit as st
from prediction import predict_depression

st.title("Prediksi Depresi Mahasiswa")
st.write("Model: **Random Forest Classifier** (di-load dari Google Drive)")

st.header("Input Data")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 15, 60, 20)

academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)

sleep_duration = st.selectbox(
    "Sleep Duration",
    ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"]
)

work_study_hours = st.slider("Work/Study Hours per day", 0, 16, 6)
financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)

family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])

if st.button("Prediksi"):
    input_data = {
        "Gender": gender,
        "Age": age,
        "Academic Pressure": academic_pressure,
        "Study Satisfaction": study_satisfaction,
        "Sleep Duration": sleep_duration,
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_history,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
    }

    label, proba = predict_depression(input_data)

    st.subheader("Hasil Prediksi")
    if label == 1:
        st.error(f"Model memprediksi **DEPRESI** dengan probabilitas {proba:.2%}")
    else:
        st.success(f"Model memprediksi **TIDAK DEPRESI** dengan probabilitas {proba:.2%}")

    st.caption("Ini hanya model data, bukan diagnosis medis.")
