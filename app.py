import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== LOAD MODEL =====
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Fix for sklearn version compatibility
for estimator in model.estimators_:
    if not hasattr(estimator, 'monotonic_cst'):
        estimator.monotonic_cst = None

# ===== CONFIG =====
st.set_page_config(page_title="Prediksi Depresi", layout="wide")

# ===== STYLE =====
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown('<p class="title">🧠 Prediksi Tingkat Depresi</p>', unsafe_allow_html=True)

st.divider()

# ===== LAYOUT 2 KOLOM =====
col1, col2 = st.columns(2)

# ===== INPUT KOLOM KIRI =====
with col1:
    st.subheader("📊 Data Pribadi")

    age = st.number_input("Age", 10, 100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    sleep = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours"])
    diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

# ===== INPUT KOLOM KANAN =====
with col2:
    st.subheader("📈 Kondisi Akademik & Mental")

    academic = st.slider("Academic Pressure", 0, 10)
    work = st.slider("Work Pressure", 0, 10)
    study_sat = st.slider("Study Satisfaction", 0, 5)
    job_sat = st.slider("Job Satisfaction", 0, 5)
    hours = st.slider("Work/Study Hours", 0, 24)
    financial = st.slider("Financial Stress", 0, 10)
    suicidal = st.selectbox("Suicidal Thoughts", ["Yes", "No"])
    family = st.selectbox("Family History", ["Yes", "No"])

st.divider()

# ===== PROSES DATA =====
input_dict = {col: 0 for col in columns}

# NUMERIK
input_dict["Age"] = age
input_dict["Academic Pressure"] = academic
input_dict["Work Pressure"] = work
input_dict["Study Satisfaction"] = study_sat
input_dict["Job Satisfaction"] = job_sat
input_dict["Work/Study Hours"] = hours
input_dict["Financial Stress"] = financial

# DUMMY MAPPING
if "Gender_Male" in input_dict and gender == "Male":
    input_dict["Gender_Male"] = 1

if sleep == "5-6 hours" and "Sleep Duration_5-6 hours" in input_dict:
    input_dict["Sleep Duration_5-6 hours"] = 1
elif sleep == "7-8 hours" and "Sleep Duration_7-8 hours" in input_dict:
    input_dict["Sleep Duration_7-8 hours"] = 1

if diet == "Healthy" and "Dietary Habits_Healthy" in input_dict:
    input_dict["Dietary Habits_Healthy"] = 1
elif diet == "Moderate" and "Dietary Habits_Moderate" in input_dict:
    input_dict["Dietary Habits_Moderate"] = 1

if suicidal == "Yes" and "Have you ever had suicidal thoughts ?_Yes" in input_dict:
    input_dict["Have you ever had suicidal thoughts ?_Yes"] = 1

if family == "Yes" and "Family History of Mental Illness_Yes" in input_dict:
    input_dict["Family History of Mental Illness_Yes"] = 1

# dataframe
input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)

# ===== BUTTON =====
st.subheader("🔍 Hasil Prediksi")

if st.button("Prediksi Sekarang"):
    pred = model.predict(input_scaled)

    if pred[0] == 1:
        st.error("⚠️ Terindikasi Depresi")
        st.markdown("### 💡 Saran:")
        st.write("- Kurangi tekanan kerja/akademik")
        st.write("- Jaga pola tidur")
        st.write("- Konsultasi ke profesional jika perlu")
    else:
        st.success("✅ Tidak Depresi")
        st.markdown("### 👍 Kondisi Baik:")
        st.write("- Pertahankan gaya hidup sehat")
        st.write("- Jaga keseimbangan aktivitas")

    with st.expander("Lihat Data Input"):
        st.write(input_df)