import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart App", layout="centered")

st.title("❤️ Heart Disease Prediction App")

# Session state (important)
if "predicted" not in st.session_state:
    st.session_state.predicted = False

# ---------------- INPUT SCREEN ----------------
if not st.session_state.predicted:

    st.subheader("📝 Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age")
        cigs = st.number_input("Cigarettes per Day")
        bp = st.number_input("Systolic BP")

    with col2:
        sex_option = st.selectbox("Sex", ["Female", "Male"])
        sex = 0 if sex_option == "Female" else 1
        chol = st.number_input("Total Cholesterol")
        glucose = st.number_input("Glucose Level")

    if st.button("🚀 Predict"):
        data = np.array([[age, sex, cigs, chol, bp, glucose]])
        data = scaler.transform(data)

        prediction = model.predict(data)
        prob = model.predict_proba(data)[0][1]

        # Save result
        st.session_state.result = prediction[0]
        st.session_state.prob = prob
        st.session_state.predicted = True

# ---------------- RESULT SCREEN ----------------
else:
    st.subheader("📊 Prediction Result")

    if st.session_state.result == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk")

    # Show probability
    st.write(f"### 🧠 Risk Probability: {st.session_state.prob * 100:.2f}%")

    # Progress bar (cool UI)
    st.progress(int(st.session_state.prob * 100))

    if st.button("🔙 Predict Again"):
        st.session_state.predicted = False
