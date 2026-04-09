import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below:")

# ONLY 6 INPUTS (IMPORTANT)
age = st.number_input("Age")
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
cigs = st.number_input("Cigarettes per Day")
chol = st.number_input("Total Cholesterol")
bp = st.number_input("Systolic Blood Pressure")
glucose = st.number_input("Glucose Level")

if st.button("Predict"):
    data = np.array([[age, sex, cigs, chol, bp, glucose]])
    data = scaler.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk")