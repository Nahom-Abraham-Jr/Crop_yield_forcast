import streamlit as st
import pandas as pd
import joblib

st.title("Yield Prediction App")
st.write("Enter farm details to predict yield.")

region = st.selectbox("Region", ["Oromia", "Amhara", "Tigray", "SNNPR", "Somali"])
crop = st.selectbox("Crop", ["Teff", "Maize", "Wheat", "Sorghum", "Coffee"])
year = st.selectbox("Year", [2021, 2022, 2023, 2024])
# more inputs would go here

if st.button("Predict"):
    st.write("Predicted Yield: 2.5 t/ha (mock)")
