import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load('house_price_model.pkl')

# Title
st.title("House Price Prediction System")

# Inputs
area = st.number_input("Enter Area (sq ft)")

bedrooms = st.number_input("Enter Number of Bedrooms")

# Predict Button
if st.button("Predict Price"):

    input_data = np.array([[area, bedrooms]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Price: {prediction[0]:.2f} Lakhs")