import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
with open('xgb_regressor_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Title
st.title("🚗 Car Price Prediction App")

# User input form
st.header("Enter Car Details")

brand = st.selectbox("Brand", label_encoders['Brand'].classes_)
model_name = st.selectbox("Model", label_encoders['Model'].classes_)
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1)
engine_size = st.number_input("Engine Size (cc)", min_value=500, max_value=10000, step=100)
fuel_type = st.selectbox("Fuel Type", label_encoders['Fuel_Type'].classes_)
transmission = st.selectbox("Transmission", label_encoders['Transmission'].classes_)
mileage = st.number_input("Mileage (in km)", min_value=0, max_value=500000, step=100)

# Prepare input
if st.button("Predict Price"):
    try:
        # Encode categorical values
        input_data = {
            'Brand': label_encoders['Brand'].transform([brand])[0],
            'Model': label_encoders['Model'].transform([model_name])[0],
            'Year': year,
            'Engine_Size': engine_size,
            'Fuel_Type': label_encoders['Fuel_Type'].transform([fuel_type])[0],
            'Transmission': label_encoders['Transmission'].transform([transmission])[0],
            'Mileage': mileage
        }

        input_df = pd.DataFrame([input_data])
        predicted_price = model.predict(input_df)[0]

        # Format as currency
        st.success(f"💰 Predicted Price: ₹ {int(predicted_price):,}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
