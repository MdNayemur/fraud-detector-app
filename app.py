import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model(autoencoder_model.h5)
scaler = joblib.load(scaler.pkl)

# Set threshold (same as used in Colab)
THRESHOLD = 0.02

st.title("Credit Card Fraud Detection")
st.markdown("Enter transaction details below to check if it's fraudulent.")

# Input fields
inputs = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", format="%.6f", value=0.0)
    inputs.append(val)

amount = st.number_input("Amount", format="%.2f", value=0.0)
inputs.append(amount)

if st.button("Check Transaction"):
    # Prepare and scale input
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Predict reconstruction error
    reconstruction = model.predict(input_scaled)
    mse = np.mean(np.square(input_scaled - reconstruction))

    # Result
    st.write(f"Reconstruction Error: {mse:.6f}")
    if mse  THRESHOLD
        st.error( "This transaction is likely FRAUDULENT!")
    else
        st.success("This transaction appears to be NORMAL.")
