import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# Define Model Path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "dbh_model.h5")

# Load the trained model
try:
   model = tf.keras.models.load_model(MODEL_PATH, compile=False)
   st.success("✅ Model loaded successfully!")
except OSError:
    st.error("❌ Error: Model file not found. Please check the path or retrain the model.")
    st.stop()

# Function to preprocess image
def preprocess_image(image):
    try:
        image = cv2.resize(image, (128, 128))
        image = image / 255.0  # Normalize
        return np.expand_dims(image, axis=0)  # Add batch dimension
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None

# Functions for Tree Calculations
def estimate_height(dbh_cm, a=1.3, b=0.75):
    return a * (dbh_cm ** b)

def calculate_agb(dbh_cm, height):
    return 0.136 * (dbh_cm ** 2.38) * (height ** 0.89)

def calculate_carbon(agb):
    return agb * 0.5

def calculate_co2(carbon):
    return carbon * 3.67

# Streamlit UI
st.title("🌳 Tree DBH & Carbon Sequestration Calculator")
st.write("Upload an image of a mango tree to estimate its DBH, height, and environmental impact.")

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Read image file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("❌ Error: Unable to read the image file. Please upload a valid image.")
        else:
            # Display uploaded image
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            
            # Preprocess and predict DBH
            processed_image = preprocess_image(image)
            if processed_image is not None:
                prediction = model.predict(processed_image)
                estimated_dbh = prediction[0][0]
                
                # Calculate other tree attributes
                height = estimate_height(estimated_dbh)
                agb = calculate_agb(estimated_dbh, height)
                carbon = calculate_carbon(agb)
                co2 = calculate_co2(carbon)
                
                # Display results
                st.subheader("📊 Estimated Tree Data:")
                st.write(f"🌲 **Estimated DBH:** {estimated_dbh:.2f} cm")
                st.write(f"📏 **Estimated Height:** {height:.2f} m")
                st.write(f"🌿 **Above-Ground Biomass (AGB):** {agb:.2f} kg")
                st.write(f"💨 **Carbon Content:** {carbon:.2f} kg")
                st.write(f"🌍 **CO₂ Sequestration:** {co2:.2f} kg")
                
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")