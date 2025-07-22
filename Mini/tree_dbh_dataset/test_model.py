import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# Load the trained model
model_path = "dbh_model.h5"  # Path to the saved model
model = tf.keras.models.load_model(model_path)

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image {image_path} not found.")
        return None
    img = cv2.resize(img, (128, 128))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    return img

# Function to estimate height based on DBH
def estimate_height(dbh_cm, a=2.5, b=0.5):
    return a * (dbh_cm ** b)

# Function to calculate above-ground biomass (AGB)
def calculate_agb(dbh_cm, height):
    return 0.25 * (dbh_cm ** 2) * height

# Function to calculate carbon content
def calculate_carbon(agb):
    return agb * 0.5

# Function to calculate CO2 sequestration
def calculate_co2(carbon):
    return carbon * 3.67

# Test the model with a single image
def test_model(image_path):
    img = load_and_preprocess_image(image_path)
    if img is not None:
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        prediction = model.predict(img)
        dbh_cm = prediction[0][0]  # Predicted DBH
        print(f"Predicted DBH for {image_path}: {dbh_cm:.2f} cm")

        # Calculate height, AGB, carbon content, and CO2 sequestration
        height = estimate_height(dbh_cm)
        agb = calculate_agb(dbh_cm, height)
        carbon = calculate_carbon(agb)
        co2 = calculate_co2(carbon)

        # Print the results
        print(f"Estimated Height: {height:.2f} m")
        print(f"Above Ground Biomass (AGB): {agb:.2f} kg")
        print(f"Carbon Content: {carbon:.2f} kg")
        print(f"CO2 Sequestration: {co2:.2f} kg")

# Example usage
if __name__ == "__main__":
    # Set the path to the test image
    test_image_path = "test_image/download.jpg"  # Change this to your test image path
    test_model(test_image_path)

    # If you want to test multiple images, you can loop through a directory
    # test_images_dir = "path/to/your/test/images"  # Directory containing test images
    # for filename in os.listdir(test_images_dir):
    #     if filename.endswith(('.jpg', '.jpeg', '.png')):
    #         test_model(os.path.join(test_images_dir, filename))