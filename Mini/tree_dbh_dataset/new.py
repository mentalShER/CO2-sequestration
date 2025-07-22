import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Set paths (relative to current working directory)
image_dir = "images"           # Folder containing tree images
csv_path = "labels.csv"        # CSV with 'image_name' and 'dbh_cm'

# Load CSV
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # Remove unwanted spaces
df.rename(columns={"image_name": "filename", "dbh_cm": "dbh"}, inplace=True)  # Ensure column consistency
print("CSV Loaded. Total entries:", len(df))

# Load images and labels
X = []
y = []

# Function to load image with multiple extensions
def load_image(image_path):
    extensions = ['.png', '.jpg', '.jpeg']
    for ext in extensions:
        file_path = image_path + ext
        if os.path.isfile(file_path):
            return cv2.imread(file_path)
    print(f"Warning: {image_path} not found")
    return None

for index, row in df.iterrows():
    filename = row["filename"]
    label = row["dbh"]
    img_path = os.path.join(image_dir, filename)
    img = load_image(os.path.splitext(img_path)[0])  # remove extension if any
    if img is not None:
        img = cv2.resize(img, (128, 128))
        img = img.flatten() / 255.0  # Flatten and normalize
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Random Forest Model - Mean Absolute Error: {mae:.4f}")

# Save model
joblib.dump(rf_model, "rf_model.pkl")
print("Random Forest model trained and saved as rf_model.pkl")
