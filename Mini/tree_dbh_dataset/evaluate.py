import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Set paths
image_dir = "images"  # Folder containing tree images
csv_path = "labels.csv"  # CSV with 'image_name' and 'dbh_cm'

# Load CSV
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df.rename(columns={"image_name": "filename", "dbh_cm": "dbh"}, inplace=True)

# Load images and labels
X = []
y = []

# Function to load image with multiple extensions
def load_image(image_path):
    extensions = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']
    for ext in extensions:
        file_path = image_path + ext
        if os.path.isfile(file_path):
            return cv2.imread(file_path)
    return None

for index, row in df.iterrows():
    filename = row["filename"].split('.')[0]  # Remove extension if present
    label = row["dbh"]
    img_path = os.path.join(image_dir, filename)
    img = load_image(img_path)
    if img is not None:
        img = cv2.resize(img, (128, 128))
        img = img / 255.0  # Normalize
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
model = load_model("dbh_model.h5")

# Predict using the model
y_pred = model.predict(X_test)

# Calculate R-squared value
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.4f}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")

# Calculate accuracy as 1 - (MAE / Mean of Actual Values)
accuracy = 1 - (mae / np.mean(y_test))
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display some sample predictions
for i in range(min(10, len(y_test))):
    print(f"Actual: {y_test[i]}, Predicted: {y_pred[i][0]:.2f}")
