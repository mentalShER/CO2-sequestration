import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Set paths
image_dir = "images"  # Folder containing tree images
csv_path = "labels.csv"  # CSV with 'image_name' and 'dbh_cm'

# Check if CSV file exists
if not os.path.isfile(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    print("Files in the dataset folder:", os.listdir("tree_dbh_dataset"))
    exit()

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
    else:
        print(f"Warning: Image {filename} not found, skipping.")

# Ensure data is not empty
if len(X) == 0:
    print("Error: No valid images loaded. Please check your dataset.")
    exit()

X = np.array(X)
y = np.array(y)

# Split data
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except ValueError as e:
    print(f"Error: {e}")
    print("Make sure there are enough valid images to split.")
    exit()

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # Regression output
])

model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=16)

# Evaluate model
y_pred = model.predict(X_test)

# Calculate R-squared and MAE
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R-squared: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Calculate accuracy as 1 - (MAE / Mean of Actual Values)
accuracy = 1 - (mae / np.mean(y_test))
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
model.save("dbh_model.h5")
print("Model trained and saved as tree_dbh_dataset/dbh_model.h5")
