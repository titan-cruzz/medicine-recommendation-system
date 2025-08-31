import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib
import os

# Load dataset
df = pd.read_csv("diseases-and-symptoms-dataset/Diseases_and_Symptoms.csv")

X = df.iloc[:, 1:]  # symptoms
y = df.iloc[:, 0]   # diseases

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Train-test split with fixed random_state
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Define model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(y_categorical.shape[1], activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=30, batch_size=128, validation_split=0.2)

# Save model & encoder
os.makedirs("models", exist_ok=True)
model.save("models/medicine_model.h5")
joblib.dump(le, "models/label_encoder.pkl")

# Save X_test and y_test for evaluate_model.py
os.makedirs("data", exist_ok=True)
X_test.to_csv("data/X_test.csv", index=False)
np.save("data/y_test.npy", y_test)
