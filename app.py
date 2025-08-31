from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from rapidfuzz import process

# Load model and encoder
model = load_model("models/medicine_model.h5")
le = joblib.load("models/label_encoder.pkl")

# Load dataset to get symptom names
df = pd.read_csv("diseases-and-symptoms-dataset/Diseases_and_Symptoms.csv")
symptoms = list(df.columns[1:])  # ensure it's a list

app = Flask(__name__)

# ---- Symptom Suggestions ----
@app.route("/suggest", methods=["GET"])
def suggest():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    matches = process.extract(query, symptoms, limit=5)
    return jsonify([m[0] for m in matches])

# ---- Disease Prediction ----
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    selected_symptoms = data.get("symptoms", [])

    # Create binary input vector
    input_vector = np.zeros(len(symptoms))
    for s in selected_symptoms:
        if s in symptoms:
            input_vector[symptoms.index(s)] = 1
        else:
            # Fuzzy match fallback (only if similarity is high enough)
            match, score, _ = process.extractOne(s, symptoms)
            if score > 80:  # threshold to avoid random matches
                input_vector[symptoms.index(match)] = 1

    input_vector = input_vector.reshape(1, -1)
    pred_prob = model.predict(input_vector)
    pred_class = np.argmax(pred_prob)
    disease = le.inverse_transform([pred_class])[0]
    confidence = float(np.max(pred_prob)) * 100

    return jsonify({
        "disease": disease,
        "confidence": round(confidence, 2)
    })

# ---- Home page ----
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
