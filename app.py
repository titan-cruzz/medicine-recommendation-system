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
med_df = pd.read_csv("medications_dataset/disease_medications_dataset.csv")
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

    # ---- Create binary input vector ----
    input_vector = np.zeros(len(symptoms))
    for s in selected_symptoms:
        if s in symptoms:
            input_vector[symptoms.index(s)] = 1
        else:
            # Fuzzy match fallback (only if similarity is high enough)
            match, score, _ = process.extractOne(s, symptoms)
            if score > 80:
                input_vector[symptoms.index(match)] = 1

    input_vector = input_vector.reshape(1, -1)

    # ---- Predict disease ----
    pred_prob = model.predict(input_vector)
    pred_class = np.argmax(pred_prob)
    disease = le.inverse_transform([pred_class])[0]
    confidence = float(np.max(pred_prob)) * 100

    # ---- Normalize once ----
    disease_clean = disease.strip().lower()
    med_df.columns = med_df.columns.str.strip().str.lower()
    med_df["diseases"] = med_df["diseases"].astype(str).str.strip().str.lower()

    # ---- Lookup medications ----
    meds_row = med_df.loc[med_df["diseases"] == disease_clean, "preferred medications"]

    if not meds_row.empty:
        medications = [m.strip() for m in meds_row.iloc[0].split(",")]
    else:
        medications = ["NA"]

    # ---- Debug print (check logs) ----
    print(f"Predicted: {disease_clean}, Found medications: {medications}")

    # ---- Return response ----
    return jsonify({
        "disease": disease,
        "confidence": round(confidence, 2),
        "medications": medications
    })

# ---- Home page ----
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
