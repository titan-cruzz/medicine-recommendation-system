# Medicine Recommendation System 💊

A machine learning + Flask based web application that predicts possible diseases based on symptoms and recommends medicines.

---

## 🚀 Features
- 🔎 Symptom search with auto-suggestions  
- 🩺 Disease prediction using a trained ML model  
- 🌐 Flask API endpoints (`/predict`, `/suggest`)  
- 🖥️ Simple web frontend with HTML/CSS/JS  

---

## 🛠️ Tech Stack
- **Python**: Flask, Pandas, NumPy, TensorFlow/Keras  
- **RapidFuzz**: for symptom suggestions  
- **Frontend**: HTML, CSS, JavaScript  

---

## 📂 Project Structure
medicine_recommendation_system/
│── app.py # Flask backend
│── main.py # Script to train the model
│── models/ # Saved ML model (.h5, .pkl)
│── templates/ # HTML files
│── static/ # CSS, JS files
│── medications_dataset/ # Dataset for diseases & medications
│── .gitignore
│── README.md


---
## ⚙️ Setup Instructions

### 1. Clone the repository

git clone https://github.com/titan-cruzz/medicine_recommendation_system.git
cd medicine_recommendation_system

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Train the Model
python main.py

### 4. Evaluate model
python evaluate_model.py

### 5. Run the web application
python app.py
```bash

