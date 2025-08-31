# Medicine Recommendation System 💊

A machine learning + Flask web application that predicts possible diseases based on symptoms and recommends medicines.

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
```
medicine_recommendation_system/
│── app.py              # Flask backend
│── main.py             # Script to train the model
│── evaluate_model.py   # Script to evaluate the model
│── models/             # Saved ML model (.h5, .pkl)
│── templates/          # HTML files
│── static/             # CSS, JS files
│── medications_dataset/ # Dataset for diseases & medications
│── .gitignore
│── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
'''bash
git clone https://github.com/titan-cruzz/medicine_recommendation_system.git
cd medicine_recommendation_system
<<<<<<< HEAD
'''
=======
```

>>>>>>> 4e331e2 (Update README with setup instructions and project details)
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python main.py
```

### 4. Evaluate the Model
```bash
python evaluate_model.py
```

### 5. Run the Web Application
```bash
python app.py
<<<<<<< HEAD
```bash
=======
```

---
>>>>>>> 4e331e2 (Update README with setup instructions and project details)

## 📈 Usage
1. Open the web application in your browser (`http://127.0.0.1:5000`)  
2. Enter symptoms in the search box  
3. View predicted disease and recommended medicines  

---

## 📂 Dataset
- The `medications_dataset/` folder contains disease-medication mappings used for recommendation.  
- Format: CSV file with diseases and corresponding medicines.

---

## ✨ Contribution
- Fork the repository  
- Create a new branch for your feature/fix  
- Submit a pull request  

---

## 📄 License
This project is licensed under the MIT License.
