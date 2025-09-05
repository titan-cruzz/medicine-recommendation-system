# Medicine Recommendation System 💊

A machine learning + Flask web application that predicts possible diseases based on symptoms and recommends medicines.

---

## 🚀 Features

* 🔎 Symptom search with auto-suggestions
* 🩺 Disease prediction using a trained ML model
* 🌐 Flask API endpoints (`/predict`, `/suggest`)
* 🖥️ Simple web frontend with HTML/CSS/JS

---

## 🛠️ Tech Stack

* **Python**: Flask, Pandas, NumPy, TensorFlow/Keras
* **RapidFuzz**: for symptom suggestions
* **Frontend**: HTML, CSS, JavaScript

---

## 📂 Project Structure

```
medicine_recommendation_system/
│── app.py                 # Flask backend
│── main.py                # Script to train the model
│── evaluate_model.py      # Script to evaluate the model
│── models/                # Saved ML model (.h5, .pkl)
│── templates/             # HTML files
│── static/                # CSS, JS files
│── medications_dataset/   # Dataset for diseases & medications
│── diseases-and-symptoms-dataset/ # Kaggle dataset for symptoms & diseases
│── .gitignore
│── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/titan-cruzz/medicine_recommendation_system.git
cd medicine_recommendation_system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

The **symptoms-to-disease dataset** must be downloaded separately from Kaggle:
👉 [Diseases and Symptoms Dataset (Kaggle)](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset)

After downloading, place the dataset CSV file inside the folder:

```
diseases-and-symptoms-dataset/
```

Alternatively, if you have the Kaggle API installed, you can download directly:

```bash
kaggle datasets download -d dhivyeshrk/diseases-and-symptoms-dataset -p diseases-and-symptoms-dataset --unzip
```

### 4. Train the Model

```bash
python main.py
```

### 5. Evaluate the Model

```bash
python evaluate_model.py
```

### 6. Run the Web Application

```bash
python app.py
```

---

## 📈 Usage

1. Open the web application in your browser (`http://127.0.0.1:5000`)
2. Enter symptoms in the search box
3. View predicted disease and recommended medicines

---

## 📂 Dataset

* The `medications_dataset/` folder contains disease-medication mappings used for recommendation.
* The `diseases-and-symptoms-dataset/` folder contains the Kaggle dataset mapping symptoms to diseases.

---

## ✨ Contribution

* Fork the repository
* Create a new branch for your feature/fix
* Submit a pull request

---

## 📄 License

This project is licensed under the MIT License.
