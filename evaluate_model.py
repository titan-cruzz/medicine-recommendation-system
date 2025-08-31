import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load model + encoder
model = load_model("models/medicine_model.h5")
le = joblib.load("models/label_encoder.pkl")

# Load dataset
df = pd.read_csv("diseases-and-symptoms-dataset/Diseases_and_Symptoms.csv")
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Train-test split (same as training script)
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Encode labels
y_test_encoded = le.transform(y_test_raw)
y_test = to_categorical(y_test_encoded)

# Integer labels for evaluation
y_test_int = np.argmax(y_test, axis=1)

# Predictions
y_pred_prob = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_prob, axis=1)

# Accuracy
acc = accuracy_score(y_test_int, y_pred_classes)
print(f"Accuracy: {acc*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test_int, y_pred_classes)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=False, cmap="Blues")

# Show labels at intervals
step = 25
plt.xticks(
    ticks=np.arange(0, len(le.classes_), step),
    labels=le.classes_[::step],
    rotation=90, fontsize=8
)
plt.yticks(
    ticks=np.arange(0, len(le.classes_), step),
    labels=le.classes_[::step],
    rotation=0, fontsize=8
)

# Save results
os.makedirs("result", exist_ok=True)
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig("result/confusion_matrix_normalized.png", dpi=400, bbox_inches="tight")
plt.show()
