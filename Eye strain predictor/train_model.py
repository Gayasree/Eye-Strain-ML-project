"""
train_model.py - Load data, train all models, save the best one.
Run once: python train_model.py
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ── Load dataset ─────────────────────────────────────────────
df = pd.read_csv("eye_strain_dataset.csv")

categorical_cols = [
    "Gender",
    "Primary_Device",
    "Screen_Brightness",
    "Break_Frequency",
    "Blue_Light_Filter",
    "Eye_Symptom"
]

# ── Encode categorical columns ─────────────────────────────
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ── Encode target ─────────────────────────────
target_encoder = LabelEncoder()
df["Eye_Strain_Risk"] = target_encoder.fit_transform(df["Eye_Strain_Risk"])

# ── Split features and target ─────────────────────────────
X = df.drop(columns=["Eye_Strain_Risk"]).values
y = df["Eye_Strain_Risk"].values

# ── MANUAL TRAIN-TEST SPLIT ─────────────────────────────
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# ── MANUAL ACCURACY FUNCTION ─────────────────────────────
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# ── INBUILT ML MODELS ─────────────────────────────
candidates = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", random_state=42)
}

# ── Train & evaluate ─────────────────────────────
best_name, best_acc, best_model = None, -1, None

for name, model in candidates.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy(y_test, y_pred)

    print(f"{name:20s} Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_name, best_acc, best_model = name, acc, model

# ── Best model result ─────────────────────────────
print(f"\nBest Model: {best_name}  ({best_acc:.4f})")

# ── Save model ─────────────────────────────
with open("best_eye_strain_model.pkl", "wb") as f:
    pickle.dump({
        "model": best_model,
        "encoders": encoders,
        "target_encoder": target_encoder,
        "model_name": best_name
    }, f)

print("Saved: best_eye_strain_model.pkl")