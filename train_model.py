"""
train_model.py
----------------
Reproduces the exact pipeline from Heart_Disease_Prediction.ipynb
(load -> clean -> split -> scale -> train 3 models -> evaluate -> pick best)
and exports everything the Streamlit app needs into the model/ folder:

    model/best_model.pkl        -> the winning classifier (by F1 Score)
    model/scaler.pkl            -> the StandardScaler fit on training data
    model/comparison_table.csv  -> Accuracy / Precision / Recall / F1 for all 3 models
    model/feature_importance.csv-> Random Forest feature importances
    model/confusion_matrix.csv  -> confusion matrix of the best model
    model/metadata.json         -> best model name, feature order, class labels

Run this once (locally or in Colab) before launching the Streamlit app:
    python train_model.py
"""

import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

DATA_PATH = "heart.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- 1. Load & clean (same as the notebook) ----------
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()

X = df.drop("target", axis=1)
y = df["target"]
feature_order = list(X.columns)

# ---------- 2. Train/test split ----------
# NOTE: the notebook runs the split twice (cell 18 with stratify=y, then cell 19
# without stratify) and the second call overwrites the first, so the split that
# actually feeds the models has no stratify. Matching that here so results line
# up exactly with the notebook / README numbers.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 3. Scale (fit on train only) ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 4. Train the 3 models ----------
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_preds = knn_model.predict(X_test_scaled)

models = {
    "Logistic Regression": (lr_model, lr_preds),
    "Random Forest": (rf_model, rf_preds),
    "KNN (k=5)": (knn_model, knn_preds),
}

# ---------- 5. Evaluate & build comparison table ----------
rows = []
for name, (_, preds) in models.items():
    rows.append(
        {
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds), 4),
            "Recall": round(recall_score(y_test, preds), 4),
            "F1 Score": round(f1_score(y_test, preds), 4),
        }
    )
results = pd.DataFrame(rows).set_index("Model")
print("Model Comparison:\n", results, "\n")

# ---------- 6. Pick the best model by F1 Score ----------
best_model_name = results["F1 Score"].idxmax()
best_model, best_preds = models[best_model_name]
print(f"Best model: {best_model_name} (F1 = {results.loc[best_model_name, 'F1 Score']})")

# ---------- 7. Feature importance (Random Forest, for the insights tab) ----------
rf_importances = pd.Series(
    rf_model.feature_importances_, index=feature_order
).sort_values(ascending=False)

# ---------- 8. Confusion matrix for the best model ----------
cm = confusion_matrix(y_test, best_preds)

# ---------- 9. Save everything ----------
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
results.to_csv(os.path.join(MODEL_DIR, "comparison_table.csv"))
rf_importances.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), header=["importance"])
pd.DataFrame(cm, index=["Actual: No Disease", "Actual: Disease"],
             columns=["Predicted: No Disease", "Predicted: Disease"]).to_csv(
    os.path.join(MODEL_DIR, "confusion_matrix.csv")
)

metadata = {
    "best_model_name": best_model_name,
    "feature_order": feature_order,
    "class_labels": {"0": "No Disease", "1": "Disease"},
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved model artifacts to '{MODEL_DIR}/'. You can now run: streamlit run app.py")
