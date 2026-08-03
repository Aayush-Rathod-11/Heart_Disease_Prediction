# 🫀 Heart Disease Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-FF7A00?style=for-the-badge" alt="Machine Learning"/>
  <img src="https://img.shields.io/badge/Streamlit-31333F?style=for-the-badge&logo=streamlit&logoColor=FF4B4B" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Deployed-E63946?style=for-the-badge" alt="Deployed"/>
</p>

<p align="center">
  <a href="https://heartdiseaseprobability.streamlit.app">
    <img src="https://img.shields.io/badge/🚀%20LAUNCH%20LIVE%20APP-STREAMLIT-E63946?style=for-the-badge&logo=streamlit&logoColor=white" alt="Launch Live App"/>
  </a>
</p>

---

## 📌 Project Overview

**Heart Disease Prediction** is an end-to-end Machine Learning project that predicts whether a patient is likely to have heart disease based on clinical, demographic, and diagnostic information.

The model is deployed as an interactive **Streamlit Web Application**, allowing users to enter patient details and receive an instant heart disease risk prediction along with the probability score.

---

## 🌐 Live Demo

<p align="center">
  <a href="https://heartdiseaseprobability.streamlit.app">
    <img src="https://img.shields.io/badge/🔗%20OPEN%20LIVE%20APPLICATION-STREAMLIT-E63946?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open Live Application"/>
  </a>
</p>

---

## ✨ Key Features

- 🫀 Heart Disease Risk Prediction
- 📊 Disease Probability & Confidence Score
- ⚠️ Patient Risk Assessment
- 🧠 3-Model Comparison (Logistic Regression, Random Forest, KNN)
- 📈 Feature Importance & Model Insights Dashboard
- 🖥️ Interactive Streamlit Dashboard
- 🔬 End-to-End Machine Learning Pipeline

---

## 📊 Dataset Features

### 👤 Patient Information

- Age
- Sex
- Chest Pain Type
- Fasting Blood Sugar
- Resting ECG Results
- Exercise-Induced Angina

### ❤️ Clinical & Diagnostic Information

- Resting Blood Pressure
- Serum Cholesterol
- Maximum Heart Rate Achieved
- ST Depression (Oldpeak)
- Slope of Peak Exercise ST Segment
- Number of Major Vessels (Fluoroscopy)
- Thalassemia

---

## 🧪 Model Performance

| Model               | Accuracy   | Precision  | Recall     | F1 Score   |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.7705     | 0.7027     | 0.8966     | 0.7879     |
| **Random Forest**   | **0.8525** | **0.8125** | **0.8966** | **0.8525** |
| KNN (k=5)           | 0.7377     | 0.6970     | 0.7931     | 0.7419     |

**🏆 Best Model:** Random Forest — selected by F1 Score (0.8525), balancing precision and recall in a medical screening context.

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · Streamlit · Matplotlib · Seaborn

---

## 📂 Project Structure

```
Heart_Disease_Prediction_ML/
├── app.py                    # Streamlit web application
├── train_model.py            # Model training & artifact export pipeline
├── Comparison table.png      # Comparison table Image
├── heart.csv                 # Dataset
├── requirements.txt          # Dependencies
├── Heart_Disease_Prediction.ipynb   # Full analysis notebook
└── model/
    ├── best_model.pkl
    ├── scaler.pkl
    ├── metadata.json
    ├── comparison_table.csv
    ├── feature_importance.csv
    └── confusion_matrix.csv
```

---

## ▶️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Aayush-Rathod-11/Heart_Disease_Prediction_ML.git
cd Heart_Disease_Prediction_ML

# Install dependencies
pip install -r requirements.txt

# (Optional) retrain the model
python train_model.py

# Launch the app
streamlit run streamlit_app.py
```

---

## 👤 Author

**Aayush Rathod**
