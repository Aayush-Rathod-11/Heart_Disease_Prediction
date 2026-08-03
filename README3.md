# Heart_Disease_Prediction_ML

## Problem Statement

You are an ML Engineer at a tech company. Your task is to build, train, and evaluate a machine learning
model that predicts an outcome from real-world data. You must compare at least 3 different ML algorithms,
evaluate them using proper metrics, and clearly explain which model performed best and why.

### Dataset

- **Source:** [Heart Disease Dataset — Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Rows:** ~1025 patients (after deduplication)
- **Features:** 13 clinical features + 1 binary target
- **Target:** 1 = Heart disease present, 0 = No heart disease

### Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Google Colab

## Project Structure

### Steps Covered

1. Data Loading & Inspection
2. Data Cleaning (duplicates, type checks)
3. Train/Test Split (80/20, stratified)
4. Feature Engineering (correlation heatmap + RF feature importance)
5. Model Training — Logistic Regression, Random Forest, KNN
6. Evaluation — Accuracy, Precision, Recall, F1 Score
7. Comparison Table (all 3 models)
8. Best Model Analysis + Confusion Matrix
9. Conclusion

### Screenshort of Camparison table

<img width="552" height="231" alt="image" src="https://github.com/user-attachments/assets/09ca10dd-5b5a-4413-ad90-16dde5661db7" />

### Results Summary (Comparison table)

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.7705   | 0.7027    | 0.8966 | 0.7879   |
| Random Forest       | 0.8525   | 0.8125    | 0.8966 | 0.8525   |
| KNN (k=5)           | 0.7377   | 0.6970    | 0.7931 | 0.7419   |

**Best Model:** Random Forest with F1 Score of 0.8525 (85.25%)

### Most Surprising Finding

The most surprising finding was how strongly **chest pain type ('cp')** correlated
with heart disease presence — more than traditional risk factors like cholesterol
('chol') or blood pressure ('trestbps'), which are the ones most commonly discussed
in public health messaging. The feature importance chart made this especially clear.
It suggests that symptom-type may be a more powerful early indicator than lab values
alone, which has real implications for triage in clinical settings.

## Conclusion

After comparing three classification models on the Heart Disease dataset,
**Random Forest** emerged as the strongest performer with an F1 Score
of **0.8525 (85.25%)** and Accuracy of **0.8525 (85.25%)**.

Key insight: chest pain type ('cp'), maximum heart rate ('thalach'), and
number of major vessels ('ca') were the most influential features —
outweighing commonly assumed risk factors like cholesterol and blood pressure.

In a medical screening context, this model's low false negative rate makes
it practically useful, not just statistically strong.

### How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/drive/1MVxifMv6Kq10ub0UEXl6yiE3KLNnYhZQ?hl=eu#scrollTo=inJYw1FAmtU3)
2. Download the dataset from the Kaggle link above
3. Upload 'heart.csv' to Colab or mount Google Drive
4. Run all cells top to bottom (Runtime → Run All)

Or clone this repo and run locally with Jupyter:

```bash
git clone https://github.com/Aayush-Rathod-11/Heart_Disease_Prediction_ML.git
```

### Author

Aayush Rathod
