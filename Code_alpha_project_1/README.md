# 💳 Credit Scoring Model

A machine learning project that predicts an individual's creditworthiness based on past financial data. Built as part of the **CodeAlpha Machine Learning Internship**.

\---

## 📌 Objective

Predict whether a loan applicant is a **Good Credit** or **Bad Credit** risk using classification algorithms and financial history features.

\---

## 🗂️ Dataset

|Property|Value|
|-|-|
|**Name**|German Credit Data|
|**Source**|[Kaggle - German Credit Data](https://www.kaggle.com/datasets/uciml/german-credit)|
|**Rows**|\~1,000|
|**Features**|9 (Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose)|
|**Target**|Credit Risk (1 = Good, 2 = Bad)|

> ⚠️ \*\*Note:\*\* The original UCI dataset contains 20 features. This project uses a simplified CSV version with 9 core features.

\---

## 🛠️ Tech Stack

* **Python 3.8+**
* **Pandas** — Data manipulation
* **NumPy** — Numerical computing
* **Scikit-learn** — ML models \& evaluation
* **XGBoost** — Gradient boosting
* **Matplotlib \& Seaborn** — Visualization
* **imbalanced-learn** — SMOTE for class imbalance

\---

## 🚀 Quick Start

### 1\. Clone the repository

```bash
git clone https://github.com/YOUR\_USERNAME/CodeAlpha\_CreditScoring.git
cd CodeAlpha\_CreditScoring
```

### 2\. Install dependencies

```bash
pip install -r requirements.txt
```

### 3\. Download the dataset

Place `german\_credit\_data\_updated.csv` in the project root.

### 4\. Run the model

```bash
python credit\_scoring.py
```

### 5\. Make predictions (without retraining)

```bash
python predict\_credit.py
```

\---

## 📊 Models Used

|Model|Accuracy|ROC-AUC|
|-|-|-|
|SVM|73.8%|**0.810**|
|Logistic Regression|73.8%|0.782|
|Random Forest|72.3%|0.779|
|XGBoost|71.2%|0.737|
|Decision Tree|63.9%|0.625|

> \*\*Best Model:\*\* SVM with ROC-AUC = 0.810

\---

## 📈 Evaluation Metrics

* **Accuracy** — Overall correctness
* **Precision** — Of predicted bad credits, how many were actually bad
* **Recall** — Of actual bad credits, how many did we catch
* **F1-Score** — Harmonic mean of Precision \& Recall
* **ROC-AUC** — Ability to distinguish between classes (best metric for imbalanced data)

\---

## 🧠 Key Features

* **Feature Engineering:** credit-to-duration ratio, is\_young, is\_high\_amount, is\_long\_duration
* **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Hyperparameter Tuning:** GridSearchCV on XGBoost
* **Model Persistence:** Save and load trained models with `joblib`

\---

## 📁 Project Structure

```
CodeAlpha\_CreditScoring/
├── credit\_scoring.py          # Main training script
├── predict\_credit.py          # Standalone prediction script
├── german\_credit\_data\_updated.csv  # Dataset
├── credit\_scoring\_model.pkl   # Saved trained model
├── credit\_scaler.pkl          # Saved feature scaler
├── credit\_label\_encoders.pkl  # Saved label encoders
├── model\_comparison.png       # Performance charts
├── roc\_curves.png             # ROC curves
├── feature\_importance.png     # Feature importance plot
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE                    # MIT License
```

\---

## 🎯 Example Prediction

```python
from predict\_credit import predict\_credit\_risk

result = predict\_credit\_risk(
    age=35, sex='male', job=2, housing='own',
    saving\_accounts='little', checking\_account='moderate',
    credit\_amount=5000, duration=24, purpose='car'
)

# Output:
# {'prediction': 'Good Credit', 'default\_probability': 23.45, 
#  'risk\_level': 'LOW RISK', 'recommendation': 'APPROVE'}
```

\---

## 

