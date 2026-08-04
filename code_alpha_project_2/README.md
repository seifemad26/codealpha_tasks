# 🏥 Disease Prediction from Medical Data

A machine learning project that predicts the possibility of diseases based on patient medical data using multiple classification algorithms. Built as part of the **CodeAlpha Machine Learning Internship**.

\---

## 📌 Objective

Apply classification techniques to structured medical data to predict the likelihood of diseases including **Heart Disease**, **Diabetes**, and **Breast Cancer**.

\---

## 🗂️ Datasets

### 1\. Heart Disease Dataset (UCI)

|Property|Value|
|-|-|
|**Source**|[Kaggle](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) / [UCI](https://archive.ics.uci.edu/ml/datasets/heart+disease)|
|**Rows**|303|
|**Features**|13 (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)|
|**Target**|target (1 = Disease, 0 = No Disease)|

### 2\. Diabetes Dataset (UCI)

|Property|Value|
|-|-|
|**Source**|[Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)|
|**Rows**|768|
|**Features**|8 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)|
|**Target**|Outcome (1 = Diabetes, 0 = No Diabetes)|

### 3\. Breast Cancer Dataset (Built-in)

|Property|Value|
|-|-|
|**Source**|`sklearn.datasets.load\_breast\_cancer()`|
|**Rows**|569|
|**Features**|30 (mean radius, mean texture, mean perimeter, etc.)|
|**Target**|target (1 = Benign, 0 = Malignant)|

> \*\*Note:\*\* You only need to use \*\*one\*\* dataset to complete the internship task. The code supports all three for comprehensive analysis.

\---

## 🛠️ Tech Stack

* **Python 3.8+**
* **Pandas \& NumPy** — Data manipulation
* **Scikit-learn** — ML models, preprocessing, metrics
* **XGBoost** — Gradient boosting
* **LightGBM** — Fast gradient boosting
* **Matplotlib \& Seaborn** — Visualization
* **imbalanced-learn** — SMOTE for class imbalance

\---

## 🚀 Quick Start

### 1\. Clone the repository

```bash
git clone https://github.com/YOUR\_USERNAME/CodeAlpha\_DiseasePrediction.git
cd CodeAlpha\_DiseasePrediction
```

### 2\. Install dependencies

```bash
pip install -r requirements.txt
```

### 3\. Download datasets (optional — Breast Cancer loads automatically)

Place in project root:

* `heart.csv` — [Download](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
* `diabetes.csv` — [Download](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### 4\. Run the training script

```bash
python disease\_prediction.py
```

### 5\. Make predictions

```python
from predict\_disease import predict\_heart\_disease, predict\_diabetes, predict\_breast\_cancer

# Heart Disease
result = predict\_heart\_disease(
    age=55, sex=1, cp=2, trestbps=140, chol=240, fbs=0,
    restecg=1, thalach=150, exang=0, oldpeak=1.5,
    slope=1, ca=0, thal=2
)
print(result)
```

\---

## 🧠 Models Implemented

|Model|Type|Best For|
|-|-|-|
|**Logistic Regression**|Linear|Baseline, interpretability|
|**SVM (RBF \& Linear)**|Kernel-based|High-dimensional data|
|**Decision Tree**|Tree-based|Interpretable rules|
|**Random Forest**|Ensemble|Robust, feature importance|
|**Gradient Boosting**|Ensemble|High accuracy|
|**XGBoost**|Gradient Boosting|Best performance|
|**LightGBM**|Gradient Boosting|Speed \& efficiency|
|**Naive Bayes**|Probabilistic|Fast baseline|
|**K-Nearest Neighbors**|Instance-based|Simple, non-parametric|
|**Voting Ensemble**|Stacked|Combines top 3 models|

\---

## 📊 Evaluation Metrics

* **Accuracy** — Overall correctness
* **Precision** — True positives / Predicted positives
* **Recall** — True positives / Actual positives
* **F1-Score** — Harmonic mean of Precision \& Recall
* **ROC-AUC** — Area under ROC curve (best for medical diagnosis)

\---

## 📁 Project Structure

```
CodeAlpha\_DiseasePrediction/
├── disease\_prediction.py          # Main training script
├── predict\_disease.py             # Standalone prediction script
├── heart.csv                      # Heart disease dataset
├── diabetes.csv                   # Diabetes dataset
├── best\_heart\_disease\_model.pkl   # Saved heart disease model
├── best\_diabetes\_model.pkl        # Saved diabetes model
├── best\_breast\_cancer\_model.pkl   # Saved breast cancer model
├── ensemble\_heart\_disease.pkl     # Ensemble model
├── ensemble\_diabetes.pkl          # Ensemble model
├── ensemble\_breast\_cancer.pkl     # Ensemble model
├── scaler\_heart\_disease.pkl       # Feature scaler
├── scaler\_diabetes.pkl            # Feature scaler
├── scaler\_breast\_cancer.pkl       # Feature scaler
├── model\_comparison\_all\_datasets.png
├── roc\_curves\_all\_datasets.png
├── confusion\_matrices\_best\_models.png
├── feature\_importance\_all\_datasets.png
├── requirements.txt
├── README.md
└── LICENSE
```

\---

## 🎯 Example Predictions

### Heart Disease

```python
result = predict\_heart\_disease(
    age=55, sex=1, cp=2, trestbps=140, chol=240, fbs=0,
    restecg=1, thalach=150, exang=0, oldpeak=1.5,
    slope=1, ca=0, thal=2
)
# Output: {'prediction': 'Heart Disease Detected', 'risk\_level': 'MEDIUM',
#          'recommendation': 'Regular checkup recommended'}
```

### Diabetes

```python
result = predict\_diabetes(
    Pregnancies=2, Glucose=140, BloodPressure=80,
    SkinThickness=30, Insulin=120, BMI=32.5,
    DiabetesPedigreeFunction=0.45, Age=35
)
# Output: {'prediction': 'Diabetes Detected', 'risk\_level': 'MEDIUM',
#          'recommendation': 'Monitor glucose levels'}
```

\---

## 📹 Internship Submission

* **LinkedIn Post:** \[Your LinkedIn Post URL]
* **GitHub Repo:** `CodeAlpha\_DiseasePrediction`
* **Video Explanation:** \[Your Video URL]

\---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**. It should **NOT** be used for actual medical diagnosis. Always consult a qualified healthcare professional for medical advice.

\---

## 

