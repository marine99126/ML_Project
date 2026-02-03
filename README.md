# Titanic Survival Prediction (Machine Learning)

## Project Overview
This project predicts whether a passenger survived the Titanic disaster using machine learning classification models.  
The goal is to demonstrate an end-to-end ML workflow including data preprocessing, feature engineering, model training, and evaluation.

---

## Problem Statement
Given passenger information such as age, gender, ticket class, and fare, the task is to predict whether the passenger survived (`1`) or not (`0`).

---

## Dataset
- Dataset: Titanic dataset
- Source: Seaborn / Kaggle
- Target variable: `survived`
- Type: Binary classification problem

---

## Machine Learning Approach
- Data cleaning and missing value handling
- Categorical feature encoding
- Train-test split
- Models used:
  - Logistic Regression (baseline)
  - Random Forest Classifier
- Evaluation metrics:
  - Accuracy
  - Precision, Recall, F1-score

---

## Results
- Logistic Regression Accuracy: ~78%
- Random Forest Accuracy: ~82%
- Random Forest performed better due to its ability to capture non-linear relationships.

---

## How to Run the Project

1. Clone the repository
```bash
git clone https://github.com/your-username/titanic-ml-project.git
