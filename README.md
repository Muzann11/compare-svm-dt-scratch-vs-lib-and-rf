# Customer Churn Prediction using Supervised Learning

This repository presents a machine learning project that predicts **customer churn** using the **Telco Customer Churn** dataset. The goal is to classify whether a customer will stop using the service (churn) based on historical attributes and behavior. The project implements and compares five models: three from **Scikit-Learn** and two **manual implementations**.

---

## 🧠 Objective

* Build a machine learning model to predict customer churn.
* Compare classification models: **Decision Tree**, **Support Vector Machine (SVM)**, and **Random Forest**.
* Focus on **recall**, **precision**, and **F1-score** for the churn class (`1`) due to **class imbalance**.

---

## 📊 Dataset Overview

* **Source**: Kaggle – Telco Customer Churn Dataset
* **Size**: \~7,000 records
* **Target**: `Churn` column (`Yes` or `No`)

### ⚠️ Class Distribution

| Class   | Percentage |
| ------- | ---------- |
| **No**  | 73.46%     |
| **Yes** | 26.54%     |

> ⚠️ **Imbalanced Data Warning**:
> Due to majority class dominance, models tend to predict “non-churn”. Therefore, metrics for the minority class (**churn**) are the primary evaluation focus:
>
> * `Recall_1`: How many churns were detected?
> * `Precision_1`: How accurate are the churn predictions?
> * `F1_1`: Balance between recall and precision.

---

## 📈 Results Summary

| Model                            | Akurasi | Precision\_0 | Precision\_1 | Recall\_0 | Recall\_1 | F1\_0  | F1\_1    | Avg\_Precision | Avg\_Recall | Avg\_F1  |
| -------------------------------- | ------- | ------------ | ------------ | --------- | --------- | ------ | -------- | -------------- | ----------- | -------- |
| **Random Forest (Scikit-Learn)** | 0.7821  | 0.8297       | 0.6098       | 0.885     | 0.4973    | 0.8564 | 0.5478   | 0.72           | 0.69        | 0.7      |
| **SVM (Scikit-Learn)**           | 0.7879  | 0.84         | **0.62**     | 0.88      | **0.53**  | 0.86   | **0.57** | **0.73**       | **0.7**     | **0.71** |
| Decision Tree (Scikit-Learn)     | 0.7253  | 0.82         | 0.48         | 0.81      | 0.5       | 0.81   | 0.49     | 0.65           | 0.65        | 0.65     |
| Decision Tree (Manual)           | 0.7139  | 0.8009       | 0.4596       | 0.8125    | 0.4411    | 0.8067 | 0.4502   | 0.63           | 0.63        | 0.63     |
| SVM (Manual)                     | 0.6025  | 0.7141       | 0.19         | 0.7652    | 0.1524    | 0.7388 | 0.1691   | 0.45           | 0.46        | 0.45     |

---

## 🏆 Insights & Model Comparison

* 🔍 **SVM (Scikit-Learn)** shows the best overall performance in detecting churn with the highest `Recall_1` and `F1_1`.
* 🌲 **Random Forest (Scikit-Learn)** performs reasonably well and is a strong second-best alternative.
* 🌳 **Decision Tree (Scikit-Learn)** is acceptable but lacks accuracy and balance.
* 🛠️ **Manual Implementations (Decision Tree & SVM)** perform worse due to limited flexibility and lack of optimization features.

### 🧪 Why Scikit-Learn is Better?

Manual models suffer from:

* No automatic hyperparameter tuning
* Limited handling of nonlinearities and complex patterns
* Less optimized for imbalanced data scenarios

✅ **Scikit-learn models** are:

* Easier to optimize
* Better at generalizing
* More robust for real-world datasets

---

## 🔧 Tools and Libraries

* Python 3.x
* Pandas, NumPy
* Scikit-learn
* Matplotlib & Seaborn
* KaggleHub (for dataset download)
