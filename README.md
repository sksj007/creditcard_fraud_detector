# 💳 Credit Card Fraud Detection

This project applies machine learning techniques to detect fraudulent credit card transactions. Using the well-known `creditcard.csv` dataset, we build a logistic regression model and handle class imbalance with SMOTE. The model is evaluated using metrics like F1 score, confusion matrix, and cross-validation.

## 📌 Features
- 📊 Data preprocessing and feature scaling (StandardScaler)
- ⚖️ SMOTE oversampling for imbalanced dataset
- 🤖 Logistic Regression model
- 🧪 Train-test split and evaluation
- ✅ Metrics: F1 Score, Classification Report, Confusion Matrix, Cross-validation
- 📈 ROC Curve Visualization

## 🧠 Dataset
- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Features: 30 anonymized input variables (`V1`–`V28`, `Time`, `Amount`)
- Target: `Class` (0 = Legitimate, 1 = Fraudulent)

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection

