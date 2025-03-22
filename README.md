# Credit Card Fraud Detection

This project aims to develop a classification model to detect fraudulent credit card transactions using machine learning techniques. The dataset contains various features, including transaction amount, merchant details, timestamps, and other features that help in identifying fraudulent activities.

## Project Overview

In this project, we use a **Random Forest Classifier** to detect fraud and address class imbalance using techniques like **SMOTE** (Synthetic Minority Over-sampling Technique). The model is trained on a dataset of credit card transactions, with the goal of identifying fraudulent transactions (Class 1) and non-fraudulent transactions (Class 0).

## Dataset

The dataset used in this project is from **[Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)**.

### Columns in the Dataset:
- `Time`: Time elapsed since the first transaction.
- `V1-V28`: Anonymized features (features such as transaction amounts, merchant details, etc.).
- `Amount`: The transaction amount.
- `Class`: Target variable (1: Fraud, 0: Non-fraud).

## Steps Performed

1. **Data Preprocessing**:
   - Handle missing values.
   - Drop unnecessary features like `Time`.
   - Scale the features using `StandardScaler`.
   
2. **Class Imbalance**:
   - Use **SMOTE** to balance the dataset by generating synthetic samples for the minority class (fraudulent transactions).

3. **Model Development**:
   - **Random Forest Classifier** was used to train the model.
   - Evaluation metrics like **Accuracy**, **ROC-AUC**, **Confusion Matrix**, and **F1-Score** were used for evaluation.

4. **Evaluation**:
   - Evaluate the model using cross-validation and performance metrics to ensure high accuracy while minimizing false positives.

## Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
