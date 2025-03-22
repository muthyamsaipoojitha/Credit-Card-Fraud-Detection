# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

# Step 1: Load the dataset from the specified path with a different encoding
file_path = r'C:\Users\saira\Downloads\Credit-Card-Fraud-Detection\data\creditcard.csv'  # Update with the correct file path

# Try loading the file with ISO-8859-1 encoding
try:
    data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Using ISO-8859-1 encoding
    print("Dataset Loaded Successfully with ISO-8859-1 encoding")
except UnicodeDecodeError:
    # If ISO-8859-1 doesn't work, try utf-16 encoding
    data = pd.read_csv(file_path, encoding='utf-16')
    print("Dataset Loaded Successfully with UTF-16 encoding")

# Step 2: Check the column names in the dataset to identify the correct target column
print("Columns in the dataset:")
print(data.columns)  # Display the column names

# Step 3: Print the first few rows to inspect the data and check for the target column
print("Dataset Head:")
print(data.head())

# Step 4: Update the target column based on the correct column name
target_column = 'Class'  # 'Class' is assumed to be the target column for fraud detection (0: Non-fraud, 1: Fraud)

# Step 5: Handle missing values if 'Class' is found
if target_column in data.columns:
    data[target_column] = data[target_column].fillna(data[target_column].mean())  # Replace missing fraud data with the mean value
else:
    print(f"Column '{target_column}' not found in the dataset. Please check the column name.")

# Step 6: Drop non-numeric columns for feature engineering
# Remove columns that are not useful for modeling like 'Time' or any other non-numeric columns if applicable
data = data.drop(['Time'], axis=1)  # For example, if 'Time' is not useful

# Step 7: Prepare the feature set (X) and target variable (y)
X = data.drop(target_column, axis=1)  # Independent variables (all columns except 'Class')
y = data[target_column]  # Target variable (fraud detection)

# Step 8: Feature Scaling
scaler = StandardScaler()  # Ensure StandardScaler is imported
X_scaled = scaler.fit_transform(X)

# Step 9: Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 10: Model Selection and Training
# Using Random Forest Classifier (you can use other models like Logistic Regression, XGBoost, etc.)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 11: Model Evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Accuracy, Precision, Recall, F1-Score
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy}")
print(f"ROC-AUC Score: {roc_auc}")

# Step 12: Visualization (Confusion Matrix Heatmap)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Optional: Cross-validation (for model evaluation)
cross_val_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean cross-validation score: {cross_val_scores.mean()}")
