import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("data/processed/cleaned_supermarket_sales.csv")

# Create Target Variable
# 1 = High Sales, 0 = Low Sales
df["HighSales"] = (df["Sales"] > df["Sales"].median()).astype(int)

# Features and Target
X = df[["Unit price", "Quantity", "Rating"]]
y = df["HighSales"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Logistic Regression
# -----------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

print("===== Logistic Regression =====")
print("Accuracy :", accuracy_score(y_test, log_pred))
print("Precision:", precision_score(y_test, log_pred))
print("Recall   :", recall_score(y_test, log_pred))
print("F1 Score :", f1_score(y_test, log_pred))

print("\nClassification Report")
print(classification_report(y_test, log_pred))

# -----------------------------
# Decision Tree
# -----------------------------
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

print("\n===== Decision Tree =====")
print("Accuracy :", accuracy_score(y_test, tree_pred))
print("Precision:", precision_score(y_test, tree_pred))
print("Recall   :", recall_score(y_test, tree_pred))
print("F1 Score :", f1_score(y_test, tree_pred))

print("\nClassification Report")
print(classification_report(y_test, tree_pred))

# -----------------------------
# Feature Importance
# -----------------------------
importance = pd.Series(
    tree.feature_importances_,
    index=X.columns
)

plt.figure(figsize=(7,4))
importance.sort_values().plot(kind="barh")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.grid(True)
plt.show()

print("\nTask 4 Prediction Model Completed Successfully.")