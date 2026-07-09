import pandas as pd
from scipy import stats

# Load the cleaned dataset
df = pd.read_csv("data/processed/cleaned_supermarket_sales.csv")

# -------------------------------
# Descriptive Statistics
# -------------------------------

print("===== Descriptive Statistics =====\n")

print(df.describe())

print("\nMean of Sales:", df["Sales"].mean())
print("Median of Sales:", df["Sales"].median())
print("Mode of Sales:", df["Sales"].mode()[0])
print("Variance of Sales:", df["Sales"].var())
print("Standard Deviation of Sales:", df["Sales"].std())
print("Skewness of Sales:", df["Sales"].skew())
print("Kurtosis of Sales:", df["Sales"].kurt())

print(df.columns)
# -------------------------------
# Hypothesis Testing (Male vs Female Spending)
# -------------------------------

male = df[df["Gender"] == "Male"]["Sales"]
female = df[df["Gender"] == "Female"]["Sales"]

t_stat, p_value = stats.ttest_ind(male, female)

print("\n===== Hypothesis Test =====")
print("T-Statistic:", t_stat)
print("P-Value:", p_value)

if p_value < 0.05:
    print("Result: Significant difference between Male and Female spending.")
else:
    print("Result: No significant difference between Male and Female spending.")

# -------------------------------
# Correlation Matrix
# -------------------------------

print("\n===== Correlation Matrix =====")
print(df.corr(numeric_only=True))