import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load dataset
df = pd.read_csv("data/processed/cleaned_supermarket_sales.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

# Sort by date
df = df.sort_values("Date")

# Create daily sales time series
sales = df.groupby("Date")["Sales"].sum()

print("===== Daily Sales =====")
print(sales.head())

# Plot Sales Trend
plt.figure(figsize=(10,5))
plt.plot(sales)
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# Moving Average
moving_avg = sales.rolling(window=7).mean()

plt.figure(figsize=(10,5))
plt.plot(sales, label="Actual Sales")
plt.plot(moving_avg, label="7-Day Moving Average", color="red")
plt.title("Moving Average")
plt.legend()
plt.grid(True)
plt.show()

# ARIMA Model
model = ARIMA(sales, order=(1,1,1))
model_fit = model.fit()

# Forecast next 7 days
forecast = model_fit.forecast(steps=7)

print("\n===== Forecast for Next 7 Days =====")
print(forecast)

# Forecast Plot
plt.figure(figsize=(10,5))
plt.plot(sales, label="Actual Sales")
plt.plot(forecast.index, forecast.values, label="Forecast", color="green")
plt.title("ARIMA Forecast")
plt.legend()
plt.grid(True)
plt.show()

# Accuracy (MAE)
predictions = model_fit.predict(start=1, end=len(sales)-1)
actual = sales.iloc[1:]

mae = mean_absolute_error(actual, predictions)

print("\nMean Absolute Error (MAE):", mae)