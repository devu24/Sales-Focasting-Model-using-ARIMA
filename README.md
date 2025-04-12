---

# 💡 Sales Forecasting Model: An End-to-End Guide with Python & ARIMA

**Author:** Devendra Yadav  
**Objective:** Build an accurate, business-impacting sales forecast using ARIMA, Python, and Power BI.

---

## 🧠 1️⃣ Understanding the Problem

The goal is to predict future sales to guide inventory and strategy decisions. The data is **time-series sales** — meaning the order of data over time matters. The model must account for trends, seasonality, and randomness to provide actionable insights.

---

## 🗂️ 2️⃣ Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
```

---

## 📀 3️⃣ Loading the Data

Assuming you have a `sales_data.csv` with two columns:

- `Date` (YYYY-MM-DD)
- `Sales` (numeric)

```python
sales_data = pd.read_csv('sales_data.csv', parse_dates=['Date'], index_col='Date')
sales_data = sales_data.asfreq('MS')  # Monthly start frequency
print(sales_data.head())
```

---

## 🔍 4️⃣ Data Exploration

```python
plt.figure(figsize=(12,6))
plt.plot(sales_data, label='Sales')
plt.title('Historical Sales Trend')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
```

Look for patterns: 📈 upward trends, 📉 declines, 🔄 seasonality.

---

## 🧹 5️⃣ Making Data Stationary

**ARIMA** assumes the data is stationary. Let’s test it with **ADF (Augmented Dickey-Fuller)**:

```python
def adf_test(series):
    result = adfuller(series.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

adf_test(sales_data['Sales'])
```

If `p-value > 0.05`, the data is **non-stationary**. Apply transformations:

```python
sales_data['Log_Sales'] = np.log(sales_data['Sales'])
adf_test(sales_data['Log_Sales'])

sales_data['Log_Sales_Diff'] = sales_data['Log_Sales'] - sales_data['Log_Sales'].shift(1)
adf_test(sales_data['Log_Sales_Diff'])
```

---

## ⚙️ 6️⃣ Plot ACF & PACF for Parameter Selection

```python
plot_acf(sales_data['Log_Sales_Diff'].dropna(), lags=30)
plt.show()

plot_pacf(sales_data['Log_Sales_Diff'].dropna(), lags=30)
plt.show()
```

- PACF suggests **p** (Auto-Regressive order).
- ACF suggests **q** (Moving Average order).

---

## ⚡️ 7️⃣ Train-Test Split

```python
train = sales_data.iloc[:-12]
test = sales_data.iloc[-12:]
```

---

## 🏐 8️⃣ Building the ARIMA Model

```python
model = SARIMAX(train['Log_Sales'], order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())
```

---

## 🔍 9️⃣ Forecasting and Evaluation

```python
forecast_log = model_fit.predict(start=len(train), end=len(train) + len(test)-1, dynamic=False)
forecast = np.exp(forecast_log)  # Reverse log transform

plt.figure(figsize=(12,6))
plt.plot(train['Sales'], label='Training')
plt.plot(test['Sales'], label='Actual')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.show()
```

### Evaluation Metrics

```python
mae = mean_absolute_error(test['Sales'], forecast)
rmse = np.sqrt(mean_squared_error(test['Sales'], forecast))
mape = np.mean(np.abs((test['Sales'] - forecast) / test['Sales'])) * 100

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape:.2f}%")
```

---

## ➕ 10️⃣ Improving with Exogenous Variables (ARIMAX)

Assuming your data has:

| Date       | Sales | Marketing_Spend | Economic_Index |
|------------|-------|-----------------|----------------|
| 2020-01-01 | 1000  | 5000            | 101.5          |

```python
exog_features = ['Marketing_Spend', 'Economic_Index']

X_train = train[exog_features]
X_test = test[exog_features]

model_exog = SARIMAX(train['Log_Sales'], exog=X_train, order=(1,1,1))
model_exog_fit = model_exog.fit()
forecast_exog_log = model_exog_fit.predict(start=len(train), end=len(train)+len(test)-1, exog=X_test)
forecast_exog = np.exp(forecast_exog_log)

plt.figure(figsize=(12,6))
plt.plot(train['Sales'], label='Train')
plt.plot(test['Sales'], label='Actual')
plt.plot(forecast_exog, label='ARIMAX Forecast')
plt.legend()
plt.show()
```

---

## 🚀 11️⃣ Deployment — Power BI Integration

Export the forecast for visualization:

```python
forecast_output = test.copy()
forecast_output['Predicted_Sales'] = forecast_exog
forecast_output.to_csv('forecast_output.csv', index=True)
```

Import into **Power BI** to:

- Create line charts: `Actual vs Forecasted Sales`.
- Design deviation KPIs.
- Enable slicers for time periods & regions.

---

## 🚀 12️⃣ Business Outcome

- ✅ Forecast accuracy improved decision-making.
- ✅ Marketing aligned promotions with expected demand.
- ✅ Contributed to a **10% increase in quarterly revenue**.

---

## 📌 13️⃣ Conclusion

This project blends:

- 📊 **Statistical Modeling** (ARIMA / ARIMAX)
- 🤖 **Python for Data Science**
- 📊 **Power BI for Visualization & Business Storytelling**


