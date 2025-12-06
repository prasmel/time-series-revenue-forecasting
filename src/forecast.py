import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("data/Month_Value_1.csv")
df['Period'] = pd.to_datetime(df['Period'], format='%d.%m.%Y')
df = df.sort_values('Period')
df.set_index('Period', inplace=True)

ts = df['Revenue'].ffill()

train_size = int(len(ts) * 0.8)
train_ts = ts.iloc[:train_size]
test_ts = ts.iloc[train_size:]

auto_model = auto_arima(
    train_ts,
    seasonal=True,
    m=12,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)

sarima_model = SARIMAX(
    train_ts,
    order=auto_model.order,
    seasonal_order=auto_model.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_results = sarima_model.fit(disp=False)
sarima_forecast = sarima_results.get_forecast(steps=len(test_ts))
sarima_predicted_mean = sarima_forecast.predicted_mean

prophet_train_df = train_ts.reset_index()
prophet_train_df.columns = ['ds', 'y']

prophet_model = Prophet(
    seasonality_mode='additive',
    yearly_seasonality=True,
    interval_width=0.95
)

prophet_model.fit(prophet_train_df)

future = prophet_model.make_future_dataframe(periods=len(test_ts), freq='MS')
prophet_forecast_df = prophet_model.predict(future)

prophet_predicted_mean = prophet_forecast_df['yhat'].iloc[-len(test_ts):]
prophet_predicted_mean.index = test_ts.index

results = pd.DataFrame({
    'Model': ['Auto-SARIMA', 'Prophet'],
    'RMSE': [
        np.sqrt(mean_squared_error(test_ts, sarima_predicted_mean)),
        np.sqrt(mean_squared_error(test_ts, prophet_predicted_mean))
    ],
    'MAE': [
        mean_absolute_error(test_ts, sarima_predicted_mean),
        mean_absolute_error(test_ts, prophet_predicted_mean)
    ]
})

results.to_csv("outputs/metrics.csv", index=False)
print(results)

plt.figure(figsize=(16, 7))
plt.plot(train_ts.index, train_ts.values, label='Training')
plt.plot(test_ts.index, test_ts.values, label='Actual')
plt.plot(sarima_predicted_mean.index, sarima_predicted_mean.values, linestyle='--', label='Auto-SARIMA')
plt.plot(prophet_predicted_mean.index, prophet_predicted_mean.values, linestyle=':', label='Prophet')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/forecast_comparison.png")
plt.show()
