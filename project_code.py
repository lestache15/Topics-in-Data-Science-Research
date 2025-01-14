import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
# Final Project
# Topics in Data Science
# Leslie Dawn

#  Monthly Highest Temperatures vs Avg Humidity in South Georgia (2018-2024)
# Time series analyzing methods: Seasonal Decomposition using STL & ARIMA Forecasting

df = pd.read_csv('dataset_.csv', parse_dates=['Date'], date_format='%b-%y')
df.set_index('Date', inplace=True)

#relationship between humidity and temperature
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Highest Temperature'], label='Highest Temperature')
plt.plot(df.index, df['Avg Humidity'], label='Avg Humidity', color='green')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Highest Temperature and Avg Humidity Over Time')
plt.legend()
plt.show()

#     Seasonal Decomposition using STL
df['Deviation of Highest Temperature'] = df['Highest Temperature'] - df['Highest Temperature'].mean()
df['Deviation of Avg Humidity'] = df['Avg Humidity'] - df['Avg Humidity'].mean()

#ACF Plot to determine the period of potential seasonal component
plt.figure(figsize=(10, 6))
plot_acf(df['Deviation of Highest Temperature'].dropna(), lags=20)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('ACF of Deviation of Highest Temperature')
plt.grid(True)
plt.show()
#seasonal component is 12

humidity_stl = STL(df['Deviation of Avg Humidity'], period=12)
humidity_result = humidity_stl.fit()
humidity_result.plot()
plt.show()
humidity_trend = humidity_result.trend
humidity_seasonal = humidity_result.seasonal

stl = STL(df['Deviation of Highest Temperature'], period=12)
result = stl.fit()
result.plot()
plt.show()
trend = result.trend
seasonal = result.seasonal

#original data, trend, and seasonal components for humidity
plt.figure(figsize=(10, 6))
plt.plot(df['Avg Humidity'], label='Original Humidity Data')
plt.plot(humidity_trend, label='Humidity Trend')
plt.plot(humidity_seasonal, label='Humidity Seasonal')
plt.xlabel('Year')
plt.ylabel('Avg Humidity')
plt.title('Original Humidity Data, Trend, and Seasonal Components')
plt.legend()
plt.show()

#original data, trend, and seasonal components for highest temperature
plt.figure(figsize=(10, 6))
plt.plot(df['Highest Temperature'], label='Original Temperature Data')
plt.plot(trend, label='Temperature Trend')
plt.plot(seasonal, label='Temperature Seasonal')
plt.xlabel('Year')
plt.ylabel('Highest Temperature')
plt.title('Original Temp Data, Trend, and Seasonal Components')
plt.legend()
plt.show()

#     ARIMA forecasting
df['Highest Temperature Diff'] = df['Highest Temperature'].diff().dropna()
df['Avg Humidity Diff'] = df['Avg Humidity'].diff().dropna()
#Determine the order of differencing to achieve stationarity
#Check if the original series is stationary
print('Original Series: ')
result = adfuller(df['Highest Temperature'].dropna())
print('ADF Statistic: ', result[0])
print('p-value: ', result[1])

print('First Order Differencing: ')
result = adfuller(df['Highest Temperature Diff'].dropna())
print('ADF Statistic: ', result[0])
print('p-value: ', result[1])
#No need for second differencing due to first being stationary

#ACF for first differencing to determine p and q
plt.figure(figsize=(10, 6))
plot_acf(df['Highest Temperature Diff'].dropna(), lags=20)
plt.title('ACF for First Order Differencing')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.show()

#determine optimal value of q
q_values = range(0, 5)
aic_scores = []
p = 12  #based on lags in ACF plot, every 12 months
d = 1  #differencing result

for q in q_values:
    model = ARIMA(df['Highest Temperature'], exog=df[['Avg Humidity']], order=(p, d, q))
    results = model.fit()
    aic = results.aic
    aic_scores.append(aic)

print("\nAIC Scores:")
for q, aic in zip(q_values, aic_scores):
    print(f"ARIMAX(p={p}, d={d}, q={q}) - AIC: {aic}")

optimal_q = q_values[aic_scores.index(min(aic_scores))]
print(f"\nOptimal q value based on AIC: {optimal_q}")
q = optimal_q

model = ARIMA(df['Highest Temperature'], exog=df[['Avg Humidity']], order=(p, d, q))
results = model.fit()

forecast_steps = 30
exog_forecast = df[['Avg Humidity']].iloc[-forecast_steps:]  #using the last exogenous variable for forecast
forecast_results = results.get_forecast(steps=forecast_steps, exog=exog_forecast, alpha=0.2)
forecast_values = forecast_results.predicted_mean
confidence_intervals = forecast_results.conf_int()

plt.figure(figsize=(10, 6))
plt.plot(df['Highest Temperature'], label='Original Time Series')
plt.plot(results.fittedvalues, color='red', label='Fitted Values')
plt.plot(forecast_values.index, forecast_values, linestyle='--', color='red', label='Forecasted Values')
plt.fill_between(forecast_values.index,
                 confidence_intervals.iloc[:, 0],
                 confidence_intervals.iloc[:, 1], color='red', alpha=0.2,
                 label='80% Confidence Interval')
plt.title('ARIMAX Forecast with 80% Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Highest Temperature')
plt.legend()
plt.show()

print("Forecasted Values:\n", forecast_values)
print("\nConfidence Intervals:\n", confidence_intervals)