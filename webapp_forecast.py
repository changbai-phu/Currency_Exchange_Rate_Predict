import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
import itertools
import matplotlib.pyplot as plt


def prep_data():
    # Get last 6 months of daily data\
    df = yf.download("CADCNY=X", start="2025-02-01", end="2025-08-01", interval='1d')
    df.columns = ['Date', 'Rate']
    df.to_csv("cad_cny_history.csv", index=False)
    # Load historical data
    #df = pd.read_csv("cad_cny_history.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    print(df.tail())
    return df

def prophet_forecast(df):
    # Prepare data for Prophet
    df_prophet = df.reset_index()
    df_prophet.columns = ['ds', 'y']
    model_prophet = Prophet()
    model_prophet.fit(df_prophet)
    future_prophet = model_prophet.make_future_dataframe(periods=14)  # forecast 14 days
    forecast_propht = model_prophet.predict(future_prophet)
    prophet_forecast_values = forecast_propht['yhat'].tail(14).values
    print("Forecasted Rates for the next 14 days (Prophet):", prophet_forecast_values)
    return prophet_forecast_values

def find_best_arima_parameters(data, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
  """find the best arima parameters to minimize AIC"""
  best_aic = np.inf
  best_order = None
  best_model = None

  for p, d, q in itertools.product(
        range(p_range[0], p_range[1] + 1),
        range(d_range[0], d_range[1] + 1),
        range(q_range[0], q_range[1] + 1)
    ):
        try:
            model = ARIMA(data, order=(p, d, q))
            model_fit = model.fit()
            aic = model_fit.aic

            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
                best_model = model_fit

        except:
            continue

  return best_order, best_aic, best_model

def arima_forecast(df, order):
    # Fit ARIMA model
    model_arima = ARIMA(df['Rate'], order=order)
    model_arima_fit = model_arima.fit()
    forecast_arima = model_arima_fit.forecast(steps=14).values
    print("Forecasted Rates for the next 14 days (ARIMA):", forecast_arima)
    return forecast_arima

def lstm_forecast(df):
    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Rate'].values.reshape(-1, 1))
    sequence_length = 60
    X = []
    y = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model_lstm = Sequential()
    model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model_lstm.add(LSTM(units=50))
    model_lstm.add(Dense(units=1))
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X, y, epochs=100, batch_size=32)

    predictions_lstm = []
    last_60_days = scaled_data[-sequence_length:]
    for _ in range(14):
        x_input = last_60_days[-sequence_length:]
        x_input = np.reshape(x_input, (1, sequence_length, 1))
        prediction = model_lstm.predict(x_input)
        predictions_lstm.append(prediction[0][0])
        last_60_days = np.append(last_60_days[1:], prediction[0][0])

    lstm_forecast = scaler.inverse_transform(np.array(predictions_lstm).reshape(-1, 1))
    print("Forecasted Rates for the next 14 days (LSTM):", lstm_forecast.flatten())
    return lstm_forecast.flatten()

def print_plot_comparison(df, arima_forecast, prophet_forecast_values, lstm_forecast_values):
    plt.figure(figsize=(12, 7))
    plt.plot(df.index, df['Rate'], label='Historical Rates', color='blue')
    
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14, freq='D')
    
    comparison_df = pd.DataFrame({
        'Date': forecast_dates,
        'ARIMA': arima_forecast,
        'Prophet': prophet_forecast_values,
        'LSTM': lstm_forecast_values
    })

    print(comparison_df)

    plt.plot(forecast_dates, arima_forecast, label='ARIMA Forecast', color='orange')
    plt.plot(forecast_dates, prophet_forecast_values, label='Prophet Forecast', color='green')
    plt.plot(forecast_dates, lstm_forecast_values, label='LSTM Forecast', color='red')

    plt.title('Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate (CAD/CNY)')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Prepare data
    df = prep_data()

    # Forecast using ARIMA
    best_order, best_aic, best_model = find_best_arima_parameters(df['Rate'])
    print("Best (p,d,q) for ARIMA using AIC:", best_order)
    #manual_order = (5, 1, 0)  # Example ARIMA order
    forecast_arima = arima_forecast(df, best_order)

    # Forecast using Prophet
    prophet_forecast_values = prophet_forecast(df)

    # Forecast using LSTM
    lstm_forecast_values = lstm_forecast(df)
    
    # Print and combine predicted results in one table for convenience
    print_plot_comparison(df, forecast_arima, prophet_forecast_values, lstm_forecast_values)