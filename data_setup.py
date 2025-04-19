import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


path = r'C:\Users\chang\Projects\Currency_Exchange_Rate_Predict'

# Download historical data for CAD/CNY
symbols = ['CADCNY=X', 'CADUSD=X', 'CADEUR=X']
data = yf.download(symbols, start='2020-01-01', end='2025-01-01')
print(data.head())

#data.to_csv(path+'\currency_data.csv')

# check if missing data 
print(data.isnull().sum())
data.fillna(data.mean(), inplace=True) # fill missing with mean value

# normalize the dataset 
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close']])
# Convert back to DataFrame
data['Close_scaled'] = scaled_data
print(data.head())

plt.figure(figsize=(12, 6))

'''
# Plot with custom colors and styles
plt.plot(data['Close']['CADCNY=X'], label='CAD/CNY', color='blue', linestyle='--')
plt.plot(data['Close']['CADUSD=X'], label='CAD/USD', color='green', linestyle='-')
plt.plot(data['Close']['CADEUR=X'], label='CAD/EUR', color='red', linestyle='-.')
'''
# Plot with moving average
plt.plot(data['Close']['CADCNY=X'].rolling(window=30).mean(), label='CAD/CNY (30-day MA)', color='blue')
plt.plot(data['Close']['CADUSD=X'].rolling(window=30).mean(), label='CAD/USD (30-day MA)', color='green')
plt.plot(data['Close']['CADEUR=X'].rolling(window=30).mean(), label='CAD/EUR (30-day MA)', color='red')


# Add labels and legend
plt.title('CAD Exchange Rate Against CNY, USD, and EUR')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()
plt.grid()

plt.show()