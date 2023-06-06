import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
from pprint import pprint as print
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import time

import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


batch_size = 8
epochs = 1
last_loss = 1
desired_loss = 0.002

# Fetch tickers
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
days = "1000"
interval = "daily"
filename = f"btc_{days}_days_{interval}_interval"
# check if the file exists
try:
    with open(f"{filename}.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": interval,
    }
    response = requests.get(url, params=params)
    data = response.json()
    # save the data
    with open(f"{filename}.json", "w") as f:
        json.dump(data, f)


# Extract timestamps and prices
timestamps = [timestamp[0] / 1000 for timestamp in data['prices'][::-1]]  # Convert milliseconds to seconds
prices = [price[1] for price in data['prices'][::-1]]
market_cap = [market_cap[1] for market_cap in data['market_caps'][::-1]]
total_volume = [total_volume[1] for total_volume in data['total_volumes'][::-1]]

# reorder data
timestamps = timestamps[::-1]
prices = prices[::-1]
market_cap = market_cap[::-1]
total_volume = total_volume[::-1]


prices = np.array(prices)
total_volumes = np.array(total_volume)
timestamps = np.array(timestamps)

# Normalize data
normalized_prices = (prices - np.mean(prices)) / np.std(prices)
normalized_volumes = (total_volumes - np.mean(total_volumes)) / np.std(total_volumes)

# Combine prices and volumes into one input vector
input_data = np.column_stack((normalized_prices, normalized_volumes))

# Split data into train and test sets
train_ratio = 0.7
train_samples = int(len(input_data) * train_ratio)

train_data = input_data[:train_samples]
test_data = input_data[train_samples:]

# Prepare training data
window_size = 7
X_train = []
y_train = []
for i in range(len(train_data) - window_size):
    X_train.append(train_data[i:i+window_size])
    y_train.append(train_data[i+window_size][0])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Prepare test data
X_test = []
y_test = []
for i in range(len(test_data) - window_size):
    X_test.append(test_data[i:i+window_size])
    y_test.append(test_data[i+window_size][0])
X_test, y_test = np.array(X_test), np.array(y_test)

try:
    with open(f"{filename}.h5", "rb") as f:
        model = tf.keras.models.load_model(f'{filename}.h5')

except FileNotFoundError:
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(window_size, 2)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    time_start = time.time()
    time_deadline = 30  # seconds
    a = model.fit(X_train, y_train, epochs=1, batch_size=batch_size)
    while a.history['loss'][-1] > desired_loss and time.time() - time_start < time_deadline:
        last_loss = a.history['loss'][-1]
        a = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        if a.history['loss'][-1] > last_loss*1.02:
            batch_size = min(batch_size * 2, 4096)
            log.info(f"Batch size: {batch_size}")

    model.save(f'{filename}.h5')
        

# Predict the next value
# Prepare test data
# X_test = input_data[-window_size:].reshape(1, window_size, 2)

# Predict the next value
y_pred = model.predict(X_test)

# Denormalize the predicted value
y_pred = (y_pred * np.std(prices)) + np.mean(prices)

# Denormalize y_test
y_test = (y_test * np.std(prices)) + np.mean(prices)

# Print the predicted value
# print(f"Predicted value: {predicted_value[0][0]}")

# Plotting the results
# plt.plot(prices, label='Actual Prices')
# plt.plot(prices, 'bo', markersize=2)
# plt.plot(len(prices) - 1 + window_size, predicted_value, 'ro', markersize=5,
# label='Predicted Value')

# Plotting the data
fig, ax = plt.subplots()
ax.plot(timestamps[:train_samples], prices[:train_samples], label='Training Data')
ax.plot(timestamps[train_samples+window_size:], y_test, label='Ground Truth')
ax.plot(timestamps[train_samples+window_size:], y_pred, label='Predictions')

# Format the x-axis as human-readable date/time
#ax.xaxis.set_major_locator(mdates.AutoDateLocator())
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

# Rotate and align the x-axis labels for better visibility
#fig.autofmt_xdate()

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('BTC Price Prediction')
plt.legend()
plt.show()

start = 1000
eur = start
btc = 0
ctr = 0

for i in range(0, len(y_test)-10):
    log.debug(f"Predicted: {y_pred[i]} Actual: {y_test[i]}")
    ctr += 1
    if y_pred[i+1] > y_test[i] * 1.01 and eur > 10:
        log.debug("Buy")
        amount = 10 / y_test[i]
        btc += amount
        eur -= 10
        log.info(f"day {ctr}, EUR: {eur} BTC: {btc:.5f}, BUY")

    if y_pred[i+1] < y_test[i] * 0.99 and btc > 0.0001:
        log.debug("Sell")
        amount = 10 / y_test[i]
        # amount = btc / 4
        eur += amount * y_test[i]
        btc -= amount

        log.info(f"day {ctr}, EUR: {eur} BTC: {btc:.5f}, SELL")

    else:
        log.info(f"day {ctr}, EUR: {eur} BTC: {btc:.5f}, HOLD")

amount = btc * y_test[-1]
eur += amount
log.debug(f"Sell {btc} BTC for {amount} EUR")
btc = 0

log.info(f"EUR: {eur} BTC: {btc}")
log.info(f"Profit: {(eur - start) * 100 / start:.1f}% in {ctr} days")
# print 2 floating point after dot