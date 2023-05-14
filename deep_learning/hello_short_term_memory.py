import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import time
import logging 

logging.getLogger('tensorflow').disabled = True
logging.getLogger('tensorflow').setLevel(logging.ERROR)
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

from pprint import pprint as print


# Generate sinusoid data
time_stamps = np.arange(0, 50, 0.001)

# Generate sinusoidal wave
amplitude_sin = np.sin(time_stamps)

# Generate Gaussian wave
amplitude_gaussian = np.exp(-(time_stamps - 10) ** 2 / 16)

# Generate sawtooth wave
amplitude_sawtooth = 2 * (time_stamps % 4) / 4 - 1

# Combine the waves
amplitude = amplitude_sin + amplitude_gaussian + amplitude_sawtooth
# add noise
amplitude += np.random.normal(size=amplitude.shape) / 100
data = amplitude.reshape(-1, 1)

# Split data into train and test sets
train_ratio = 0.5
train_samples = int(len(data) * train_ratio)

train_data = data[:train_samples]
test_data = data[train_samples:]

# Prepare training data
window_size = 2  # Number of time steps to look back
X_train = []
y_train = []
for i in range(len(train_data) - window_size):
    X_train.append(train_data[i:i+window_size])
    y_train.append(train_data[i+window_size])
X_train, y_train = np.array(X_train), np.array(y_train)

print(f"X_train: {X_train}")
print(f"y_train: {y_train}")
print(f"X_train.shape: {X_train.shape}")

# Prepare test data
X_test = []
y_test = []
for i in range(len(test_data) - window_size):
    X_test.append(test_data[i:i+window_size])
    y_test.append(test_data[i+window_size])
X_test, y_test = np.array(X_test), np.array(y_test)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

batch_size = 512
last_loss = 1
desired_loss = 0.002
# Train the model
time_start = time.time()
time_deadline = 30  # seconds
a = model.fit(X_train, y_train, epochs=1, batch_size=batch_size)
while a.history['loss'][-1] > desired_loss and time.time() - time_start < time_deadline:
    batch_size = batch_size // 2
    last_loss = a.history['loss'][-1]
    a = model.fit(X_train, y_train, epochs=1, batch_size=batch_size)
    if a.history['loss'][-1] > last_loss:
        break

# Predict on test data
y_pred = model.predict(X_test)

# Plotting the results
plt.plot(time_stamps[:train_samples], train_data, label='Training Data')
plt.plot(time_stamps[train_samples+window_size:], y_test, label='Ground Truth')
plt.plot(time_stamps[train_samples+window_size:], y_pred, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sinusoid Prediction using LSTM')
plt.legend()
plt.show()

