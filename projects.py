import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('banknifty_data.csv')

# Prepare the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Split the data into training and test sets
train_data = scaled_data[:-7]
test_data = scaled_data[-7:]

# Prepare the training data
X_train = []
y_train = []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Prepare the test data
inputs = data[len(data) - len(test_data) - 60:]
X_test = scaler.transform(inputs['Close'].values.reshape(-1, 1))
X_test = np.array([X_test[i-60:i, 0] for i in range(60, len(X_test))])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict the prices for the next week
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Create a DataFrame to store the predicted prices
predicted_data = pd.DataFrame({
    'Date': data['Date'].iloc[-7:].values,
    'Predicted Price': predicted_prices[:, 0]
})

# Print the predicted prices
print(predicted_data)