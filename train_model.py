# Importing Libraries 
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load the training data
dtrain = pd.read_csv('train_file_path.csv')

# If necessary, convert the 'Open' column to numeric (this will convert non-numeric entries to NaN)
dtrain['Open'] = pd.to_numeric(dtrain['Open'], errors='coerce')

# Optionally drop rows with NaN values if any conversion issues occurred
dtrain.dropna(subset=['Open'], inplace=True)

# Use the 'Open' column directly
training_set = dtrain[['Open']].values

# Proceed with scaling and creating the data structure...
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)


# Creating Data Structure with 60 Timesteps
X_train, y_train = [], []
for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Initializing the LSTM Model
regressor = Sequential()

# Adding LSTM Layers with Dropout Regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))

# Compiling and Training the Model
regressor.compile(loss='mean_squared_error', optimizer='adam')
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Saving the Model and Scaler
regressor.save("/mnt/data/stock_model.h5")
np.save("/mnt/data/scaler.npy", sc)
print("Model and scaler saved successfully.")
