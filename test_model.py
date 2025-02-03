# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, r2_score
import math

# Loading the Test Data
# Option A: If the first row is not data, skip it
dtest = pd.read_csv('test_file_path.csv')

# Option B: Convert the 'Open' column to numeric and drop problematic rows
dtest['Open'] = pd.to_numeric(dtest['Open'], errors='coerce')
dtest.dropna(subset=['Open'], inplace=True)
real_stock_price = dtest[['Open']].values


# Loading the Saved Model and Scaler
model = load_model("/mnt/data/stock_model.h5")
sc = np.load("/mnt/data/scaler.npy", allow_pickle=True).item()

# Preparing Test Inputs
total_data = pd.concat((pd.read_csv("trainig_file_path.csv")["Open"], dtest["Open"]), axis=0)
inputs = total_data[len(total_data) - len(dtest) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Making Predictions
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Calculating the RMSE and R2 Score
rmse = math.sqrt(root_mean_squared_error(real_stock_price, predicted_stock_price))
print("Root Mean Squared Error:", rmse)

r2 = r2_score(real_stock_price, predicted_stock_price)
print("R2 Score:", r2)

# Visualizing the Results
plt.plot(real_stock_price, color='red', label='Real Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
