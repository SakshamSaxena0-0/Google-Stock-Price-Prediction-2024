# Google-Stock-Price-Prediction-2024

# Stock Price Prediction using LSTM

This repository contains code for predicting Google stock prices using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. The project is divided into two main scripts:
- **`train_model.py`**: Preprocesses training data, builds and trains an LSTM model, and saves the trained model along with the scaler.
- **`test_model.py`**: Loads the saved model and scaler, processes test data (e.g., 2024 data), makes predictions, and visualizes the results. It also calculates the RMSE and R² score.


## Features

- **Data Preprocessing:** Uses MinMax scaling to preprocess the stock price data.
- **Model Architecture:** Implements a multi-layer LSTM model with dropout regularization.
- **Model Saving and Loading:** Saves the trained model and scaler for later use.
- **Evaluation Metrics:** Calculates RMSE (Root Mean Squared Error) and R² Score.
- **Visualization:** Plots the real vs. predicted stock prices.

## Contributing
Feel free to open issues or submit pull requests for improvements and bug fixes.

## License
This project is open source and available under the MIT License.

