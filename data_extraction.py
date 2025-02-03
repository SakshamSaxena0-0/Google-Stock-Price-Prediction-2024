import yfinance as yf
import pandas as pd

# Fetch data up to today
data = yf.download("GOOGL", start="2017-01-01", end="2023-12-31")
data.to_csv("GOOGL_2023_train.csv")  # Save updated data

# Fetch data up to today
data = yf.download("GOOGL", start="2024-01-01", end="2024-01-30") 
data.to_csv("GOOGL_2024_test.csv")  # Save updated data