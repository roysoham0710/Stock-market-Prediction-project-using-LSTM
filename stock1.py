# infosys_lstm_pipeline.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta  # technical indicators library (optional)

# -----------------------------
# Config
# -----------------------------
TICKER = "INFY.NS"           # Infosys NSE ticker for yfinance
START = "2013-01-01"         # historical start date (change as needed)
END = datetime.today().strftime("%Y-%m-%d")  # up to today (real-time/latest)
SEQ_LEN = 60                 # number of past days used to predict next day
TEST_RATIO = 0.20            # 20% test
RANDOM_SEED = 42
EPOCHS = 30
BATCH_SIZE = 32

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# -----------------------------
# Step 1: Fetch data
# -----------------------------
print(f"Downloading {TICKER} from {START} to {END} ...")
data = yf.download(TICKER, start=START, end=END, progress=False)

if data.empty:
    raise SystemExit(f"No data found for {TICKER}. Check ticker or internet connection.")

# Use Adjusted Close if available (accounts for splits/dividends), otherwise Close
price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
df = data[[price_col, "Volume"]].rename(columns={price_col: "Adj_Close"})

# -----------------------------
# Step 2: Basic EDA & plots
# -----------------------------
print("Basic info:")
print(df.head())
print("\nSummary statistics:")
print(df["Adj_Close"].describe())

# Plot closing price
plt.figure(figsize=(12,5))
plt.plot(df.index, df["Adj_Close"], label="Adj Close")
plt.title(f"{TICKER} Adjusted Close Price")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True)
plt.show()

# Moving averages and volatility
df["MA50"] = df["Adj_Close"].rolling(window=50).mean()
df["MA200"] = df["Adj_Close"].rolling(window=200).mean()
df["Daily_Return"] = df["Adj_Close"].pct_change()
df["Vol30"] = df["Daily_Return"].rolling(window=30).std() * np.sqrt(252)  # annualized vol approx

plt.figure(figsize=(12,6))
plt.plot(df.index, df["Adj_Close"], label="Adj Close")
plt.plot(df.index, df["MA50"], label="MA50")
plt.plot(df.index, df["MA200"], label="MA200")
plt.title(f"{TICKER} Price with MA50 & MA200")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.plot(df.index, df["Vol30"], label="30-day rolling vol (annualized)")
plt.title(f"{TICKER} Volatility")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Step 3: Cleaning & handling missing values
# -----------------------------
df_clean = df.copy()
df_clean.dropna(subset=["Adj_Close"], inplace=True)   # remove rows without prices
df_clean.fillna(method="ffill", inplace=True)         # forward-fill any other NaNs
df_clean = df_clean[["Adj_Close"]]                    # keep only price for modeling

# -----------------------------
# Step 4: Prepare sequences (sliding windows)
# -----------------------------
def create_sequences(values, seq_len):
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i-seq_len:i])
        y.append(values[i])
    return np.array(X), np.array(y)

# scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(df_clean.values)  # shape (n_samples, 1)

# split into train/test by time (no shuffling)
split_idx = int(len(scaled) * (1 - TEST_RATIO))
train_data = scaled[:split_idx]
test_data = scaled[split_idx - SEQ_LEN:]  # ensure we have initial history for test sequences

X_train, y_train = create_sequences(train_data, SEQ_LEN)
X_test, y_test = create_sequences(test_data, SEQ_LEN)

print("Shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# reshape already (samples, seq_len, features=1)

# -----------------------------
# Step 5: Build LSTM model
# -----------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)  # linear output for regression
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# -----------------------------
# Step 6: Train
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    shuffle=False  # important for time series
)

# Plot training loss
plt.figure(figsize=(8,4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.yscale("log")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Step 7: Predict & invert scaling
# -----------------------------
pred_scaled = model.predict(X_test)
pred = scaler.inverse_transform(pred_scaled)
y_true = scaler.inverse_transform(y_test.reshape(-1,1))

# -----------------------------
# Step 8: Baseline (persistence): predict previous day's price as next day's
# -----------------------------
# For baseline we can use last value in each input seq as prediction
baseline_preds = X_test[:, -1, 0]  # scaled last value of each sequence
baseline_preds = scaler.inverse_transform(baseline_preds.reshape(-1,1))

# -----------------------------
# Step 9: Evaluation metrics
# -----------------------------
def rmse(a,b):
    return math.sqrt(mean_squared_error(a,b))

def mape(a,b):
    a, b = np.array(a).reshape(-1), np.array(b).reshape(-1)
    # avoid divide by zero
    mask = a != 0
    return np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100

print("\nEvaluation on Test Set:")
print(f"LSTM RMSE: {rmse(y_true, pred):.4f}")
print(f"LSTM MAE: {mean_absolute_error(y_true, pred):.4f}")
print(f"LSTM MAPE: {mape(y_true, pred):.4f}%")

print("\nBaseline (persistence) Performance:")
print(f"Baseline RMSE: {rmse(y_true, baseline_preds):.4f}")
print(f"Baseline MAE: {mean_absolute_error(y_true, baseline_preds):.4f}")
print(f"Baseline MAPE: {mape(y_true, baseline_preds):.4f}%")

# -----------------------------
# Step 10: Plot Actual vs Predicted
# -----------------------------
# Build a timeline index for test predictions
test_index = df_clean.index[split_idx:]  # dates corresponding to test y's
# BUT because we constructed test_data starting from split_idx - SEQ_LEN,
# first SEQ_LEN samples are used for warmup, so predictions start at split_idx
test_index = test_index[SEQ_LEN:] if len(test_index) > SEQ_LEN else test_index

plt.figure(figsize=(12,6))
plt.plot(test_index, y_true, label="Actual", color="green")
plt.plot(test_index, pred, label="LSTM Predicted", color="red")
plt.plot(test_index, baseline_preds, label="Baseline (prev-day)", color="blue", alpha=0.5)
plt.title(f"{TICKER} Actual vs Predicted (LSTM)")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Step 11: Save model (optional)
# -----------------------------
model_save_path = "infosys_lstm_model.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# End
