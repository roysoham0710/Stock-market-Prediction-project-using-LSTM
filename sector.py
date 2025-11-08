import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
# List of reliable financial sector tickers
TICKERS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS"
]

# Download Adjusted Close prices
raw = yf.download(TICKERS, start="2015-01-01", end="2025-01-01")["Close"]

# Drop columns with no data
raw = raw.dropna(axis=1, how="all")

print("Tickers successfully downloaded:")
print(raw.columns.tolist())
print("\nPrice DataFrame Shape:", raw.shape)

# Market caps (approx values, used only for weighting purpose)
market_caps = {
    "HDFCBANK.NS": 1100000,
    "ICICIBANK.NS": 800000,
    "SBIN.NS": 700000,
    "KOTAKBANK.NS": 350000,
    "AXISBANK.NS": 300000,
    "BAJFINANCE.NS": 450000,
    "BAJAJFINSV.NS": 200000,
    "HDFCLIFE.NS": 140000,
    "SBILIFE.NS": 110000,
    "ICICIPRULI.NS": 90000
}

# Keep only tickers that actually downloaded
market_caps = {t: market_caps[t] for t in raw.columns if t in market_caps}

weights = np.array(list(market_caps.values()))
weights = weights / weights.sum()  # normalize to sum = 1

# Compute sector index
sector_index = (raw * weights).sum(axis=1)

print("\nSector Index (first 10 values):")
print(sector_index.head(10))

# Plot the weighted index
plt.figure(figsize=(10,5))
plt.plot(sector_index, linewidth=2)
plt.title("Financial Sector Market Cap Weighted Index")
plt.xlabel("Date")
plt.ylabel("Index Value (Weighted Price)")
plt.grid(True)
plt.show()




# Convert sector_index Series to DataFrame
df = sector_index.to_frame(name="Index")

# Scale values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
df["Scaled"] = scaler.fit_transform(df[["Index"]])

# Prepare training sequences
def create_sequences(data, window_size=60):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

WINDOW = 60
values = df["Scaled"].values
X, y = create_sequences(values, WINDOW)

# Split into train and test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM: (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(WINDOW, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Prediction
pred = model.predict(X_test)

# Inverse scale back to original values
predicted_prices = scaler.inverse_transform(pred)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(actual_prices, label="Actual Trend")
plt.plot(predicted_prices, label="Predicted Trend")
plt.title("Financial Sector Index Prediction")
plt.xlabel("Time (Test Range)")
plt.ylabel("Sector Index Value")
plt.legend()
plt.grid(True)
plt.show()
