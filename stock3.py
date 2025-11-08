import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Correct tickers
tickers = [
    "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","LICI.NS","BAJFINANCE.NS","KOTAKBANK.NS","AXISBANK.NS",
    "BAJAJFINSV.NS","JIOFIN.NS","SBILIFE.NS","IRFC.NS","HDFCLIFE.NS","BAJAJHLDNG.NS","PFC.NS",
    "CHOLAFIN.NS","PNB.NS","SHRIRAMFIN.NS","BANKBARODA.NS","UNIONBANK.NS","HDFCAMC.NS","IDBI.NS",
    "MUTHOOTFIN.NS","RECLTD.NS","CANBK.NS"
]

# Download Adjusted Close prices
prices = yf.download(tickers, start="2015-01-01")['Close']

# Get market caps for weighting
mcap = {}
for t in tickers:
    info = yf.Ticker(t).info
    if "marketCap" in info and info["marketCap"] is not None:
        mcap[t] = info["marketCap"]

# Compute weights
total_mcap = sum(mcap.values())
weights = {t: mcap[t] / total_mcap for t in mcap}

# Build market-cap weighted index
weighted_index = sum(prices[t] * weights.get(t, 0) for t in prices.columns)
weighted_index = weighted_index.dropna()

# Linear Trend Model
days = np.arange(len(weighted_index)).reshape(-1, 1)
model = LinearRegression().fit(days, weighted_index.values.reshape(-1, 1))
trend_line = model.predict(days)

# Plot
plt.figure(figsize=(12,6))
plt.plot(weighted_index.index, weighted_index, label="Market Cap Weighted Index", linewidth=2)
plt.plot(weighted_index.index, trend_line, label="Trend (Linear Prediction)", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.title("Finance Sector Market Cap Weighted Trend")
plt.legend()
plt.show()

print("✅ Done")
