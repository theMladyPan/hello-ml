import requests
import matplotlib.pyplot as plt
import json

# Fetch tickers
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
days = "50"
interval = "hourly"
filename = f"btc_{days}_days_{interval}_interval.json"
# check if the file exists
try:
    with open(filename, "r") as f:
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
    with open(filename, "w") as f:
        json.dump(data, f)
    

# Extract timestamps and prices
timestamps = [timestamp[0] / 1000 for timestamp in data['prices'][::-1][0:1000]]  # Convert milliseconds to seconds
prices = [price[1] for price in data['prices'][::-1][0:1000]]
market_cap = [market_cap[1] for market_cap in data['market_caps'][::-1][0:1000]]
total_volume = [total_volume[1] for total_volume in data['total_volumes'][::-1][0:1000]]

# Plotting the tickers
plt.plot(timestamps, prices)
# plot market cap in a different color on secondary axis
plt.twinx()
plt.plot(timestamps, total_volume, color='red')
plt.xlabel('Timestamp')
plt.ylabel('Price (USD)')
plt.title('BTC Tickers')
plt.grid(True)
plt.show()
