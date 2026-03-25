"""
Fetch 10,000 5-min BTCUSDT candles from Binance at a given offset.

Usage:
    python fetch_older_data.py <set_number>

Set 1 = candles 10k-20k ago  (already fetched)
Set 2 = candles 20k-30k ago
Set 3 = candles 30k-40k ago
Set 4 = candles 40k-50k ago
Set 5 = candles 50k-60k ago

Each set of 10,000 candles covers ~34.7 days.
"""

import os
import sys
import requests
import pandas as pd
import datetime
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

# Training data starts at 2026-02-19 02:55:00 UTC
TRAIN_START = pd.Timestamp("2026-02-19 02:55:00", tz="UTC")
TRAIN_START_MS = int(TRAIN_START.timestamp() * 1000)
CANDLE_MS = 5 * 60 * 1000  # 5 minutes in ms


def fetch_set(set_number):
    """Fetch 10k candles for the given set number (1-based)."""
    # Each set is 10,000 candles further back
    offset_candles = set_number * 10_000
    end_ms = TRAIN_START_MS - ((set_number - 1) * 10_000 * CANDLE_MS) - 1

    output_file = os.path.join(_ROOT, "data", f"btc_candles_set{set_number}.csv")
    print(f"=== Set {set_number}: candles {set_number*10}k-{(set_number+1)*10}k ago ===")

    all_candles = []
    current_end = end_ms

    for batch in range(10):
        params = {
            "symbol": "BTCUSDT",
            "interval": "5m",
            "endTime": current_end,
            "limit": 1000,
        }
        r = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if not data:
            print(f"  Batch {batch+1}: no data returned, stopping.")
            break

        all_candles = data + all_candles
        current_end = data[0][0] - 1

        oldest = datetime.datetime.utcfromtimestamp(data[0][0] / 1000).strftime("%Y-%m-%d %H:%M")
        newest = datetime.datetime.utcfromtimestamp(data[-1][0] / 1000).strftime("%Y-%m-%d %H:%M")
        print(f"  Batch {batch+1}/10: {len(data)} candles  ({oldest} to {newest})")

        time.sleep(0.2)

    df = pd.DataFrame(all_candles, columns=COLUMNS)
    df = df[["timestamp", "open", "high", "low", "close", "quote_volume", "num_trades"]].copy()
    df = df.rename(columns={"quote_volume": "volume"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    print(f"  Candles: {len(df)}")
    print(f"  Range:   {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    df.to_csv(output_file, index=False)
    print(f"  Saved:   {output_file}\n")
    return output_file


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sets = [int(x) for x in sys.argv[1:]]
    else:
        sets = [1, 2, 3, 4, 5]

    for s in sets:
        fetch_set(s)
