"""
data_fetcher.py
Fetch BTCUSDT 5-minute klines from Binance, aligned to UTC 5-min boundaries.

Binance /api/v3/klines response columns (per the API docs):
  [0]  open_time         - ms timestamp
  [1]  open
  [2]  high
  [3]  low
  [4]  close
  [5]  volume
  [6]  close_time        - ms timestamp
  [7]  quote_asset_volume
  [8]  num_trades
  [9]  taker_buy_base_volume
  [10] taker_buy_quote_volume
  [11] ignore
"""

import calendar
import datetime

import pandas as pd
import requests

BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

_BINANCE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def current_boundary_utc() -> datetime.datetime:
    """Return the current UTC 5-minute boundary as a naive UTC datetime."""
    now = datetime.datetime.utcnow()
    return now.replace(second=0, microsecond=0, minute=(now.minute // 5) * 5)


def _boundary_to_ms(boundary: datetime.datetime) -> int:
    """
    Convert a naive UTC datetime to milliseconds timestamp.
    Uses calendar.timegm() to avoid local-timezone contamination
    that datetime.timestamp() would introduce.
    """
    return calendar.timegm(boundary.timetuple()) * 1000


def fetch_candles(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    limit: int = 100,
) -> pd.DataFrame:
    """
    Fetch the latest `limit` completed klines ending at the current UTC
    5-minute boundary.

    Parameters
    ----------
    symbol   : Binance trading pair, e.g. "BTCUSDT"
    interval : Kline interval, e.g. "5m"
    limit    : Number of candles to fetch (max 1000 per Binance docs)

    Returns
    -------
    DataFrame with columns:
        timestamp (UTC-aware datetime), open, high, low, close,
        volume (float), num_trades (int)

    The last row is the most recently completed candle at the current
    5-minute boundary.
    """
    boundary = current_boundary_utc()
    end_time_ms = _boundary_to_ms(boundary)

    params = {
        "symbol": symbol,
        "interval": interval,
        # Subtract 1ms so the in-progress candle (which opens exactly at the
        # boundary) is excluded. Only fully completed candles are returned.
        "endTime": end_time_ms - 1,
        "limit": limit,
    }

    response = requests.get(
        BASE_URL + KLINES_ENDPOINT,
        params=params,
        timeout=10,
    )
    response.raise_for_status()

    df = pd.DataFrame(response.json(), columns=_BINANCE_COLUMNS)

    # Use quote_volume (USDT) as "volume" — this matches the training data,
    # which used quote asset volume (millions of USDT), not base volume (BTC).
    df = df[["timestamp", "open", "high", "low", "close", "quote_volume", "num_trades"]].copy()
    df = df.rename(columns={"quote_volume": "volume"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)

    return df.reset_index(drop=True)
