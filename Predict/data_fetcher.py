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
TIME_ENDPOINT = "/api/v3/time"

_BINANCE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _binance_server_utc() -> datetime.datetime:
    """Return the current Binance server time as a timezone-aware UTC datetime."""
    response = requests.get(BASE_URL + TIME_ENDPOINT, timeout=5)
    response.raise_for_status()
    server_ms = response.json()["serverTime"]
    return datetime.datetime.fromtimestamp(server_ms / 1000, tz=datetime.timezone.utc)


def current_boundary_utc() -> datetime.datetime:
    """
    Return the current UTC 5-minute boundary.

    Prefer Binance server time so candle selection stays aligned with the
    exchange even if the local machine clock drifts. Fall back to local UTC
    if the server-time request fails.
    """
    try:
        now = _binance_server_utc()
    except (requests.RequestException, KeyError, ValueError):
        now = datetime.datetime.now(datetime.timezone.utc)
    return now.replace(second=0, microsecond=0, minute=(now.minute // 5) * 5)


def _boundary_to_ms(boundary: datetime.datetime) -> int:
    """
    Convert a UTC datetime to milliseconds timestamp.
    Uses calendar.timegm() to avoid local-timezone contamination
    that datetime.timestamp() would introduce.
    """
    if boundary.tzinfo is not None:
        boundary = boundary.astimezone(datetime.timezone.utc).replace(tzinfo=None)
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
    limit    : Number of candles to fetch. Requests are paginated in chunks
               of up to 1000 candles to satisfy larger lookbacks.

    Returns
    -------
    DataFrame with columns:
        timestamp (UTC-aware datetime), open, high, low, close,
        volume (float), num_trades (int)

    The last row is the most recently completed candle at the current
    5-minute boundary.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")

    boundary = current_boundary_utc()
    end_time_ms = _boundary_to_ms(boundary) - 1
    batches = []
    remaining = limit

    while remaining > 0:
        batch_limit = min(1000, remaining)
        params = {
            "symbol": symbol,
            "interval": interval,
            # Subtract 1ms so the in-progress candle (which opens exactly at the
            # boundary) is excluded. Only fully completed candles are returned.
            "endTime": end_time_ms,
            "limit": batch_limit,
        }

        response = requests.get(
            BASE_URL + KLINES_ENDPOINT,
            params=params,
            timeout=10,
        )
        response.raise_for_status()

        payload = response.json()
        if not payload:
            break

        batches.append(pd.DataFrame(payload, columns=_BINANCE_COLUMNS))
        remaining -= len(payload)
        end_time_ms = int(payload[0][0]) - 1

        if len(payload) < batch_limit:
            break

    if not batches:
        raise ValueError("Binance returned no candles")

    df = pd.concat(reversed(batches), ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Use quote_volume (USDT) as "volume" — this matches the training data,
    # which used quote asset volume (millions of USDT), not base volume (BTC).
    df = df[["timestamp", "open", "high", "low", "close", "quote_volume", "num_trades"]].copy()
    df = df.rename(columns={"quote_volume": "volume"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)

    return df.reset_index(drop=True)
