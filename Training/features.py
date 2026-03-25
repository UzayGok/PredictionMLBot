"""
features.py
Shared feature engineering for all training approaches.
Computes technical indicators from raw OHLCV data and creates the label.
"""

import numpy as np
import pandas as pd
import ta


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators from raw OHLCV data.

    Input df must have columns: open, high, low, close, volume, num_trades.
    Returns a copy with new feature columns added.
    """
    df = df.copy()

    # ===================================================================
    # ORIGINAL FEATURES (15)
    # ===================================================================

    # --- RSI (14) ---
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)

    # --- MACD (12, 26, signal=9) ---
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # --- Bollinger Bands %B (20, 2) ---
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_pct"] = bb.bollinger_pband()

    # --- ATR (14) ---
    df["atr_14"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )

    # --- Williams %R (14) ---
    df["williams_r"] = ta.momentum.williams_r(
        df["high"], df["low"], df["close"], lbp=14
    )

    # --- Momentum (10) ---
    df["momentum_10"] = df["close"] / df["close"].shift(10) - 1

    # --- Volume SMA (20) ---
    df["volume_sma"] = df["volume"].rolling(window=20).mean()

    # --- Volume ratio ---
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # --- Price-bar features ---
    df["price_range"] = df["high"] - df["low"]
    df["price_change"] = df["close"] - df["open"]
    df["price_change_pct"] = (df["close"] - df["open"]) / df["open"] * 100
    df["high_low_ratio"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

    # ===================================================================
    # RETURN-BASED FEATURES — log returns over multiple windows
    # ===================================================================

    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    df["log_return_3"] = np.log(df["close"] / df["close"].shift(3))
    df["log_return_5"] = np.log(df["close"] / df["close"].shift(5))
    df["log_return_10"] = np.log(df["close"] / df["close"].shift(10))
    df["log_return_20"] = np.log(df["close"] / df["close"].shift(20))
    df["log_return_50"] = np.log(df["close"] / df["close"].shift(50))

    # Rolling volatility of returns (std of log returns)
    df["volatility_10"] = df["log_return_1"].rolling(10).std()
    df["volatility_20"] = df["log_return_1"].rolling(20).std()
    df["volatility_50"] = df["log_return_1"].rolling(50).std()

    # Volatility ratio (short vs long) — regime detector
    df["vol_ratio_10_50"] = df["volatility_10"] / df["volatility_50"]

    # ===================================================================
    # LAGGED FEATURES — what happened in recent candles
    # ===================================================================

    # Lagged returns (previous candles' price changes)
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_return_{lag}"] = df["log_return_1"].shift(lag)

    # Lagged volume ratios
    for lag in [1, 2, 3]:
        df[f"lag_volume_ratio_{lag}"] = df["volume_ratio"].shift(lag)

    # Lagged RSI
    df["lag_rsi_1"] = df["rsi_14"].shift(1)
    df["lag_rsi_3"] = df["rsi_14"].shift(3)

    # Consecutive up/down candles (streak)
    up = (df["close"] > df["open"]).astype(int)
    down = (df["close"] < df["open"]).astype(int)
    # Count consecutive ups: reset on down
    streak = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if up.iloc[i]:
            streak.iloc[i] = streak.iloc[i - 1] + 1 if streak.iloc[i - 1] > 0 else 1
        elif down.iloc[i]:
            streak.iloc[i] = streak.iloc[i - 1] - 1 if streak.iloc[i - 1] < 0 else -1
        # doji (open == close) → streak = 0
    df["candle_streak"] = streak

    # ===================================================================
    # EMA SLOPES — direction and steepness of moving averages
    # ===================================================================

    for span in [5, 10, 20, 50]:
        ema = df["close"].ewm(span=span, adjust=False).mean()
        df[f"ema_{span}"] = ema
        # Slope: pct change of EMA over 1 period
        df[f"ema_slope_{span}"] = ema.pct_change()

    # Price vs EMAs (distance from trend)
    df["price_vs_ema_5"] = (df["close"] - df["ema_5"]) / df["ema_5"]
    df["price_vs_ema_20"] = (df["close"] - df["ema_20"]) / df["ema_20"]
    df["price_vs_ema_50"] = (df["close"] - df["ema_50"]) / df["ema_50"]

    # EMA crossover signals (short vs long)
    df["ema_5_20_diff"] = (df["ema_5"] - df["ema_20"]) / df["ema_20"]
    df["ema_10_50_diff"] = (df["ema_10"] - df["ema_50"]) / df["ema_50"]

    # ===================================================================
    # MULTI-TIMEFRAME SIGNALS — aggregate into longer windows
    # ===================================================================

    # 1h context (12 candles of 5 min)
    df["rsi_1h"] = ta.momentum.rsi(df["close"], window=12 * 14)
    df["atr_1h"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=12 * 14
    )
    df["return_1h"] = np.log(df["close"] / df["close"].shift(12))
    df["return_4h"] = np.log(df["close"] / df["close"].shift(48))

    # Volume trend over longer windows
    df["volume_sma_60"] = df["volume"].rolling(60).mean()
    df["volume_ratio_60"] = df["volume"] / df["volume_sma_60"]

    # High/Low of last 1h and 4h — where are we in the range?
    df["high_12"] = df["high"].rolling(12).max()
    df["low_12"] = df["low"].rolling(12).min()
    df["range_pos_12"] = (df["close"] - df["low_12"]) / (df["high_12"] - df["low_12"])

    df["high_48"] = df["high"].rolling(48).max()
    df["low_48"] = df["low"].rolling(48).min()
    df["range_pos_48"] = (df["close"] - df["low_48"]) / (df["high_48"] - df["low_48"])

    # ===================================================================
    # CANDLE PATTERN FEATURES
    # ===================================================================

    body = (df["close"] - df["open"]).abs()
    full_range = df["high"] - df["low"]
    full_range_safe = full_range.replace(0, np.nan)

    # Body to range ratio — how much of the candle is body vs wicks
    df["body_ratio"] = body / full_range_safe

    # Upper wick ratio
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    df["upper_wick_ratio"] = upper_wick / full_range_safe

    # Lower wick ratio
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    df["lower_wick_ratio"] = lower_wick / full_range_safe

    # Is bullish candle
    df["is_bullish"] = (df["close"] > df["open"]).astype(int)

    # Gap (open vs previous close)
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100

    # Drop intermediate columns we don't want as features
    df = df.drop(columns=["ema_5", "ema_10", "ema_20", "ema_50",
                           "high_12", "low_12", "high_48", "low_48",
                           "volume_sma_60"])

    # ===================================================================
    # REGIME-AWARE FEATURES
    # ===================================================================

    # 24h trend direction (288 candles = 24h of 5-min)
    df["return_24h"] = np.log(df["close"] / df["close"].shift(288))

    # Realized volatility percentile (rolling rank over past week)
    # 2016 candles = 7 days of 5-min
    vol_20 = df["volatility_20"]
    df["vol_percentile_week"] = vol_20.rolling(2016).rank(pct=True)

    # Volume trend: is activity increasing or decreasing?
    # Ratio of recent volume avg to longer-term volume avg
    vol_sma_12 = df["volume"].rolling(12).mean()
    vol_sma_288 = df["volume"].rolling(288).mean()
    df["volume_trend_1h_24h"] = vol_sma_12 / vol_sma_288

    # ATR regime: current vs long-term ATR
    atr_short = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )
    atr_long = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=100
    )
    df["atr_regime"] = atr_short / atr_long

    # Trend strength: absolute return_24h / volatility_50 (like a Sharpe)
    df["trend_strength_24h"] = df["return_24h"].abs() / df["volatility_50"]

    # Mean reversion signal: distance from 24h VWAP approximation
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).rolling(288).sum()
    cum_vol = df["volume"].rolling(288).sum()
    vwap_24h = cum_tp_vol / cum_vol
    df["price_vs_vwap_24h"] = (df["close"] - vwap_24h) / vwap_24h

    # ===================================================================
    # MAGNITUDE-SPECIFIC FEATURES — for predicting big moves
    # ===================================================================

    # Candle range as % of close (current candle volatility)
    df["candle_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100

    # Volatility expansion: ATR(14) vs ATR(50) — is volatility expanding?
    atr_50 = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=50
    )
    df["vol_expansion"] = df["atr_14"] / atr_50

    # Recent big-move clustering: count of candles with |return|>0.10% in last 12
    abs_ret = ((df["close"] - df["close"].shift(1)) / df["close"].shift(1) * 100).abs()
    df["big_move_count_12"] = (abs_ret > 0.10).rolling(12).sum()
    df["big_move_count_48"] = (abs_ret > 0.10).rolling(48).sum()

    # Time since last big move (candles since |return|>0.10%)
    is_big = (abs_ret > 0.10).astype(int)
    # Efficient: use cumsum trick
    cumsum_big = is_big.cumsum()
    last_big = cumsum_big.where(is_big == 1).ffill()
    df["candles_since_big_move"] = cumsum_big - last_big

    # Volume acceleration: rate of change of volume
    vol_sma_5 = df["volume"].rolling(5).mean()
    vol_sma_20 = df["volume"].rolling(20).mean()
    df["volume_acceleration"] = vol_sma_5 / vol_sma_20

    # Spread expansion: recent high-low range vs longer-term
    range_5 = (df["high"] - df["low"]).rolling(5).mean()
    range_20 = (df["high"] - df["low"]).rolling(20).mean()
    df["range_expansion"] = range_5 / range_20

    # Hour of day (UTC) — volatility varies by trading session
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
        df["hour_of_day"] = ts.dt.hour
        df["is_us_session"] = ((ts.dt.hour >= 13) & (ts.dt.hour <= 21)).astype(int)
        df["is_asia_session"] = ((ts.dt.hour >= 0) & (ts.dt.hour <= 8)).astype(int)
    else:
        df["hour_of_day"] = 0
        df["is_us_session"] = 0
        df["is_asia_session"] = 0

    # Bollinger Band width — wider bands = more volatile period
    df["bb_width"] = bb.bollinger_wband()

    # Volume spike: current volume vs rolling max
    vol_max_20 = df["volume"].rolling(20).max()
    df["volume_spike_ratio"] = df["volume"] / vol_max_20

    # Drop intermediate columns
    df = df.drop(columns=["volume_sma_60_drop"], errors="ignore")

    return df


FEATURE_COLS = [
    # Original 15
    "rsi_14", "macd_signal", "macd_diff", "bb_pct", "atr_14",
    "williams_r", "momentum_10", "volume_sma", "volume_ratio",
    "price_range", "price_change", "price_change_pct", "high_low_ratio",
    "volume", "num_trades",
    # Return-based (10)
    "log_return_1", "log_return_3", "log_return_5", "log_return_10",
    "log_return_20", "log_return_50",
    "volatility_10", "volatility_20", "volatility_50", "vol_ratio_10_50",
    # Lagged (10)
    "lag_return_1", "lag_return_2", "lag_return_3", "lag_return_5", "lag_return_10",
    "lag_volume_ratio_1", "lag_volume_ratio_2", "lag_volume_ratio_3",
    "lag_rsi_1", "lag_rsi_3",
    # Streak (1)
    "candle_streak",
    # EMA slopes (9)
    "ema_slope_5", "ema_slope_10", "ema_slope_20", "ema_slope_50",
    "price_vs_ema_5", "price_vs_ema_20", "price_vs_ema_50",
    "ema_5_20_diff", "ema_10_50_diff",
    # Multi-timeframe (7)
    "rsi_1h", "atr_1h", "return_1h", "return_4h",
    "volume_ratio_60", "range_pos_12", "range_pos_48",
    # Candle patterns (5)
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "is_bullish", "gap_pct",
    # Regime-aware (6)
    "return_24h", "vol_percentile_week", "volume_trend_1h_24h",
    "atr_regime", "trend_strength_24h", "price_vs_vwap_24h",
    # Magnitude-specific (13)
    "candle_range_pct", "vol_expansion", "big_move_count_12",
    "big_move_count_48", "candles_since_big_move", "volume_acceleration",
    "range_expansion", "hour_of_day", "is_us_session", "is_asia_session",
    "bb_width", "volume_spike_ratio",
]

# Features used by the direction model (original top 45 from ablation)
DIRECTION_FEATURES = [
    "lag_return_5", "body_ratio", "return_24h", "trend_strength_24h",
    "lag_return_3", "atr_regime", "high_low_ratio", "lag_volume_ratio_2",
    "atr_1h", "lag_volume_ratio_1", "volume_ratio_60", "lag_return_2",
    "upper_wick_ratio", "lower_wick_ratio", "volume_trend_1h_24h",
    "lag_return_10", "rsi_1h", "vol_percentile_week", "volume_ratio",
    "lag_volume_ratio_3", "price_vs_vwap_24h", "macd_diff", "lag_return_1",
    "vol_ratio_10_50", "volume", "gap_pct", "volatility_20", "volatility_50",
    "log_return_5", "log_return_3", "volume_sma", "range_pos_48",
    "num_trades", "log_return_20", "ema_slope_10", "lag_rsi_3",
    "range_pos_12", "return_4h", "price_range", "williams_r",
    "lag_rsi_1", "rsi_14", "macd_signal", "bb_pct", "log_return_50",
]


def make_label(df: pd.DataFrame) -> pd.Series:
    """
    Label: will the NEXT candle close higher than the current close?
    y = 1 if close[t+1] > close[t], else 0.
    The last row gets NaN (no future candle).
    """
    return (df["close"].shift(-1) > df["close"]).astype(float)


def make_label_large_moves(df: pd.DataFrame, pct_threshold: float = 0.02):
    """
    Label only large moves. Returns y with NaN for small moves (dead zone).
    y = 1 if next candle return > +threshold%
    y = 0 if next candle return < -threshold%
    y = NaN if move is within dead zone (will be dropped during training)

    pct_threshold: minimum absolute % move to count (default 0.02%)
    """
    next_return_pct = (df["close"].shift(-1) - df["close"]) / df["close"] * 100
    y = pd.Series(np.nan, index=df.index)
    y[next_return_pct > pct_threshold] = 1.0
    y[next_return_pct < -pct_threshold] = 0.0
    return y


def prepare_dataset(df: pd.DataFrame):
    """
    Full pipeline: compute features, create label, drop NaN rows.
    Returns (X, y) where X is a DataFrame of features and y is a Series.
    """
    df = calculate_features(df)
    df["label"] = make_label(df)
    df = df.dropna().reset_index(drop=True)
    X = df[FEATURE_COLS]
    y = df["label"]
    return X, y
