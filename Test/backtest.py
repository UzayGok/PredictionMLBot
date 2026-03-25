"""
backtest.py
Backtest the model on historical data.

Computes features once on the full dataset (same as training), then
walks forward predicting each candle.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from Training.features import calculate_features

# Load model artifacts
models_dir = os.path.join(_ROOT, "models")
model = pickle.load(open(os.path.join(models_dir, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(models_dir, "scaler.pkl"), "rb"))
features = pickle.load(open(os.path.join(models_dir, "features.pkl"), "rb"))

# Allow specifying a different data file via command line
default_data = os.path.join(_ROOT, "data", "btc_candles_10k.csv")
data_file = sys.argv[1] if len(sys.argv) > 1 else default_data

# Load raw candles
df = pd.read_csv(data_file)
print(f"Loaded {len(df)} candles from {data_file}")
print(f"Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Compute features on the full dataset at once
df = calculate_features(df)
df = df.dropna().reset_index(drop=True)
print(f"Rows after feature computation: {len(df)}")

# Target: did the NEXT candle close higher?
df["actual_up"] = (df["close"].shift(-1) > df["close"]).astype(float)
df = df.iloc[:-1].reset_index(drop=True)  # drop last row (no next candle)

# Scale and predict all at once
X_scaled = scaler.transform(df[features].values)
probas = model.predict_proba(X_scaled)
df["prob_up"] = probas[:, 1]
df["confidence"] = np.maximum(probas[:, 0], probas[:, 1])
df["predicted_up"] = df["prob_up"] >= 0.5
df["correct"] = df["predicted_up"] == df["actual_up"].astype(bool)

total = len(df)
correct = df["correct"].sum()
accuracy = correct / total * 100

print(f"\nTotal predictions: {total}")
print(f"Correct:           {correct}")
print(f"Accuracy:          {accuracy:.2f}%")

# High confidence
CONF_THRESHOLD = 0.55
hc = df[df["confidence"] >= CONF_THRESHOLD]
if len(hc) > 0:
    hc_acc = hc["correct"].sum() / len(hc) * 100
    print(f"\nHigh confidence (>= {CONF_THRESHOLD:.0%}):")
    print(f"  Predictions:     {len(hc)} ({len(hc)/total*100:.1f}% of all)")
    print(f"  Correct:         {hc['correct'].sum()}")
    print(f"  Accuracy:        {hc_acc:.2f}%")

# Breakdown by confidence bands
print(f"\nAccuracy by confidence band:")
for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.0)]:
    band = df[(df["confidence"] >= lo) & (df["confidence"] < hi)]
    if len(band) > 0:
        band_acc = band["correct"].sum() / len(band) * 100
        print(f"  {lo:.0%}-{hi:.0%}:  {band_acc:6.2f}%  ({len(band)} predictions)")
