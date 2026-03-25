"""
feature_ablation.py
Test model accuracy as we progressively remove the least important features.
Uses large-moves model (Strategy 2) with Optuna-tuned hyperparameters.
Trains on 100k, tests separately on set5+set4 (20k) and btc_candles_10k.

Usage:
    python Training/feature_ablation.py
"""

import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from Training.features import (
    calculate_features, make_label_large_moves, FEATURE_COLS,
)

# Optuna-tuned hyperparameters for large-moves model
TUNED_PARAMS = {
    "n_estimators": 500, "max_depth": 6,
    "learning_rate": 0.020784549897083126,
    "num_leaves": 37, "min_child_samples": 112,
    "subsample": 0.8103767331776779,
    "colsample_bytree": 0.5486659787425067,
    "reg_alpha": 0.0034916132094567013,
    "reg_lambda": 0.1118425108233888,
    "random_state": 42, "verbose": -1,
}
PCT_THRESHOLD = 0.0372654293444779


def load_and_prepare(path):
    df = pd.read_csv(path)
    df = calculate_features(df)
    df["label"] = make_label_large_moves(df, PCT_THRESHOLD)
    df = df.dropna(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)
    return df


def evaluate_at_n(X_train, y_train, test_sets, feature_subset):
    """Train once, test on multiple sets. Returns list of (overall, bands) per test set."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train[feature_subset])

    model = lgb.LGBMClassifier(**TUNED_PARAMS)
    model.fit(Xtr, y_train)

    results = []
    for X_test, y_test in test_sets:
        Xte = scaler.transform(X_test[feature_subset])
        y_pred = model.predict(Xte)
        proba = model.predict_proba(Xte)
        confidence = np.maximum(proba[:, 0], proba[:, 1])

        overall = accuracy_score(y_test, y_pred)
        bands = {}
        for thresh in [0.55, 0.60, 0.65]:
            mask = confidence >= thresh
            n = mask.sum()
            if n > 0:
                acc = accuracy_score(y_test[mask], y_pred[mask])
                bands[thresh] = (acc, n)
            else:
                bands[thresh] = (None, 0)
        results.append((overall, bands))

    return results, model, scaler


def main():
    # Load training data
    print("Loading data...")
    print(f"Label: large moves only (dead zone ±{PCT_THRESHOLD:.4f}%)")
    print(f"Using Optuna-tuned hyperparameters\n")

    train_df = load_and_prepare(os.path.join(_ROOT, "data", "btc_candles_100k.csv"))

    test_20k_df = pd.concat([
        load_and_prepare(os.path.join(_ROOT, "data", "btc_candles_set5.csv")),
        load_and_prepare(os.path.join(_ROOT, "data", "btc_candles_set4.csv")),
    ], ignore_index=True)

    test_10k_df = load_and_prepare(os.path.join(_ROOT, "data", "btc_candles_10k.csv"))

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]

    X_20k = test_20k_df[FEATURE_COLS]
    y_20k = test_20k_df["label"]

    X_10k = test_10k_df[FEATURE_COLS]
    y_10k = test_10k_df["label"]

    print(f"Train: {len(X_train):,}  |  Test 20k: {len(X_20k):,}  |  Test 10k: {len(X_10k):,}")
    print(f"Features: {len(FEATURE_COLS)}\n")

    # Get feature importance ranking from full model
    print("Training full model (63 features) for importance ranking...")
    scaler = StandardScaler()
    Xtr_full = scaler.fit_transform(X_train)
    full_model = lgb.LGBMClassifier(**TUNED_PARAMS)
    full_model.fit(Xtr_full, y_train)

    importance = full_model.feature_importances_
    ranking = sorted(zip(FEATURE_COLS, importance), key=lambda x: -x[1])

    print("\n=== Feature ranking (most to least important) ===")
    for i, (feat, imp) in enumerate(ranking):
        print(f"  {i+1:>2}. {feat:25s}  {imp:6.0f}")

    ranked_features = [f for f, _ in ranking]

    # Test from 63 features down to 40
    print("\n" + "=" * 170)
    print(f"{'N':>3}  {'Removed feature':25s}  {'20k overall':>11s}  {'20k >=55%':>12s}  {'20k >=60%':>12s}  {'20k >=65%':>12s}  {'10k overall':>11s}  {'10k >=55%':>12s}  {'10k >=60%':>12s}  {'10k >=65%':>12s}")
    print("=" * 170)

    for n_feat in range(63, 39, -1):
        subset = ranked_features[:n_feat]
        removed = ranked_features[n_feat] if n_feat < 63 else "—"

        test_sets = [(X_20k, y_20k), (X_10k, y_10k)]
        results, _, _ = evaluate_at_n(X_train, y_train, test_sets, subset)
        (acc_20k, bands_20k) = results[0]
        (acc_10k, bands_10k) = results[1]

        def fmt_band(bands, t):
            acc, n = bands[t]
            if acc is None:
                return "  —         "
            return f"{acc*100:5.2f}% ({n:>4})"

        print(f"{n_feat:>3}  {removed:25s}  {acc_20k*100:>10.2f}%  {fmt_band(bands_20k, 0.55)}  {fmt_band(bands_20k, 0.60)}  {fmt_band(bands_20k, 0.65)}  {acc_10k*100:>10.2f}%  {fmt_band(bands_10k, 0.55)}  {fmt_band(bands_10k, 0.60)}  {fmt_band(bands_10k, 0.65)}")


if __name__ == "__main__":
    main()
