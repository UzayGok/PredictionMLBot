"""
train_compare.py
Train 3 different models on 100k candles, compare on the next 20k (set5 + set4).

Label: y = 1 if close[t+1] > close[t], else 0  (pure direction prediction)

Usage:
    python Training/train_compare.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from Training.features import FEATURE_COLS, prepare_dataset


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_selected_features():
    """Load selected features from pkl if available, otherwise use all."""
    feat_path = os.path.join(_ROOT, "models", "features.pkl")
    if os.path.exists(feat_path):
        with open(feat_path, "rb") as f:
            selected = pickle.load(f)
        print(f"Using {len(selected)} selected features from models/features.pkl")
        return selected
    print(f"No features.pkl found, using all {len(FEATURE_COLS)} features")
    return FEATURE_COLS


def load_data(feature_cols):
    train_path = os.path.join(_ROOT, "data", "btc_candles_100k.csv")
    set5_path = os.path.join(_ROOT, "data", "btc_candles_set5.csv")
    set4_path = os.path.join(_ROOT, "data", "btc_candles_set4.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.concat([
        pd.read_csv(set5_path),
        pd.read_csv(set4_path),
    ], ignore_index=True)

    print(f"Train candles : {len(train_df):,}  ({train_df['timestamp'].iloc[0]} to {train_df['timestamp'].iloc[-1]})")
    print(f"Test candles  : {len(test_df):,}  ({test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]})")

    X_train, y_train = prepare_dataset(train_df)
    X_test, y_test = prepare_dataset(test_df)

    # Subset to selected features
    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    print(f"Train samples : {len(X_train):,}  (after feature NaN drop)")
    print(f"Test samples  : {len(X_test):,}")
    print(f"Train label balance: UP {y_train.mean():.4f}  DOWN {1 - y_train.mean():.4f}")
    print(f"Test  label balance: UP {y_test.mean():.4f}  DOWN {1 - y_test.mean():.4f}")
    print()

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models():
    """Return dict of name -> model."""
    return {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=50,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(name, model, scaler, X_test, y_test):
    """Evaluate a trained model on the test set and print results."""
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"--- {name} ---")
    print(f"  Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Confusion matrix:")
    print(f"    Predicted DOWN  UP")
    print(f"    Actual DOWN  {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"    Actual UP    {cm[1][0]:>5}  {cm[1][1]:>5}")

    # Accuracy at different confidence thresholds
    confidence = np.maximum(proba, 1 - proba)
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        mask = confidence >= thresh
        n = mask.sum()
        if n > 0:
            acc_t = accuracy_score(y_test[mask], y_pred[mask])
            print(f"  Conf >= {thresh:.0%}: {acc_t:.4f} ({acc_t*100:.2f}%) on {n:,} samples ({n/len(y_test)*100:.1f}%)")
        else:
            print(f"  Conf >= {thresh:.0%}: no samples")
    print()
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    feature_cols = load_selected_features()
    X_train, y_train, X_test, y_test = load_data(feature_cols)

    # Scale features (same scaler for all models for fair comparison)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        acc = evaluate(name, model, scaler, X_test, y_test)
        results[name] = (model, acc)

    # Summary
    print("=" * 50)
    print("SUMMARY (out-of-sample accuracy on 20k test candles)")
    print("=" * 50)
    for name, (model, acc) in sorted(results.items(), key=lambda x: -x[1][1]):
        print(f"  {name:15s}  {acc:.4f}  ({acc*100:.2f}%)")

    # Save the best model
    best_name = max(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    print(f"\nBest: {best_name}")

    models_dir = os.path.join(_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    features_path = os.path.join(models_dir, "features.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(features_path, "wb") as f:
        pickle.dump(feature_cols, f)

    print(f"Saved {best_name} to {models_dir}/")
    print(f"  model.pkl, scaler.pkl, features.pkl")


if __name__ == "__main__":
    main()
