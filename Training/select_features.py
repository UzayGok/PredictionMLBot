"""
select_features.py
Rank features by importance using time-series cross-validation on the training
data. Uses 5 chronological folds (no shuffling) to get stable importance scores.

Outputs:
  - Ranked feature list with importance scores
  - Accuracy curve: how accuracy changes as we add features one by one
  - Saves the optimal feature subset to models/features.pkl

Usage:
    python Training/select_features.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from Training.features import FEATURE_COLS, prepare_dataset


def time_series_cv_importance(X, y, n_splits=5):
    """
    Train LightGBM on chronological folds and collect feature importances.
    Returns mean importance per feature and mean OOS accuracy per fold.
    """
    n = len(X)
    fold_size = n // (n_splits + 1)

    importances = []
    fold_accs = []

    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        val_start = train_end
        val_end = min(train_end + fold_size, n)
        if val_end <= val_start:
            break

        X_tr = X.iloc[:train_end]
        y_tr = y.iloc[:train_end]
        X_val = X.iloc[val_start:val_end]
        y_val = y.iloc[val_start:val_end]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1,
        )
        model.fit(X_tr_s, y_tr)

        y_pred = model.predict(X_val_s)
        acc = accuracy_score(y_val, y_pred)
        fold_accs.append(acc)
        importances.append(model.feature_importances_)

        print(f"  Fold {i+1}: train {len(X_tr):,}, val {len(X_val):,}, acc {acc:.4f}")

    mean_imp = np.mean(importances, axis=0)
    return mean_imp, fold_accs


def incremental_accuracy(X, y, ranked_features, steps=None):
    """
    For each number of top-N features, train on first 80% and test on last 20%.
    Returns list of (n_features, accuracy).
    """
    n = len(X)
    split = int(n * 0.8)
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_val, y_val = X.iloc[split:], y.iloc[split:]

    if steps is None:
        steps = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, len(ranked_features)]
        steps = sorted(set(s for s in steps if s <= len(ranked_features)))

    results = []
    for n_feat in steps:
        cols = ranked_features[:n_feat]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[cols])
        X_val_s = scaler.transform(X_val[cols])

        model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1,
        )
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_val_s)
        acc = accuracy_score(y_val, y_pred)
        results.append((n_feat, acc))
        print(f"  Top {n_feat:>2} features: {acc:.4f} ({acc*100:.2f}%)")

    return results


def main():
    # Load training data
    train_path = os.path.join(_ROOT, "data", "btc_candles_100k.csv")
    train_df = pd.read_csv(train_path)
    X, y = prepare_dataset(train_df)
    print(f"Samples: {len(X):,}, Features: {len(FEATURE_COLS)}\n")

    # Step 1: Rank features by cross-validated importance
    print("=== Cross-validated feature importance (5 folds) ===")
    mean_imp, fold_accs = time_series_cv_importance(X, y, n_splits=5)
    print(f"  Mean fold accuracy: {np.mean(fold_accs):.4f}\n")

    ranking = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": mean_imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("=== Feature ranking ===")
    for i, row in ranking.iterrows():
        bar = "#" * int(row["importance"] / ranking["importance"].max() * 40)
        print(f"  {i+1:>2}. {row['feature']:25s}  {row['importance']:8.1f}  {bar}")

    ranked_features = ranking["feature"].tolist()

    # Step 2: Incremental accuracy — find the sweet spot
    print("\n=== Incremental accuracy (top-N features, 80/20 split) ===")
    results = incremental_accuracy(X, y, ranked_features)

    # Find best N
    best_n, best_acc = max(results, key=lambda x: x[1])
    print(f"\n  Best: top {best_n} features at {best_acc:.4f} ({best_acc*100:.2f}%)")

    selected = ranked_features[:best_n]
    print(f"\n=== Selected features ({best_n}) ===")
    for f in selected:
        print(f"  - {f}")

    # Save
    models_dir = os.path.join(_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "features.pkl"), "wb") as f:
        pickle.dump(selected, f)
    print(f"\nSaved {best_n} features to models/features.pkl")


if __name__ == "__main__":
    main()
