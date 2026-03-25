"""
tune_model.py
Bayesian hyperparameter tuning with Optuna for LightGBM.
Tests both label strategies: all moves vs large moves only.
Uses time-based train/val split on the 100k training data.

Usage:
    python Training/tune_model.py
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from Training.features import (
    FEATURE_COLS, calculate_features, make_label, make_label_large_moves,
)


def load_train_val():
    """Load 100k, split 80/20 chronologically for tuning."""
    df = pd.read_csv(os.path.join(_ROOT, "data", "btc_candles_100k.csv"))
    df = calculate_features(df)

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    val_df = df.iloc[split:].copy()
    return train_df, val_df, df


def prepare_xy(df, label_func, feature_cols):
    y = label_func(df)
    df = df.copy()
    df["label"] = y
    df = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
    X = df[feature_cols]
    y = df["label"]
    return X, y


def objective_all_moves(trial, Xtr_scaled, y_train, Xva_scaled, y_val):
    """Optuna objective for all-moves labeling (pre-scaled data)."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 50),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(Xtr_scaled, y_train)
    y_pred = model.predict(Xva_scaled)
    return accuracy_score(y_val, y_pred)


def objective_large_moves(trial, train_df, val_df, feature_cols):
    """Optuna objective for large-moves-only labeling."""
    pct_threshold = trial.suggest_float("pct_threshold", 0.01, 0.10)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 50),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }

    label_fn = lambda df: make_label_large_moves(df, pct_threshold)
    X_train, y_train = prepare_xy(train_df, label_fn, feature_cols)
    X_val, y_val = prepare_xy(val_df, label_fn, feature_cols)

    if len(X_val) < 100:
        return 0.0  # too few samples

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)

    model = lgb.LGBMClassifier(**params)
    model.fit(Xtr, y_train)

    y_pred = model.predict(Xva)
    return accuracy_score(y_val, y_pred)


def evaluate_on_test(model, scaler, feature_cols, test_path, label_func, name):
    """Evaluate a model on a test set and print results."""
    df = pd.read_csv(test_path)
    df = calculate_features(df)
    y_all = label_func(df)
    df["label"] = y_all
    df = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
    X = df[feature_cols]
    y = df["label"]

    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)
    proba = model.predict_proba(Xs)
    confidence = np.maximum(proba[:, 0], proba[:, 1])

    acc = accuracy_score(y, y_pred)
    print(f"  {name}: {acc:.4f} ({acc*100:.2f}%) on {len(y):,} samples")
    for thresh in [0.55, 0.60, 0.65]:
        mask = confidence >= thresh
        n = mask.sum()
        if n > 0:
            acc_t = accuracy_score(y[mask], y_pred[mask])
            print(f"    >= {thresh:.0%}: {acc_t:.4f} ({acc_t*100:.2f}%) on {n:,} ({n/len(y)*100:.1f}%)")
    return acc


def main():
    print("Loading and preparing data...")
    train_df, val_df, full_df = load_train_val()
    feature_cols = FEATURE_COLS
    print(f"Features: {len(feature_cols)}")
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}\n")

    # Pre-compute for strategy 1
    X_train_1, y_train_1 = prepare_xy(train_df, make_label, feature_cols)
    X_val_1, y_val_1 = prepare_xy(val_df, make_label, feature_cols)
    scaler_pre = StandardScaler()
    Xtr1_s = scaler_pre.fit_transform(X_train_1)
    Xva1_s = scaler_pre.transform(X_val_1)

    # ---- Strategy 1: All moves ----
    print("=" * 60)
    print("STRATEGY 1: All moves (standard label)")
    print("=" * 60)
    study1 = optuna.create_study(direction="maximize", study_name="all_moves")
    study1.optimize(
        lambda trial: objective_all_moves(trial, Xtr1_s, y_train_1, Xva1_s, y_val_1),
        n_trials=50,
        show_progress_bar=True,
        catch=(KeyboardInterrupt,),
    )
    print(f"Best val accuracy: {study1.best_value:.4f}")
    print(f"Best params: {study1.best_params}\n")

    # ---- Strategy 2: Large moves only ----
    print("=" * 60)
    print("STRATEGY 2: Large moves only (dead zone filter)")
    print("=" * 60)
    study2 = optuna.create_study(direction="maximize", study_name="large_moves")
    study2.optimize(
        lambda trial: objective_large_moves(trial, train_df, val_df, feature_cols),
        n_trials=50,
        show_progress_bar=True,
        catch=(KeyboardInterrupt,),
    )
    print(f"Best val accuracy: {study2.best_value:.4f}")
    best2 = study2.best_params
    print(f"Best params: {best2}\n")

    # ---- Train final models on full 100k and evaluate on test sets ----
    print("\n" + "=" * 60)
    print("FINAL EVALUATION on test sets")
    print("=" * 60)
    test_files = {
        "20k (set5+4)": [
            os.path.join(_ROOT, "data", "btc_candles_set5.csv"),
            os.path.join(_ROOT, "data", "btc_candles_set4.csv"),
        ],
        "10k (recent)": [
            os.path.join(_ROOT, "data", "btc_candles_10k.csv"),
        ],
    }

    # Strategy 1 final model
    print("\n--- Strategy 1: All moves (tuned) ---")
    p1 = {k: v for k, v in study1.best_params.items()}
    p1["random_state"] = 42
    p1["verbose"] = -1
    X_full, y_full = prepare_xy(full_df, make_label, feature_cols)
    scaler1 = StandardScaler()
    Xf1 = scaler1.fit_transform(X_full)
    model1 = lgb.LGBMClassifier(**p1)
    model1.fit(Xf1, y_full)

    for test_name, paths in test_files.items():
        dfs = [pd.read_csv(p) for p in paths]
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(os.path.join(_ROOT, "data", "_tmp_test.csv"), index=False)
        evaluate_on_test(model1, scaler1, feature_cols,
                         os.path.join(_ROOT, "data", "_tmp_test.csv"),
                         make_label, test_name)

    # Strategy 2 final model
    print("\n--- Strategy 2: Large moves (tuned) ---")
    pct_thr = best2.pop("pct_threshold")
    p2 = {k: v for k, v in best2.items()}
    p2["random_state"] = 42
    p2["verbose"] = -1
    label_fn2 = lambda df: make_label_large_moves(df, pct_thr)
    X_full2, y_full2 = prepare_xy(full_df, label_fn2, feature_cols)
    scaler2 = StandardScaler()
    Xf2 = scaler2.fit_transform(X_full2)
    model2 = lgb.LGBMClassifier(**p2)
    model2.fit(Xf2, y_full2)
    print(f"  Dead zone threshold: ±{pct_thr:.4f}%")
    print(f"  Training samples after filter: {len(X_full2):,} (dropped {len(full_df) - len(X_full2) - 300:,} small moves)")

    for test_name, paths in test_files.items():
        dfs = [pd.read_csv(p) for p in paths]
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(os.path.join(_ROOT, "data", "_tmp_test.csv"), index=False)
        evaluate_on_test(model2, scaler2, feature_cols,
                         os.path.join(_ROOT, "data", "_tmp_test.csv"),
                         label_fn2, test_name)

    # Cleanup temp file
    tmp = os.path.join(_ROOT, "data", "_tmp_test.csv")
    if os.path.exists(tmp):
        os.remove(tmp)

    # Save the best overall model
    print("\n" + "=" * 60)
    print("Saving models...")
    models_dir = os.path.join(_ROOT, "models")

    # Save Strategy 1 (all moves, tuned)
    with open(os.path.join(models_dir, "model.pkl"), "wb") as f:
        pickle.dump(model1, f)
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler1, f)
    with open(os.path.join(models_dir, "features.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    print("Saved tuned all-moves model to models/model.pkl")

    # Save Strategy 2 (large moves) separately
    with open(os.path.join(models_dir, "model_large_moves.pkl"), "wb") as f:
        pickle.dump(model2, f)
    with open(os.path.join(models_dir, "scaler_large_moves.pkl"), "wb") as f:
        pickle.dump(scaler2, f)
    with open(os.path.join(models_dir, "threshold_large_moves.pkl"), "wb") as f:
        pickle.dump(pct_thr, f)
    print(f"Saved tuned large-moves model to models/model_large_moves.pkl (threshold: ±{pct_thr:.4f}%)")

    # Save best params for reference
    with open(os.path.join(models_dir, "best_params_all.pkl"), "wb") as f:
        pickle.dump(study1.best_params, f)
    with open(os.path.join(models_dir, "best_params_large.pkl"), "wb") as f:
        best2["pct_threshold"] = pct_thr
        pickle.dump(best2, f)


if __name__ == "__main__":
    main()
