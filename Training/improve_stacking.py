"""
improve_stacking.py
Experiment: improve the direction model's stacking ensemble.
  A) Baseline = current B2 (LGB+XGB+RF → LogReg meta, 3 proba features)
  B) Expanded ensemble: add CatBoost + ExtraTrees (5 base models → LogReg)
  C) Rich meta-learner: 5 base probas + top raw features → LogReg/GBM meta
  D) Feature re-selection: test different feature subsets for base models

Trains on 100k, evaluates on 10k only.
"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from Training.features import (
    calculate_features, make_label, make_label_large_moves,
    DIRECTION_FEATURES, FEATURE_COLS,
)

PCT_THR = 0.0372654293444779
MAG_THRESHOLD = 0.10

MAG_PARAMS = {
    "n_estimators": 400, "max_depth": 3, "learning_rate": 0.0183,
    "num_leaves": 37, "min_child_samples": 46,
    "subsample": 0.80, "colsample_bytree": 0.76,
    "reg_alpha": 0.74, "reg_lambda": 0.07,
    "random_state": 42, "verbose": -1,
}

DIR_PARAMS = {
    "n_estimators": 500, "max_depth": 6,
    "learning_rate": 0.020784549897083126,
    "num_leaves": 37, "min_child_samples": 112,
    "subsample": 0.8103767331776779,
    "colsample_bytree": 0.5486659787425067,
    "reg_alpha": 0.0034916132094567013,
    "reg_lambda": 0.1118425108233888,
    "random_state": 42, "verbose": -1,
}

MAG_PROBA_THR = 0.50
DIR_CONF_THR = 0.55

# Top features most likely to help the meta-learner (from feature importance)
TOP_RAW_FEATS = [
    "lag_return_5", "trend_strength_24h", "body_ratio", "return_24h",
    "atr_regime", "volume_ratio", "lag_return_3", "high_low_ratio",
]


def load_data(path):
    df = pd.read_csv(path)
    df = calculate_features(df)
    df["direction_label"] = make_label(df)
    df["lm_label"] = make_label_large_moves(df, PCT_THR)
    df["next_return_pct"] = (df["close"].shift(-1) - df["close"]) / df["close"] * 100
    df["next_abs_return"] = df["next_return_pct"].abs()
    df = df.dropna(subset=DIRECTION_FEATURES + ["direction_label"]).reset_index(drop=True)
    return df


def eval_two_stage(mag_proba, dir_pred, dir_conf, y_true, label):
    """Print results grid for a configuration."""
    print(f"\n--- {label} ---")
    for mt in [0.4, 0.5]:
        for dc in [0.53, 0.55, 0.57]:
            m = (mag_proba >= mt) & (dir_conf >= dc)
            nn = m.sum()
            if nn > 0:
                a = accuracy_score(y_true[m], dir_pred[m])
                tag = " <<<" if mt == MAG_PROBA_THR and dc == DIR_CONF_THR else ""
                print(f"  mag>={mt:.2f}, dir>={dc:.2f}: {a*100:.2f}% on {nn:>4} trades{tag}")
    # Direction-only
    for dc in [0.55, 0.57]:
        m = dir_conf >= dc
        nn = m.sum()
        if nn > 0:
            a = accuracy_score(y_true[m], dir_pred[m])
            print(f"  dir-only>={dc:.2f}:      {a*100:.2f}% on {nn:>4} trades")


def train_base_models_3(X_base, y_base):
    """Current 3-model ensemble."""
    d_lgb = lgb.LGBMClassifier(**DIR_PARAMS); d_lgb.fit(X_base, y_base)
    d_xgb = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=0.1, random_state=42,
        use_label_encoder=False, eval_metric="logloss", verbosity=0,
    ); d_xgb.fit(X_base, y_base)
    d_rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=50,
        max_features="sqrt", random_state=42, n_jobs=-1,
    ); d_rf.fit(X_base, y_base)
    return [d_lgb, d_xgb, d_rf]


def train_base_models_5(X_base, y_base):
    """Expanded 5-model ensemble: LGB + XGB + RF + CatBoost + ExtraTrees."""
    d_lgb = lgb.LGBMClassifier(**DIR_PARAMS); d_lgb.fit(X_base, y_base)
    d_xgb = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=0.1, random_state=42,
        use_label_encoder=False, eval_metric="logloss", verbosity=0,
    ); d_xgb.fit(X_base, y_base)
    d_rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=50,
        max_features="sqrt", random_state=42, n_jobs=-1,
    ); d_rf.fit(X_base, y_base)
    d_cb = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.02,
        l2_leaf_reg=3.0, subsample=0.8, random_seed=42,
        verbose=0, allow_writing_files=False,
    ); d_cb.fit(X_base, y_base)
    d_et = ExtraTreesClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=50,
        max_features="sqrt", random_state=42, n_jobs=-1,
    ); d_et.fit(X_base, y_base)
    return [d_lgb, d_xgb, d_rf, d_cb, d_et]


def get_meta_probas(models, X):
    """Stack predict_proba[:, 1] from all models."""
    return np.column_stack([m.predict_proba(X)[:, 1] for m in models])


def main():
    print("=" * 60)
    print("STACKING IMPROVEMENT EXPERIMENT")
    print("=" * 60)

    # ---- Load data ----
    print("\nLoading training data (100k)...")
    train_df = load_data(os.path.join(_ROOT, "data", "btc_candles_100k.csv"))
    print(f"  Training samples: {len(train_df):,}")

    print("Loading test data (10k)...")
    test_df = load_data(os.path.join(_ROOT, "data", "btc_candles_10k.csv"))
    print(f"  Test samples: {len(test_df):,}")

    feats = DIRECTION_FEATURES

    # ---- Stage 1: Magnitude model (same for all experiments) ----
    print("\nTraining magnitude model (shared across all experiments)...")
    mag_scaler = StandardScaler()
    X_mag_train = mag_scaler.fit_transform(train_df[feats])
    y_mag_train = (train_df["next_abs_return"] > MAG_THRESHOLD).astype(int).values
    mag_model = lgb.LGBMClassifier(**MAG_PARAMS)
    mag_model.fit(X_mag_train, y_mag_train)

    X_mag_test = mag_scaler.transform(test_df[feats])
    mag_proba = mag_model.predict_proba(X_mag_test)[:, 1]
    y_true = test_df["direction_label"].values

    # ---- Direction data: large-moves only ----
    train_lm = train_df.dropna(subset=["lm_label"]).reset_index(drop=True)
    dir_scaler = StandardScaler()
    X_dir_all = dir_scaler.fit_transform(train_lm[feats])
    y_dir_all = train_lm["lm_label"].values

    # 70/30 split for meta-learner
    sp = int(len(X_dir_all) * 0.7)
    X_base, X_meta = X_dir_all[:sp], X_dir_all[sp:]
    y_base, y_meta = y_dir_all[:sp], y_dir_all[sp:]

    X_test = dir_scaler.transform(test_df[feats])

    # Also prepare raw features for rich meta-learner
    raw_meta_scaler = StandardScaler()
    raw_meta_train = raw_meta_scaler.fit_transform(train_lm[TOP_RAW_FEATS])
    raw_meta_base, raw_meta_hold = raw_meta_train[:sp], raw_meta_train[sp:]
    raw_meta_test = raw_meta_scaler.transform(test_df[TOP_RAW_FEATS])

    # =================================================================
    # A) BASELINE: current B2 (3 models → LogReg on 3 probas)
    # =================================================================
    print("\n[A] Training baseline B2 (3 models → LogReg)...")
    models_3_base = train_base_models_3(X_base, y_base)
    meta_3_hold = get_meta_probas(models_3_base, X_meta)
    meta_A = LogisticRegression(max_iter=1000, random_state=42)
    meta_A.fit(meta_3_hold, y_meta)

    # Retrain on full data for test eval
    models_3_full = train_base_models_3(X_dir_all, y_dir_all)
    meta_3_test = get_meta_probas(models_3_full, X_test)
    pred_A = meta_A.predict(meta_3_test)
    proba_A = meta_A.predict_proba(meta_3_test)
    conf_A = np.maximum(proba_A[:, 0], proba_A[:, 1])
    eval_two_stage(mag_proba, pred_A, conf_A, y_true, "A) Baseline B2 (3→LogReg)")

    # =================================================================
    # B) EXPANDED ENSEMBLE: 5 models → LogReg on 5 probas
    # =================================================================
    print("\n[B] Training expanded ensemble (5 models → LogReg)...")
    models_5_base = train_base_models_5(X_base, y_base)
    meta_5_hold = get_meta_probas(models_5_base, X_meta)
    meta_B = LogisticRegression(max_iter=1000, random_state=42)
    meta_B.fit(meta_5_hold, y_meta)

    models_5_full = train_base_models_5(X_dir_all, y_dir_all)
    meta_5_test = get_meta_probas(models_5_full, X_test)
    pred_B = meta_B.predict(meta_5_test)
    proba_B = meta_B.predict_proba(meta_5_test)
    conf_B = np.maximum(proba_B[:, 0], proba_B[:, 1])
    eval_two_stage(mag_proba, pred_B, conf_B, y_true, "B) Expanded (5→LogReg)")

    # =================================================================
    # C) RICH META-LEARNER: 5 probas + 8 raw features → LogReg
    # =================================================================
    print("\n[C] Training rich meta-learner (5 probas + 8 raw feats → LogReg)...")
    rich_hold = np.hstack([meta_5_hold, raw_meta_hold])
    meta_C = LogisticRegression(max_iter=1000, random_state=42)
    meta_C.fit(rich_hold, y_meta)

    rich_test = np.hstack([meta_5_test, raw_meta_test])
    pred_C = meta_C.predict(rich_test)
    proba_C = meta_C.predict_proba(rich_test)
    conf_C = np.maximum(proba_C[:, 0], proba_C[:, 1])
    eval_two_stage(mag_proba, pred_C, conf_C, y_true, "C) Rich meta (5 probas + 8 raw → LogReg)")

    # =================================================================
    # D) RICH META with LGB meta-learner instead of LogReg
    # =================================================================
    print("\n[D] Training rich meta with LGB meta-learner...")
    meta_D = lgb.LGBMClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        num_leaves=8, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.7,
        random_state=42, verbose=-1,
    )
    meta_D.fit(rich_hold, y_meta)

    pred_D = meta_D.predict(rich_test)
    proba_D = meta_D.predict_proba(rich_test)
    conf_D = np.maximum(proba_D[:, 0], proba_D[:, 1])
    eval_two_stage(mag_proba, pred_D, conf_D, y_true, "D) Rich meta (5 probas + 8 raw → LGB meta)")

    # =================================================================
    # E) 3 base models + rich meta (to isolate improvement source)
    # =================================================================
    print("\n[E] Training 3 models + rich meta (isolate meta-learner effect)...")
    rich3_hold = np.hstack([meta_3_hold, raw_meta_hold])
    meta_E = LogisticRegression(max_iter=1000, random_state=42)
    meta_E.fit(rich3_hold, y_meta)

    rich3_test = np.hstack([meta_3_test, raw_meta_test])
    pred_E = meta_E.predict(rich3_test)
    proba_E = meta_E.predict_proba(rich3_test)
    conf_E = np.maximum(proba_E[:, 0], proba_E[:, 1])
    eval_two_stage(mag_proba, pred_E, conf_E, y_true, "E) 3 models + rich meta (isolate meta effect)")

    # =================================================================
    # SUMMARY TABLE
    # =================================================================
    print("\n" + "=" * 60)
    print("SUMMARY  —  Production threshold (mag>=0.50, dir>=0.55)")
    print("=" * 60)

    configs = [
        ("A) Baseline B2 (3→LogReg)", pred_A, conf_A),
        ("B) Expanded (5→LogReg)", pred_B, conf_B),
        ("C) Rich meta (5+8→LogReg)", pred_C, conf_C),
        ("D) Rich meta (5+8→LGB)", pred_D, conf_D),
        ("E) 3 models + rich meta", pred_E, conf_E),
    ]
    print(f"\n{'Config':<35} {'Accuracy':>8} {'Trades':>7}")
    print("-" * 55)
    for label, pred, conf in configs:
        m = (mag_proba >= MAG_PROBA_THR) & (conf >= DIR_CONF_THR)
        nn = m.sum()
        if nn > 0:
            a = accuracy_score(y_true[m], pred[m])
            print(f"{label:<35} {a*100:>7.2f}% {nn:>6}")
        else:
            print(f"{label:<35}    N/A      0")

    # Also show at dir>=0.53 for more trades
    print(f"\nAt mag>=0.50, dir>=0.53:")
    print(f"{'Config':<35} {'Accuracy':>8} {'Trades':>7}")
    print("-" * 55)
    for label, pred, conf in configs:
        m = (mag_proba >= MAG_PROBA_THR) & (conf >= 0.53)
        nn = m.sum()
        if nn > 0:
            a = accuracy_score(y_true[m], pred[m])
            print(f"{label:<35} {a*100:>7.2f}% {nn:>6}")

    # Direction-only at 0.55
    print(f"\nDirection-only (no mag filter), dir>=0.55:")
    print(f"{'Config':<35} {'Accuracy':>8} {'Trades':>7}")
    print("-" * 55)
    for label, pred, conf in configs:
        m = conf >= DIR_CONF_THR
        nn = m.sum()
        if nn > 0:
            a = accuracy_score(y_true[m], pred[m])
            print(f"{label:<35} {a*100:>7.2f}% {nn:>6}")


if __name__ == "__main__":
    main()
