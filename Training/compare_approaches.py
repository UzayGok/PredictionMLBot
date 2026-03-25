"""
compare_approaches.py
Compare 3 approaches on OOS test sets:
  A) Current: Large-moves LightGBM (tuned, 45 features)
  B) Stacking ensemble: LightGBM + XGBoost + RandomForest -> LogisticRegression meta
  C) Regression: Predict return magnitude, threshold into trades

All use the same 45 features, large-moves label, tuned LightGBM params as base.
Train on 100k candles, test on 20k (set5+4) and 10k (recent).
"""

import os, sys, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from Training.features import calculate_features, make_label, make_label_large_moves

# ---- Shared config ----
TOP_45 = [
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
PCT_THR = 0.0372654293444779

LGB_PARAMS = {
    "n_estimators": 500, "max_depth": 6,
    "learning_rate": 0.020784549897083126,
    "num_leaves": 37, "min_child_samples": 112,
    "subsample": 0.8103767331776779,
    "colsample_bytree": 0.5486659787425067,
    "reg_alpha": 0.0034916132094567013,
    "reg_lambda": 0.1118425108233888,
    "random_state": 42, "verbose": -1,
}


def load_prep_train(path):
    """Load training data with large-moves filter."""
    df = pd.read_csv(path)
    df = calculate_features(df)
    df["next_return_pct"] = (df["close"].shift(-1) - df["close"]) / df["close"] * 100
    df["label"] = make_label_large_moves(df, PCT_THR)
    df = df.dropna(subset=TOP_45 + ["label", "next_return_pct"]).reset_index(drop=True)
    return df


def load_prep_test(path):
    """Load test data with standard label (ALL candles, not just large moves)."""
    df = pd.read_csv(path)
    df = calculate_features(df)
    df["next_return_pct"] = (df["close"].shift(-1) - df["close"]) / df["close"] * 100
    df["label"] = make_label(df)
    df = df.dropna(subset=TOP_45 + ["label", "next_return_pct"]).reset_index(drop=True)
    return df


def eval_predictions(y_true, y_pred, proba, name, test_name):
    """Print accuracy overall and by confidence band."""
    conf = np.maximum(proba[:, 0], proba[:, 1])
    acc = accuracy_score(y_true, y_pred)
    print(f"  {test_name}: {acc:.4f} ({acc*100:.2f}%) on {len(y_true):,}")
    for t in [0.55, 0.60, 0.65]:
        m = conf >= t
        n = m.sum()
        if n > 0:
            a = accuracy_score(y_true[m], y_pred[m])
            print(f"    >= {t:.0%}: {a:.4f} ({a*100:.2f}%) on {n:,} ({n/len(y_true)*100:.1f}%)")
    return acc


def eval_regression_predictions(y_class_true, predicted_return, test_name, thresholds=[0.01, 0.02, 0.03, 0.04, 0.05]):
    """Evaluate regression model: convert predicted return to trade signals at various thresholds."""
    # Basic: predict UP if predicted_return > 0, DOWN otherwise
    y_pred_basic = (predicted_return > 0).astype(float)
    acc_basic = accuracy_score(y_class_true, y_pred_basic)
    print(f"  {test_name} (sign only): {acc_basic:.4f} ({acc_basic*100:.2f}%) on {len(y_class_true):,}")

    # Threshold: only trade when |predicted_return| > threshold
    for thr in thresholds:
        mask = np.abs(predicted_return) >= thr
        n = mask.sum()
        if n > 0:
            y_pred_t = (predicted_return[mask] > 0).astype(float)
            acc_t = accuracy_score(y_class_true[mask], y_pred_t)
            print(f"    |pred| >= {thr:.2f}%: {acc_t:.4f} ({acc_t*100:.2f}%) on {n:,} ({n/len(y_class_true)*100:.1f}%)")
    return acc_basic


def main():
    print("Loading data...")
    train_df = load_prep_train(os.path.join(_ROOT, "data", "btc_candles_100k.csv"))

    # All test sets: set5..set1 + 10k = ~60k candles (Aug 2025 - Mar 2026)
    # Test on ALL candles (standard label), not just large moves
    test_all_df = pd.concat([
        load_prep_test(os.path.join(_ROOT, "data", f"btc_candles_set{i}.csv"))
        for i in range(5, 0, -1)
    ] + [
        load_prep_test(os.path.join(_ROOT, "data", "btc_candles_10k.csv")),
    ], ignore_index=True)

    X_train = train_df[TOP_45]
    y_train = train_df["label"]
    ret_train = train_df["next_return_pct"]

    X_test, y_test = test_all_df[TOP_45], test_all_df["label"]
    ret_test = test_all_df["next_return_pct"]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xts = scaler.transform(X_test)

    print(f"Train: {len(X_train):,}  |  Test (all 6 files): {len(X_test):,}")
    print(f"Features: {len(TOP_45)}\n")

    # ==================================================================
    # A) Baseline: Large-moves LightGBM (current production model)
    # ==================================================================
    print("=" * 65)
    print("A) BASELINE: Large-moves LightGBM (tuned, 45 features)")
    print("=" * 65)
    mdl_a = lgb.LGBMClassifier(**LGB_PARAMS)
    mdl_a.fit(Xtr, y_train)
    yp = mdl_a.predict(Xts)
    pr = mdl_a.predict_proba(Xts)
    eval_predictions(y_test, yp, pr, "A", "all")

    # ==================================================================
    # B) Ensemble: LGB + XGB + RF (averaged probabilities + meta-model)
    # ==================================================================
    print("\n" + "=" * 65)
    print("B) ENSEMBLE: LightGBM + XGBoost + RandomForest")
    print("=" * 65)
    print("Training 3 base models...")

    mdl_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
    mdl_lgb.fit(Xtr, y_train)

    mdl_xgb = xgb.XGBClassifier(
        n_estimators=500, max_depth=6,
        learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.01,
        reg_lambda=0.1, random_state=42,
        use_label_encoder=False, eval_metric="logloss",
        verbosity=0,
    )
    mdl_xgb.fit(Xtr, y_train)
    print("  XGBoost done.")

    mdl_rf = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        min_samples_leaf=50, max_features="sqrt",
        random_state=42, n_jobs=-1,
    )
    mdl_rf.fit(Xtr, y_train)
    print("  RandomForest done.")

    # B1: Simple average of probabilities
    print("\n--- B1: Simple probability average ---")
    pr_l = mdl_lgb.predict_proba(Xts)
    pr_x = mdl_xgb.predict_proba(Xts)
    pr_r = mdl_rf.predict_proba(Xts)
    pr_avg = (pr_l + pr_x + pr_r) / 3.0
    yp = (pr_avg[:, 1] >= 0.5).astype(float)
    eval_predictions(y_test, yp, pr_avg, "B1", "all")

    # B2: Stacking with held-out meta-features (70/30 split)
    print("\n--- B2: Stacking with LogReg meta-model (70/30 split) ---")
    split = int(len(Xtr) * 0.7)
    Xtr_base, Xtr_meta = Xtr[:split], Xtr[split:]
    ytr_base, ytr_meta = y_train.iloc[:split], y_train.iloc[split:]

    m_lgb2 = lgb.LGBMClassifier(**LGB_PARAMS); m_lgb2.fit(Xtr_base, ytr_base)
    m_xgb2 = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=0.1, random_state=42,
        use_label_encoder=False, eval_metric="logloss", verbosity=0,
    ); m_xgb2.fit(Xtr_base, ytr_base)
    m_rf2 = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=50,
        max_features="sqrt", random_state=42, n_jobs=-1,
    ); m_rf2.fit(Xtr_base, ytr_base)

    # Generate meta-features on held-out 30%
    meta_train = np.column_stack([
        m_lgb2.predict_proba(Xtr_meta)[:, 1],
        m_xgb2.predict_proba(Xtr_meta)[:, 1],
        m_rf2.predict_proba(Xtr_meta)[:, 1],
    ])
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    meta_clf.fit(meta_train, ytr_meta)
    print(f"  Meta-model weights: {meta_clf.coef_[0].round(3)}")

    # For test: use full-data base models, meta-model from held-out
    meta_test = np.column_stack([
        mdl_lgb.predict_proba(Xts)[:, 1],
        mdl_xgb.predict_proba(Xts)[:, 1],
        mdl_rf.predict_proba(Xts)[:, 1],
    ])
    yp = meta_clf.predict(meta_test)
    pr = meta_clf.predict_proba(meta_test)
    eval_predictions(y_test, yp, pr, "B2", "all")

    # ==================================================================
    # C) Regression: Predict return magnitude, threshold into trades
    # ==================================================================
    print("\n" + "=" * 65)
    print("C) REGRESSION: Predict return %, threshold into trade signals")
    print("=" * 65)

    # C1: LightGBM regressor
    print("\n--- C1: LightGBM Regressor ---")
    reg_lgb = lgb.LGBMRegressor(
        n_estimators=500, max_depth=6,
        learning_rate=0.02, num_leaves=37,
        min_child_samples=112, subsample=0.81,
        colsample_bytree=0.55, reg_alpha=0.003,
        reg_lambda=0.11, random_state=42, verbose=-1,
    )
    reg_lgb.fit(Xtr, ret_train)

    pred_ret_lgb = reg_lgb.predict(Xts)
    eval_regression_predictions(y_test.values, pred_ret_lgb, "all")

    # C2: Ridge regression (linear baseline)
    print("\n--- C2: Ridge Regression ---")
    reg_ridge = Ridge(alpha=1.0)
    reg_ridge.fit(Xtr, ret_train)

    pred_ret_ridge = reg_ridge.predict(Xts)
    eval_regression_predictions(y_test.values, pred_ret_ridge, "all")

    # C3: Hybrid — regression confidence as filter on classification
    print("\n--- C3: Hybrid (LGB classifier + LGB regressor filter) ---")
    print("  Trade only when classifier AND regressor agree on direction")
    cls_proba = mdl_a.predict_proba(Xts)
    cls_pred = mdl_a.predict(Xts)
    cls_conf = np.maximum(cls_proba[:, 0], cls_proba[:, 1])
    reg_pred_ret = reg_lgb.predict(Xts)
    reg_direction = (reg_pred_ret > 0).astype(float)

    agree = cls_pred == reg_direction
    for conf_t in [0.50, 0.55, 0.60]:
        mask = agree & (cls_conf >= conf_t)
        n = mask.sum()
        if n > 0:
            a = accuracy_score(y_test[mask], cls_pred[mask])
            print(f"  agree + cls>={conf_t:.0%}: {a:.4f} ({a*100:.2f}%) on {n:,} ({n/len(y_test)*100:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
