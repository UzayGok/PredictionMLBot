"""
Test if large price moves are predictable.
Binary classifier: will |next candle return| > threshold?
If this has signal, a two-stage approach makes sense.
"""
import os, sys, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from Training.features import calculate_features, make_label, FEATURE_COLS

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


def load(path):
    df = pd.read_csv(path)
    df = calculate_features(df)
    df["next_return_pct"] = (df["close"].shift(-1) - df["close"]) / df["close"] * 100
    df["next_abs_return"] = df["next_return_pct"].abs()
    df["direction_label"] = make_label(df)
    df = df.dropna(subset=TOP_45 + ["next_return_pct", "direction_label"]).reset_index(drop=True)
    return df


print("Loading data...")
train_df = load(os.path.join(_ROOT, "data", "btc_candles_100k.csv"))
test_df = pd.concat([
    load(os.path.join(_ROOT, "data", f"btc_candles_set{i}.csv"))
    for i in range(5, 0, -1)
] + [
    load(os.path.join(_ROOT, "data", "btc_candles_10k.csv")),
], ignore_index=True)

print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}\n")

# Distribution of next-candle absolute returns
print("=== Distribution of |next candle return| ===")
for pct in [0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20]:
    n_train = (train_df["next_abs_return"] > pct).sum()
    n_test = (test_df["next_abs_return"] > pct).sum()
    print(f"  > {pct:.2f}%: train {n_train:,} ({n_train/len(train_df)*100:.1f}%) | test {n_test:,} ({n_test/len(test_df)*100:.1f}%)")

scaler = StandardScaler()
Xtr = scaler.fit_transform(train_df[TOP_45])
Xts = scaler.transform(test_df[TOP_45])

# Test multiple thresholds for "big move" definition
print("\n=== Stage 1: Can we predict 'big move coming'? ===")
for threshold in [0.03, 0.05, 0.07, 0.10]:
    y_train_mag = (train_df["next_abs_return"] > threshold).astype(int)
    y_test_mag = (test_df["next_abs_return"] > threshold).astype(int)
    base_rate = y_test_mag.mean()

    mdl = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        num_leaves=31, min_child_samples=100,
        subsample=0.8, colsample_bytree=0.6,
        reg_alpha=0.01, reg_lambda=0.1,
        random_state=42, verbose=-1,
    )
    mdl.fit(Xtr, y_train_mag)
    y_pred = mdl.predict(Xts)
    proba = mdl.predict_proba(Xts)[:, 1]

    acc = accuracy_score(y_test_mag, y_pred)
    prec = precision_score(y_test_mag, y_pred, zero_division=0)
    rec = recall_score(y_test_mag, y_pred, zero_division=0)
    n_pred_big = y_pred.sum()

    print(f"\n  Threshold: |return| > {threshold:.2f}%")
    print(f"  Base rate (test): {base_rate:.3f} ({base_rate*100:.1f}%)")
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"  Predicted 'big': {n_pred_big:,} / {len(y_test_mag):,}")

    # Top-probability filter: when model is most confident a big move is coming
    for prob_t in [0.3, 0.4, 0.5, 0.6]:
        mask = proba >= prob_t
        n = mask.sum()
        if n > 0:
            actual_big_rate = y_test_mag[mask].mean()
            print(f"    proba >= {prob_t}: {n:,} candles, {actual_big_rate:.3f} actual big-move rate (vs {base_rate:.3f} base)")

# === Two-stage test ===
print("\n\n=== STAGE 2: Two-stage approach ===")
print("Stage 1 filters for big moves, Stage 2 predicts direction\n")

# Direction model (our existing B2 stacking approach components)
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from Training.features import make_label_large_moves

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

# Train direction models on large-moves-filtered training data
y_train_lm = make_label_large_moves(train_df, PCT_THR)
train_lm = train_df.copy()
train_lm["lm_label"] = y_train_lm
train_lm = train_lm.dropna(subset=["lm_label"]).reset_index(drop=True)
Xtr_lm = scaler.fit_transform(train_lm[TOP_45])

# B2 stacking components
split = int(len(Xtr_lm) * 0.7)
Xtr_base, Xtr_meta = Xtr_lm[:split], Xtr_lm[split:]
ytr_base = train_lm["lm_label"].iloc[:split]
ytr_meta = train_lm["lm_label"].iloc[split:]

m_lgb = lgb.LGBMClassifier(**LGB_PARAMS); m_lgb.fit(Xtr_base, ytr_base)
m_xgb = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.8,
    colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=0.1, random_state=42,
    use_label_encoder=False, eval_metric="logloss", verbosity=0,
); m_xgb.fit(Xtr_base, ytr_base)
m_rf = RandomForestClassifier(
    n_estimators=100, max_depth=8, min_samples_leaf=50,
    max_features="sqrt", random_state=42, n_jobs=-1,
); m_rf.fit(Xtr_base, ytr_base)

meta_train = np.column_stack([
    m_lgb.predict_proba(Xtr_meta)[:, 1],
    m_xgb.predict_proba(Xtr_meta)[:, 1],
    m_rf.predict_proba(Xtr_meta)[:, 1],
])
meta_clf = LogisticRegression(max_iter=1000, random_state=42)
meta_clf.fit(meta_train, ytr_meta)

# Full-data direction models
m_lgb_full = lgb.LGBMClassifier(**LGB_PARAMS); m_lgb_full.fit(Xtr_lm, train_lm["lm_label"])
m_xgb_full = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.8,
    colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=0.1, random_state=42,
    use_label_encoder=False, eval_metric="logloss", verbosity=0,
); m_xgb_full.fit(Xtr_lm, train_lm["lm_label"])
m_rf_full = RandomForestClassifier(
    n_estimators=100, max_depth=8, min_samples_leaf=50,
    max_features="sqrt", random_state=42, n_jobs=-1,
); m_rf_full.fit(Xtr_lm, train_lm["lm_label"])

# Rescale test with same scaler
Xts2 = scaler.transform(test_df[TOP_45])
y_test_dir = test_df["direction_label"]

# B2 direction predictions on test
meta_test = np.column_stack([
    m_lgb_full.predict_proba(Xts2)[:, 1],
    m_xgb_full.predict_proba(Xts2)[:, 1],
    m_rf_full.predict_proba(Xts2)[:, 1],
])
dir_pred = meta_clf.predict(meta_test)
dir_proba = meta_clf.predict_proba(meta_test)
dir_conf = np.maximum(dir_proba[:, 0], dir_proba[:, 1])

# Magnitude model predictions
for mag_thr in [0.05, 0.07, 0.10]:
    y_train_mag = (train_df["next_abs_return"] > mag_thr).astype(int)
    mag_mdl = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        num_leaves=31, min_child_samples=100,
        subsample=0.8, colsample_bytree=0.6,
        reg_alpha=0.01, reg_lambda=0.1,
        random_state=42, verbose=-1,
    )
    # Use original scaler for magnitude model
    Xtr_orig = StandardScaler().fit_transform(train_df[TOP_45])
    Xts_orig = StandardScaler().fit(train_df[TOP_45]).transform(test_df[TOP_45])
    mag_mdl.fit(Xtr_orig, y_train_mag)
    mag_proba = mag_mdl.predict_proba(Xts_orig)[:, 1]

    print(f"\n--- Two-stage: magnitude threshold > {mag_thr:.2f}% ---")

    # Combine: trade when mag model says "big" AND direction model is confident
    for mag_p in [0.3, 0.4, 0.5]:
        for dir_c in [0.50, 0.55]:
            mask = (mag_proba >= mag_p) & (dir_conf >= dir_c)
            n = mask.sum()
            if n > 50:
                acc = accuracy_score(y_test_dir[mask], dir_pred[mask])
                print(f"  mag>={mag_p} + dir>={dir_c:.0%}: {acc:.4f} ({acc*100:.2f}%) on {n:,} ({n/len(y_test_dir)*100:.1f}%)")

# Baseline comparison
print(f"\n--- Baseline: B2 direction model alone ---")
for dir_c in [0.50, 0.55]:
    mask = dir_conf >= dir_c
    n = mask.sum()
    acc = accuracy_score(y_test_dir[mask], dir_pred[mask])
    print(f"  dir>={dir_c:.0%}: {acc:.4f} ({acc*100:.2f}%) on {n:,} ({n/len(y_test_dir)*100:.1f}%)")

print("\nDone.")
