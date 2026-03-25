"""Save production model: large-moves LightGBM with top 45 features."""
import os, sys, pickle
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from Training.features import FEATURE_COLS, calculate_features, make_label_large_moves

# Top 45 features from ablation importance ranking
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

# Tuned params
PARAMS = {
    "n_estimators": 500, "max_depth": 6,
    "learning_rate": 0.020784549897083126,
    "num_leaves": 37, "min_child_samples": 112,
    "subsample": 0.8103767331776779,
    "colsample_bytree": 0.5486659787425067,
    "reg_alpha": 0.0034916132094567013,
    "reg_lambda": 0.1118425108233888,
    "random_state": 42, "verbose": -1,
}
PCT_THR = 0.0372654293444779

def eval_test(model, scaler, fc, df, name):
    df = calculate_features(df)
    y = make_label_large_moves(df, PCT_THR)
    df["label"] = y
    df = df.dropna(subset=fc + ["label"]).reset_index(drop=True)
    X, y = df[fc], df["label"]
    Xs = scaler.transform(X)
    yp = model.predict(Xs)
    pr = model.predict_proba(Xs)
    conf = np.maximum(pr[:, 0], pr[:, 1])
    acc = accuracy_score(y, yp)
    print(f"  {name}: {acc:.4f} ({acc*100:.2f}%) on {len(y):,}")
    for t in [0.55, 0.60, 0.65]:
        m = conf >= t
        n = m.sum()
        if n > 0:
            a = accuracy_score(y[m], yp[m])
            print(f"    >= {t:.0%}: {a:.4f} ({a*100:.2f}%) on {n:,} ({n/len(y)*100:.1f}%)")

# Load & train
print(f"Training large-moves model with {len(TOP_45)} features...")
full = pd.read_csv(os.path.join(_ROOT, "data", "btc_candles_100k.csv"))
full = calculate_features(full)
y = make_label_large_moves(full, PCT_THR)
full["label"] = y
full = full.dropna(subset=TOP_45 + ["label"]).reset_index(drop=True)
X, y = full[TOP_45], full["label"]
print(f"Training samples: {len(X):,}")

sc = StandardScaler()
Xsc = sc.fit_transform(X)
mdl = lgb.LGBMClassifier(**PARAMS)
mdl.fit(Xsc, y)

# Evaluate
print("\n=== Test results (45 features, tuned large-moves) ===")
t20k = pd.concat([
    pd.read_csv(os.path.join(_ROOT, "data", "btc_candles_set5.csv")),
    pd.read_csv(os.path.join(_ROOT, "data", "btc_candles_set4.csv")),
], ignore_index=True)
t10k = pd.read_csv(os.path.join(_ROOT, "data", "btc_candles_10k.csv"))
eval_test(mdl, sc, TOP_45, t20k, "20k (set5+4)")
eval_test(mdl, sc, TOP_45, t10k, "10k (recent)")

# Save
md = os.path.join(_ROOT, "models")
os.makedirs(md, exist_ok=True)
pickle.dump(mdl, open(os.path.join(md, "model.pkl"), "wb"))
pickle.dump(sc, open(os.path.join(md, "scaler.pkl"), "wb"))
pickle.dump(TOP_45, open(os.path.join(md, "features.pkl"), "wb"))
print(f"\nSaved: model.pkl, scaler.pkl, features.pkl ({len(TOP_45)} features)")
