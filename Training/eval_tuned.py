"""Quick final eval with the tuned params from Optuna."""
import os, sys, pickle
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from Training.features import FEATURE_COLS, calculate_features, make_label, make_label_large_moves

def prepare_xy(df, label_func, feature_cols):
    y = label_func(df)
    df = df.copy()
    df["label"] = y
    df = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
    return df[feature_cols], df["label"]

def eval_model(model, scaler, fc, test_df, label_func, name):
    test_df = calculate_features(test_df)
    X, y = prepare_xy(test_df, label_func, fc)
    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)
    proba = model.predict_proba(Xs)
    conf = np.maximum(proba[:, 0], proba[:, 1])
    acc = accuracy_score(y, y_pred)
    print(f"  {name}: {acc:.4f} ({acc*100:.2f}%) on {len(y):,} samples")
    for t in [0.55, 0.60, 0.65]:
        m = conf >= t
        n = m.sum()
        if n > 0:
            a = accuracy_score(y[m], y_pred[m])
            print(f"    >= {t:.0%}: {a:.4f} ({a*100:.2f}%) on {n:,} ({n/len(y)*100:.1f}%)")


# Load full training data
print("Loading data...")
full_df = pd.read_csv(os.path.join(_ROOT, "data", "btc_candles_100k.csv"))
full_df = calculate_features(full_df)

test_20k = pd.concat([
    pd.read_csv(os.path.join(_ROOT, "data", "btc_candles_set5.csv")),
    pd.read_csv(os.path.join(_ROOT, "data", "btc_candles_set4.csv")),
], ignore_index=True)
test_10k = pd.read_csv(os.path.join(_ROOT, "data", "btc_candles_10k.csv"))

fc = FEATURE_COLS

# ---- Strategy 1: All moves (tuned params from Optuna) ----
print("\n=== Strategy 1: All moves (Optuna-tuned) ===")
p1 = {
    "n_estimators": 500, "max_depth": 7,
    "learning_rate": 0.02027479146436495,
    "num_leaves": 33, "min_child_samples": 92,
    "subsample": 0.6754111707378173,
    "colsample_bytree": 0.7127674190690785,
    "reg_alpha": 0.6892532018530912,
    "reg_lambda": 0.02273577641331527,
    "random_state": 42, "verbose": -1,
}
X1, y1 = prepare_xy(full_df, make_label, fc)
sc1 = StandardScaler()
sc1.fit_transform(X1)
m1 = lgb.LGBMClassifier(**p1)
m1.fit(sc1.transform(X1), y1)
eval_model(m1, sc1, fc, test_20k, make_label, "20k (set5+4)")
eval_model(m1, sc1, fc, test_10k, make_label, "10k (recent)")

# ---- Strategy 2: Large moves (tuned params) ----
print("\n=== Strategy 2: Large moves (Optuna-tuned, threshold ±0.0373%) ===")
pct_thr = 0.0372654293444779
p2 = {
    "n_estimators": 500, "max_depth": 6,
    "learning_rate": 0.020784549897083126,
    "num_leaves": 37, "min_child_samples": 112,
    "subsample": 0.8103767331776779,
    "colsample_bytree": 0.5486659787425067,
    "reg_alpha": 0.0034916132094567013,
    "reg_lambda": 0.1118425108233888,
    "random_state": 42, "verbose": -1,
}
label_fn2 = lambda df: make_label_large_moves(df, pct_thr)
X2, y2 = prepare_xy(full_df, label_fn2, fc)
sc2 = StandardScaler()
sc2.fit_transform(X2)
m2 = lgb.LGBMClassifier(**p2)
m2.fit(sc2.transform(X2), y2)
print(f"  Training samples: {len(X2):,} (after filtering small moves)")
eval_model(m2, sc2, fc, test_20k, label_fn2, "20k (set5+4)")
eval_model(m2, sc2, fc, test_10k, label_fn2, "10k (recent)")

# ---- Save both models ----
md = os.path.join(_ROOT, "models")
os.makedirs(md, exist_ok=True)

pickle.dump(m1, open(os.path.join(md, "model.pkl"), "wb"))
pickle.dump(sc1, open(os.path.join(md, "scaler.pkl"), "wb"))
pickle.dump(fc, open(os.path.join(md, "features.pkl"), "wb"))
print("\nSaved Strategy 1 (all moves, tuned) -> models/model.pkl")

pickle.dump(m2, open(os.path.join(md, "model_large_moves.pkl"), "wb"))
pickle.dump(sc2, open(os.path.join(md, "scaler_large_moves.pkl"), "wb"))
pickle.dump(pct_thr, open(os.path.join(md, "threshold_large_moves.pkl"), "wb"))
print(f"Saved Strategy 2 (large moves, tuned) -> models/model_large_moves.pkl")
