"""Evaluate saved two-stage production models on btc_candles_10k.csv only."""
import os, sys, pickle
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from Training.features import calculate_features, make_label

MD = os.path.join(_ROOT, "models")

# Load models
mag_model   = pickle.load(open(os.path.join(MD, "mag_model.pkl"), "rb"))
mag_scaler  = pickle.load(open(os.path.join(MD, "mag_scaler.pkl"), "rb"))
dir_lgb     = pickle.load(open(os.path.join(MD, "dir_lgb.pkl"), "rb"))
dir_xgb     = pickle.load(open(os.path.join(MD, "dir_xgb.pkl"), "rb"))
dir_rf      = pickle.load(open(os.path.join(MD, "dir_rf.pkl"), "rb"))
dir_meta    = pickle.load(open(os.path.join(MD, "dir_meta.pkl"), "rb"))
dir_scaler  = pickle.load(open(os.path.join(MD, "dir_scaler.pkl"), "rb"))
feats       = pickle.load(open(os.path.join(MD, "features.pkl"), "rb"))
config      = pickle.load(open(os.path.join(MD, "two_stage_config.pkl"), "rb"))

MAG_PROBA_THR = config["mag_proba_thr"]
DIR_CONF_THR  = config["dir_conf_thr"]

# Load test data
print("Loading btc_candles_10k.csv ...")
df = pd.read_csv(os.path.join(_ROOT, "data", "btc_candles_10k.csv"))
df = calculate_features(df)
df["direction_label"] = make_label(df)
df = df.dropna(subset=feats + ["direction_label"]).reset_index(drop=True)
print(f"Test samples: {len(df):,}\n")

# Stage 1: Magnitude
Xm = mag_scaler.transform(df[feats])
mag_proba = mag_model.predict_proba(Xm)[:, 1]

# Stage 2: Direction (B2 stacking)
Xd = dir_scaler.transform(df[feats])
meta_feats = np.column_stack([
    dir_lgb.predict_proba(Xd)[:, 1],
    dir_xgb.predict_proba(Xd)[:, 1],
    dir_rf.predict_proba(Xd)[:, 1],
])
dir_pred  = dir_meta.predict(meta_feats)
dir_proba = dir_meta.predict_proba(meta_feats)
dir_conf  = np.maximum(dir_proba[:, 0], dir_proba[:, 1])
y_true    = df["direction_label"].values

# Results
print("=" * 60)
print("TWO-STAGE RESULTS  —  btc_candles_10k.csv ONLY")
print("=" * 60)

print(f"\nProduction thresholds: mag >= {MAG_PROBA_THR}, dir >= {DIR_CONF_THR}")
mask = (mag_proba >= MAG_PROBA_THR) & (dir_conf >= DIR_CONF_THR)
n = mask.sum()
if n:
    acc = accuracy_score(y_true[mask], dir_pred[mask])
    up  = (dir_pred[mask] == 1).sum()
    dn  = (dir_pred[mask] == 0).sum()
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Trades:   {n:,} / {len(y_true):,} ({n/len(y_true)*100:.1f}%)")
    print(f"  UP calls: {up}  |  DOWN calls: {dn}")
else:
    print("  No trades at production thresholds.")

print(f"\nFull grid:")
for mt in [0.4, 0.45, 0.5, 0.55, 0.6]:
    for dc in [0.53, 0.55, 0.57, 0.59]:
        m = (mag_proba >= mt) & (dir_conf >= dc)
        nn = m.sum()
        if nn > 0:
            a = accuracy_score(y_true[m], dir_pred[m])
            tag = " <<<" if mt == MAG_PROBA_THR and dc == DIR_CONF_THR else ""
            print(f"  mag>={mt:.2f}, dir>={dc:.2f}: {a*100:.2f}% on {nn:>4,} trades{tag}")

# Direction-only baselines
print(f"\nDirection-only baselines (no mag filter):")
for dc in [0.53, 0.55, 0.57, 0.59]:
    m = dir_conf >= dc
    nn = m.sum()
    if nn > 0:
        a = accuracy_score(y_true[m], dir_pred[m])
        print(f"  dir>={dc:.2f}: {a*100:.2f}% on {nn:,}")

# Overall baseline
base = max(y_true.mean(), 1 - y_true.mean())
print(f"\nBaseline (always majority): {base*100:.2f}%")
