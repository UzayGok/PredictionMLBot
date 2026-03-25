"""
save_two_stage_production.py
Save the two-stage production system:
  Stage 1: OLD magnitude LGB (45 direction features, old params)
  Stage 2: B2 stacking direction model (LGB + XGB + RF → LogReg meta)
  Thresholds: mag_proba >= 0.5, dir_conf >= 0.55
  OOS result: 57.81% accuracy on 896 trades out of 47,784 (1.9%)
"""
import os, sys, pickle
import numpy as np, pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from Training.features import (
    calculate_features, make_label, make_label_large_moves,
    DIRECTION_FEATURES,
)

PCT_THR = 0.0372654293444779
MAG_THRESHOLD = 0.10

# Original mag model params (DO NOT CHANGE)
MAG_PARAMS = {
    "n_estimators": 400, "max_depth": 3, "learning_rate": 0.0183,
    "num_leaves": 37, "min_child_samples": 46,
    "subsample": 0.80, "colsample_bytree": 0.76,
    "reg_alpha": 0.74, "reg_lambda": 0.07,
    "random_state": 42, "verbose": -1,
}

# Direction model params (tuned)
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


def main():
    print("Loading training data...")
    full = pd.read_csv(os.path.join(_ROOT, "data", "btc_candles_100k.csv"))
    full = calculate_features(full)
    full["next_return_pct"] = (full["close"].shift(-1) - full["close"]) / full["close"] * 100
    full["next_abs_return"] = full["next_return_pct"].abs()
    full["direction_label"] = make_label(full)
    full["lm_label"] = make_label_large_moves(full, PCT_THR)
    full = full.dropna(subset=DIRECTION_FEATURES + ["direction_label", "next_return_pct"]).reset_index(drop=True)
    print(f"Training samples: {len(full):,}")

    feats = DIRECTION_FEATURES  # same 45 features for both models

    # ---- Stage 1: Magnitude model ----
    print("\nTraining magnitude model...")
    mag_scaler = StandardScaler()
    X_mag = mag_scaler.fit_transform(full[feats])
    y_mag = (full["next_abs_return"] > MAG_THRESHOLD).astype(int).values
    mag_model = lgb.LGBMClassifier(**MAG_PARAMS)
    mag_model.fit(X_mag, y_mag)
    print(f"  Big move rate: {y_mag.mean():.2%}")

    # ---- Stage 2: B2 stacking direction model ----
    print("Training B2 stacking direction model...")
    train_lm = full.dropna(subset=["lm_label"]).reset_index(drop=True)
    dir_scaler = StandardScaler()
    X_dir = dir_scaler.fit_transform(train_lm[feats])
    y_dir = train_lm["lm_label"].values

    # 70/30 split for meta-learner
    sp = int(len(X_dir) * 0.7)
    X_base, X_meta = X_dir[:sp], X_dir[sp:]
    y_base, y_meta = y_dir[:sp], y_dir[sp:]

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

    meta_feats = np.column_stack([
        d_lgb.predict_proba(X_meta)[:, 1],
        d_xgb.predict_proba(X_meta)[:, 1],
        d_rf.predict_proba(X_meta)[:, 1],
    ])
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    meta_clf.fit(meta_feats, y_meta)

    # Retrain base models on full data
    d_lgb_full = lgb.LGBMClassifier(**DIR_PARAMS); d_lgb_full.fit(X_dir, y_dir)
    d_xgb_full = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=0.1, random_state=42,
        use_label_encoder=False, eval_metric="logloss", verbosity=0,
    ); d_xgb_full.fit(X_dir, y_dir)
    d_rf_full = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=50,
        max_features="sqrt", random_state=42, n_jobs=-1,
    ); d_rf_full.fit(X_dir, y_dir)

    # ---- OOS Evaluation ----
    print("\n" + "=" * 60)
    print("OOS EVALUATION")
    print("=" * 60)

    test_files = [f"btc_candles_set{i}.csv" for i in range(5, 0, -1)] + ["btc_candles_10k.csv"]
    test_dfs = []
    for f in test_files:
        p = os.path.join(_ROOT, "data", f)
        t = pd.read_csv(p)
        t = calculate_features(t)
        t["direction_label"] = make_label(t)
        t = t.dropna(subset=feats + ["direction_label"]).reset_index(drop=True)
        test_dfs.append(t)
    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"Test samples: {len(test_df):,}")

    # Mag predictions
    Xt_mag = mag_scaler.transform(test_df[feats])
    mag_proba = mag_model.predict_proba(Xt_mag)[:, 1]

    # Dir predictions (B2 stacking)
    Xt_dir = dir_scaler.transform(test_df[feats])
    meta_test = np.column_stack([
        d_lgb_full.predict_proba(Xt_dir)[:, 1],
        d_xgb_full.predict_proba(Xt_dir)[:, 1],
        d_rf_full.predict_proba(Xt_dir)[:, 1],
    ])
    dir_pred = meta_clf.predict(meta_test)
    dir_proba = meta_clf.predict_proba(meta_test)
    dir_conf = np.maximum(dir_proba[:, 0], dir_proba[:, 1])
    y_test = test_df["direction_label"].values

    # Production thresholds
    mask = (mag_proba >= MAG_PROBA_THR) & (dir_conf >= DIR_CONF_THR)
    n = mask.sum()
    acc = accuracy_score(y_test[mask], dir_pred[mask])
    print(f"\n  PRODUCTION (mag>={MAG_PROBA_THR}, dir>={DIR_CONF_THR}):")
    print(f"    {acc:.4f} ({acc*100:.2f}%) on {n:,} trades ({n/len(y_test)*100:.1f}%)")

    # Grid for reference
    print(f"\n  Grid:")
    for mt in [0.4, 0.5, 0.6]:
        for dc in [0.55, 0.57]:
            m = (mag_proba >= mt) & (dir_conf >= dc)
            nn = m.sum()
            if nn > 0:
                a = accuracy_score(y_test[m], dir_pred[m])
                print(f"    mag>={mt}, dir>={dc}: {a:.4f} ({a*100:.2f}%) on {nn:,}")

    # Direction-only baseline
    print(f"\n  Direction-only baselines:")
    for dc in [0.55, 0.57]:
        m = dir_conf >= dc
        nn = m.sum()
        if nn > 0:
            a = accuracy_score(y_test[m], dir_pred[m])
            print(f"    dir>={dc}: {a:.4f} ({a*100:.2f}%) on {nn:,}")

    # ---- Save ----
    print("\n" + "=" * 60)
    print("SAVING PRODUCTION MODELS")
    print("=" * 60)

    md = os.path.join(_ROOT, "models")
    os.makedirs(md, exist_ok=True)

    pickle.dump(mag_model, open(os.path.join(md, "mag_model.pkl"), "wb"))
    pickle.dump(mag_scaler, open(os.path.join(md, "mag_scaler.pkl"), "wb"))
    pickle.dump(d_lgb_full, open(os.path.join(md, "dir_lgb.pkl"), "wb"))
    pickle.dump(d_xgb_full, open(os.path.join(md, "dir_xgb.pkl"), "wb"))
    pickle.dump(d_rf_full, open(os.path.join(md, "dir_rf.pkl"), "wb"))
    pickle.dump(meta_clf, open(os.path.join(md, "dir_meta.pkl"), "wb"))
    pickle.dump(dir_scaler, open(os.path.join(md, "dir_scaler.pkl"), "wb"))
    pickle.dump(feats, open(os.path.join(md, "features.pkl"), "wb"))

    config = {
        "mag_params": MAG_PARAMS,
        "dir_params": DIR_PARAMS,
        "features": feats,
        "mag_threshold": MAG_THRESHOLD,
        "mag_proba_thr": MAG_PROBA_THR,
        "dir_conf_thr": DIR_CONF_THR,
        "pct_thr": PCT_THR,
    }
    pickle.dump(config, open(os.path.join(md, "two_stage_config.pkl"), "wb"))

    print(f"  Saved to {md}/:")
    print(f"    mag_model.pkl, mag_scaler.pkl")
    print(f"    dir_lgb.pkl, dir_xgb.pkl, dir_rf.pkl, dir_meta.pkl, dir_scaler.pkl")
    print(f"    features.pkl ({len(feats)} features)")
    print(f"    two_stage_config.pkl")
    print(f"\n  Thresholds: mag>={MAG_PROBA_THR}, dir>={DIR_CONF_THR}")
    print(f"  OOS: {acc*100:.2f}% on {n:,} trades")


if __name__ == "__main__":
    main()
