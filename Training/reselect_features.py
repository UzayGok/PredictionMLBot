"""
reselect_features.py
Feature re-selection for the stacking pipeline.
Tests whether a different feature subset works better for the 3-model stacking
ensemble than the current 45 DIRECTION_FEATURES (which were selected for LGB alone).

Approach:
  1) Train the full stacking pipeline with ALL 75 features
  2) Rank features by combined importance across LGB+XGB+RF
  3) Test subsets: top-30, top-40, top-45, top-50, top-55, all-75
  4) Compare to baseline (current 45)
"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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


def load_data(path, feats):
    df = pd.read_csv(path)
    df = calculate_features(df)
    df["direction_label"] = make_label(df)
    df["lm_label"] = make_label_large_moves(df, PCT_THR)
    df["next_return_pct"] = (df["close"].shift(-1) - df["close"]) / df["close"] * 100
    df["next_abs_return"] = df["next_return_pct"].abs()
    df = df.dropna(subset=feats + ["direction_label"]).reset_index(drop=True)
    return df


def run_stacking(X_base, y_base, X_meta, y_meta, X_test):
    """Train 3-model stacking, return (predictions, confidence) on X_test."""
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

    # Meta-learner
    meta_hold = np.column_stack([
        d_lgb.predict_proba(X_meta)[:, 1],
        d_xgb.predict_proba(X_meta)[:, 1],
        d_rf.predict_proba(X_meta)[:, 1],
    ])
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    meta_clf.fit(meta_hold, y_meta)

    # Retrain on all (base+meta)
    X_all = np.vstack([X_base, X_meta])
    y_all = np.concatenate([y_base, y_meta])
    d_lgb2 = lgb.LGBMClassifier(**DIR_PARAMS); d_lgb2.fit(X_all, y_all)
    d_xgb2 = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=0.1, random_state=42,
        use_label_encoder=False, eval_metric="logloss", verbosity=0,
    ); d_xgb2.fit(X_all, y_all)
    d_rf2 = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=50,
        max_features="sqrt", random_state=42, n_jobs=-1,
    ); d_rf2.fit(X_all, y_all)

    meta_test = np.column_stack([
        d_lgb2.predict_proba(X_test)[:, 1],
        d_xgb2.predict_proba(X_test)[:, 1],
        d_rf2.predict_proba(X_test)[:, 1],
    ])
    pred = meta_clf.predict(meta_test)
    proba = meta_clf.predict_proba(meta_test)
    conf = np.maximum(proba[:, 0], proba[:, 1])

    # Feature importances (combined)
    imp = {}
    for name, m in [("lgb", d_lgb2), ("xgb", d_xgb2)]:
        fi = m.feature_importances_
        fi = fi / fi.sum()  # Normalize
        for i, v in enumerate(fi):
            imp[i] = imp.get(i, 0) + v
    # RF importances
    fi_rf = d_rf2.feature_importances_
    fi_rf = fi_rf / fi_rf.sum()
    for i, v in enumerate(fi_rf):
        imp[i] = imp.get(i, 0) + v

    return pred, conf, imp


def main():
    print("=" * 60)
    print("FEATURE RE-SELECTION FOR STACKING")
    print("=" * 60)

    # Use all features for initial ranking
    all_feats = FEATURE_COLS

    print("\nLoading data with all features...")
    train_df = load_data(os.path.join(_ROOT, "data", "btc_candles_100k.csv"), all_feats)
    test_df = load_data(os.path.join(_ROOT, "data", "btc_candles_10k.csv"), all_feats)
    print(f"  Train: {len(train_df):,}  Test: {len(test_df):,}")

    # Mag model (same as production, uses DIRECTION_FEATURES)
    print("\nTraining mag model (DIRECTION_FEATURES as baseline)...")
    mag_scaler = StandardScaler()
    X_mag_train = mag_scaler.fit_transform(train_df[DIRECTION_FEATURES])
    y_mag_train = (train_df["next_abs_return"] > MAG_THRESHOLD).astype(int).values
    mag_model = lgb.LGBMClassifier(**MAG_PARAMS)
    mag_model.fit(X_mag_train, y_mag_train)

    X_mag_test = mag_scaler.transform(test_df[DIRECTION_FEATURES])
    mag_proba = mag_model.predict_proba(X_mag_test)[:, 1]
    y_true = test_df["direction_label"].values

    # ---- Step 1: Feature ranking using ALL features ----
    print("\n--- Step 1: Feature importance ranking (all 75 features) ---")
    train_lm = train_df.dropna(subset=["lm_label"]).reset_index(drop=True)
    scaler_all = StandardScaler()
    X_all = scaler_all.fit_transform(train_lm[all_feats])
    y_all = train_lm["lm_label"].values

    sp = int(len(X_all) * 0.7)
    X_base, X_meta = X_all[:sp], X_all[sp:]
    y_base, y_meta = y_all[:sp], y_all[sp:]
    X_test_all = scaler_all.transform(test_df[all_feats])

    _, _, imp_all = run_stacking(X_base, y_base, X_meta, y_meta, X_test_all)

    # Rank features by combined importance
    ranked = sorted(imp_all.items(), key=lambda x: x[1], reverse=True)
    feat_names_ranked = [all_feats[i] for i, _ in ranked]
    feat_imps_ranked = [v for _, v in ranked]

    print(f"\nTop 30 features (combined LGB+XGB+RF importance):")
    for i, (name, imp) in enumerate(zip(feat_names_ranked[:30], feat_imps_ranked[:30])):
        in_45 = "  *" if name in DIRECTION_FEATURES else ""
        print(f"  {i+1:2}. {name:<30} {imp:.4f}{in_45}")

    # Show features NOT in current 45 that rank high
    print(f"\nFeatures NOT in current 45 that rank in top 50:")
    for i, name in enumerate(feat_names_ranked[:50]):
        if name not in DIRECTION_FEATURES:
            print(f"  rank {i+1}: {name} (imp={feat_imps_ranked[i]:.4f})")

    # ---- Step 2: Test different subsets ----
    print("\n\n--- Step 2: Testing different feature subsets ---")

    subsets = {
        "Current 45 (baseline)": DIRECTION_FEATURES,
        "Top 30 (stacking-ranked)": feat_names_ranked[:30],
        "Top 40 (stacking-ranked)": feat_names_ranked[:40],
        "Top 45 (stacking-ranked)": feat_names_ranked[:45],
        "Top 50 (stacking-ranked)": feat_names_ranked[:50],
        "Top 55 (stacking-ranked)": feat_names_ranked[:55],
        "All 75": all_feats,
    }

    results = []
    for label, feats in subsets.items():
        print(f"\n  Testing: {label} ({len(feats)} features)...")

        # Need to re-prepare direction data with this subset
        dir_scaler = StandardScaler()
        X_dir = dir_scaler.fit_transform(train_lm[feats])
        y_dir = train_lm["lm_label"].values
        sp = int(len(X_dir) * 0.7)
        Xb, Xm = X_dir[:sp], X_dir[sp:]
        yb, ym = y_dir[:sp], y_dir[sp:]
        Xt = dir_scaler.transform(test_df[feats])

        pred, conf, _ = run_stacking(Xb, yb, Xm, ym, Xt)

        # Eval at production threshold
        for mt in [0.5]:
            for dc in [0.53, 0.55, 0.57]:
                m = (mag_proba >= mt) & (conf >= dc)
                nn = m.sum()
                if nn > 0:
                    a = accuracy_score(y_true[m], pred[m])
                    tag = " <<<" if dc == 0.55 else ""
                    print(f"    mag>=0.50, dir>={dc}: {a*100:.2f}% on {nn:>4}{tag}")
                    if dc == 0.55:
                        results.append((label, a, nn))
        # dir-only
        m = conf >= 0.55
        nn = m.sum()
        if nn > 0:
            a = accuracy_score(y_true[m], pred[m])
            print(f"    dir-only>=0.55: {a*100:.2f}% on {nn:>4}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY  --  mag>=0.50, dir>=0.55")
    print("=" * 60)
    print(f"\n{'Feature Set':<35} {'Accuracy':>8} {'Trades':>7}")
    print("-" * 55)
    for label, acc, n in results:
        print(f"{label:<35} {acc*100:>7.2f}% {n:>6}")

    # Show the new top-N features for the winning config
    print(f"\n\nStacking-ranked top 50 features:")
    for i, name in enumerate(feat_names_ranked[:50]):
        print(f"  {i+1:2}. {name}")


if __name__ == "__main__":
    main()
