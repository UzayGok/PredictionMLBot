"""
tune_magnitude.py
Optuna tuning for the magnitude model (Stage 1: "is a big move coming?")
Also jointly optimizes the two-stage thresholds (mag_proba, dir_conf).

Trains on 100k candles (80/20 time split for tuning).
"""
import os, sys, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from Training.features import calculate_features, make_label, make_label_large_moves

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

# Direction model tuned params (fixed)
PCT_THR = 0.0372654293444779
LGB_DIR_PARAMS = {
    "n_estimators": 500, "max_depth": 6,
    "learning_rate": 0.020784549897083126,
    "num_leaves": 37, "min_child_samples": 112,
    "subsample": 0.8103767331776779,
    "colsample_bytree": 0.5486659787425067,
    "reg_alpha": 0.0034916132094567013,
    "reg_lambda": 0.1118425108233888,
    "random_state": 42, "verbose": -1,
}
MAG_THRESHOLD = 0.10  # Fixed: predict |return| > 0.10%


def load_full(path):
    df = pd.read_csv(path)
    df = calculate_features(df)
    df["next_return_pct"] = (df["close"].shift(-1) - df["close"]) / df["close"] * 100
    df["next_abs_return"] = df["next_return_pct"].abs()
    df["direction_label"] = make_label(df)
    df["lm_label"] = make_label_large_moves(df, PCT_THR)
    df = df.dropna(subset=TOP_45 + ["direction_label", "next_return_pct"]).reset_index(drop=True)
    return df


def main():
    print("Loading data...")
    full_df = load_full(os.path.join(_ROOT, "data", "btc_candles_100k.csv"))

    # 80/20 time split
    split = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split].copy().reset_index(drop=True)
    val_df = full_df.iloc[split:].copy().reset_index(drop=True)
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}")

    # Pre-build direction model (B2 stacking) on train split
    # so we can evaluate the combined two-stage on val
    print("Building direction model (B2 stacking) on train split...")
    train_lm = train_df.dropna(subset=["lm_label"]).reset_index(drop=True)
    scaler_dir = StandardScaler()
    Xtr_dir = scaler_dir.fit_transform(train_lm[TOP_45])

    # Stacking: 70/30 internal split for meta-model
    sp2 = int(len(Xtr_dir) * 0.7)
    Xd_base, Xd_meta = Xtr_dir[:sp2], Xtr_dir[sp2:]
    yd_base = train_lm["lm_label"].iloc[:sp2]
    yd_meta = train_lm["lm_label"].iloc[sp2:]

    d_lgb = lgb.LGBMClassifier(**LGB_DIR_PARAMS); d_lgb.fit(Xd_base, yd_base)
    d_xgb = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=0.1, random_state=42,
        use_label_encoder=False, eval_metric="logloss", verbosity=0,
    ); d_xgb.fit(Xd_base, yd_base)
    d_rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=50,
        max_features="sqrt", random_state=42, n_jobs=-1,
    ); d_rf.fit(Xd_base, yd_base)

    meta_tr = np.column_stack([
        d_lgb.predict_proba(Xd_meta)[:, 1],
        d_xgb.predict_proba(Xd_meta)[:, 1],
        d_rf.predict_proba(Xd_meta)[:, 1],
    ])
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    meta_clf.fit(meta_tr, yd_meta)

    # Full-train direction models
    d_lgb_f = lgb.LGBMClassifier(**LGB_DIR_PARAMS); d_lgb_f.fit(Xtr_dir, train_lm["lm_label"])
    d_xgb_f = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=0.1, random_state=42,
        use_label_encoder=False, eval_metric="logloss", verbosity=0,
    ); d_xgb_f.fit(Xtr_dir, train_lm["lm_label"])
    d_rf_f = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=50,
        max_features="sqrt", random_state=42, n_jobs=-1,
    ); d_rf_f.fit(Xtr_dir, train_lm["lm_label"])

    # Val set direction predictions (B2)
    Xval_dir = scaler_dir.transform(val_df[TOP_45])
    val_meta = np.column_stack([
        d_lgb_f.predict_proba(Xval_dir)[:, 1],
        d_xgb_f.predict_proba(Xval_dir)[:, 1],
        d_rf_f.predict_proba(Xval_dir)[:, 1],
    ])
    dir_pred_val = meta_clf.predict(val_meta)
    dir_proba_val = meta_clf.predict_proba(val_meta)
    dir_conf_val = np.maximum(dir_proba_val[:, 0], dir_proba_val[:, 1])
    y_val_dir = val_df["direction_label"].values

    print("Direction model ready.\n")

    # Pre-scale magnitude model data
    scaler_mag = StandardScaler()
    Xtr_mag = scaler_mag.fit_transform(train_df[TOP_45])
    Xval_mag = scaler_mag.transform(val_df[TOP_45])
    y_train_mag = (train_df["next_abs_return"] > MAG_THRESHOLD).astype(int).values

    # ---- Optuna: tune magnitude model hyperparameters ----
    # Objective: maximize two-stage accuracy on val (mag_proba>=0.4, dir_conf>=0.55)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 50),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42, "verbose": -1,
        }
        mag_proba_thr = trial.suggest_float("mag_proba_thr", 0.25, 0.60)
        dir_conf_thr = trial.suggest_float("dir_conf_thr", 0.50, 0.60)

        mdl = lgb.LGBMClassifier(**params)
        mdl.fit(Xtr_mag, y_train_mag)
        mag_proba = mdl.predict_proba(Xval_mag)[:, 1]

        # Two-stage filter
        mask = (mag_proba >= mag_proba_thr) & (dir_conf_val >= dir_conf_thr)
        n = mask.sum()
        if n < 100:
            return 0.0  # too few trades

        acc = accuracy_score(y_val_dir[mask], dir_pred_val[mask])

        # Bonus for having more trades (avoid over-filtering)
        trade_frac = n / len(y_val_dir)
        # Score = accuracy, but penalize if < 2% of candles traded
        score = acc if trade_frac >= 0.02 else acc * (trade_frac / 0.02)

        return score

    print("=" * 60)
    print("TUNING MAGNITUDE MODEL (50 trials)")
    print(f"Fixed: magnitude threshold > {MAG_THRESHOLD}%")
    print("Optimizing: LGB params + mag_proba_thr + dir_conf_thr")
    print("=" * 60)

    study = optuna.create_study(direction="maximize", study_name="magnitude_model")
    study.optimize(objective, n_trials=50, show_progress_bar=True, catch=(KeyboardInterrupt,))

    best = study.best_params
    print(f"\nBest score: {study.best_value:.4f}")
    print(f"Best params: {best}")

    mag_proba_thr = best.pop("mag_proba_thr")
    dir_conf_thr = best.pop("dir_conf_thr")
    best["random_state"] = 42
    best["verbose"] = -1

    print(f"\nBest thresholds: mag_proba >= {mag_proba_thr:.4f}, dir_conf >= {dir_conf_thr:.4f}")
    print(f"Best LGB params: {best}")

    # ---- Evaluate on val ----
    print("\n--- Validation results ---")
    mdl_best = lgb.LGBMClassifier(**best)
    mdl_best.fit(Xtr_mag, y_train_mag)
    mag_proba_best = mdl_best.predict_proba(Xval_mag)[:, 1]

    mask = (mag_proba_best >= mag_proba_thr) & (dir_conf_val >= dir_conf_thr)
    n = mask.sum()
    acc = accuracy_score(y_val_dir[mask], dir_pred_val[mask])
    print(f"  Two-stage: {acc:.4f} ({acc*100:.2f}%) on {n:,} trades ({n/len(y_val_dir)*100:.1f}%)")

    # Baseline: direction alone at same conf
    mask_base = dir_conf_val >= dir_conf_thr
    n_base = mask_base.sum()
    acc_base = accuracy_score(y_val_dir[mask_base], dir_pred_val[mask_base])
    print(f"  Baseline dir>={dir_conf_thr:.2f}: {acc_base:.4f} ({acc_base*100:.2f}%) on {n_base:,}")

    # ---- Now evaluate on ALL test files ----
    print("\n" + "=" * 60)
    print("FINAL TEST on all 6 OOS files (set5..set1 + 10k)")
    print("=" * 60)

    test_df = pd.concat([
        load_full(os.path.join(_ROOT, "data", f"btc_candles_set{i}.csv"))
        for i in range(5, 0, -1)
    ] + [
        load_full(os.path.join(_ROOT, "data", "btc_candles_10k.csv")),
    ], ignore_index=True)
    print(f"Test samples: {len(test_df):,}")

    # Retrain magnitude model on FULL 100k
    scaler_mag_full = StandardScaler()
    Xtr_mag_full = scaler_mag_full.fit_transform(full_df[TOP_45])
    y_mag_full = (full_df["next_abs_return"] > MAG_THRESHOLD).astype(int).values
    mag_model_final = lgb.LGBMClassifier(**best)
    mag_model_final.fit(Xtr_mag_full, y_mag_full)

    # Retrain direction models on FULL 100k (large-moves filtered)
    full_lm = full_df.dropna(subset=["lm_label"]).reset_index(drop=True)
    scaler_dir_full = StandardScaler()
    Xtr_dir_full = scaler_dir_full.fit_transform(full_lm[TOP_45])

    sp3 = int(len(Xtr_dir_full) * 0.7)
    Xdf_base, Xdf_meta = Xtr_dir_full[:sp3], Xtr_dir_full[sp3:]
    ydf_base = full_lm["lm_label"].iloc[:sp3]
    ydf_meta = full_lm["lm_label"].iloc[sp3:]

    df_lgb = lgb.LGBMClassifier(**LGB_DIR_PARAMS); df_lgb.fit(Xdf_base, ydf_base)
    df_xgb = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=0.1, random_state=42,
        use_label_encoder=False, eval_metric="logloss", verbosity=0,
    ); df_xgb.fit(Xdf_base, ydf_base)
    df_rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=50,
        max_features="sqrt", random_state=42, n_jobs=-1,
    ); df_rf.fit(Xdf_base, ydf_base)

    meta_tr_f = np.column_stack([
        df_lgb.predict_proba(Xdf_meta)[:, 1],
        df_xgb.predict_proba(Xdf_meta)[:, 1],
        df_rf.predict_proba(Xdf_meta)[:, 1],
    ])
    meta_clf_f = LogisticRegression(max_iter=1000, random_state=42)
    meta_clf_f.fit(meta_tr_f, ydf_meta)

    # Full-data direction models
    df_lgb_full = lgb.LGBMClassifier(**LGB_DIR_PARAMS); df_lgb_full.fit(Xtr_dir_full, full_lm["lm_label"])
    df_xgb_full = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.6, reg_alpha=0.01, reg_lambda=0.1, random_state=42,
        use_label_encoder=False, eval_metric="logloss", verbosity=0,
    ); df_xgb_full.fit(Xtr_dir_full, full_lm["lm_label"])
    df_rf_full = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=50,
        max_features="sqrt", random_state=42, n_jobs=-1,
    ); df_rf_full.fit(Xtr_dir_full, full_lm["lm_label"])

    # Test predictions
    Xts_mag = scaler_mag_full.transform(test_df[TOP_45])
    Xts_dir = scaler_dir_full.transform(test_df[TOP_45])

    mag_proba_test = mag_model_final.predict_proba(Xts_mag)[:, 1]

    test_meta = np.column_stack([
        df_lgb_full.predict_proba(Xts_dir)[:, 1],
        df_xgb_full.predict_proba(Xts_dir)[:, 1],
        df_rf_full.predict_proba(Xts_dir)[:, 1],
    ])
    dir_pred_test = meta_clf_f.predict(test_meta)
    dir_proba_test = meta_clf_f.predict_proba(test_meta)
    dir_conf_test = np.maximum(dir_proba_test[:, 0], dir_proba_test[:, 1])
    y_test_dir = test_df["direction_label"].values

    # Two-stage with tuned thresholds
    mask_ts = (mag_proba_test >= mag_proba_thr) & (dir_conf_test >= dir_conf_thr)
    n_ts = mask_ts.sum()
    acc_ts = accuracy_score(y_test_dir[mask_ts], dir_pred_test[mask_ts]) if n_ts > 0 else 0
    print(f"\n  TUNED two-stage (mag>={mag_proba_thr:.3f}, dir>={dir_conf_thr:.3f}):")
    print(f"    {acc_ts:.4f} ({acc_ts*100:.2f}%) on {n_ts:,} trades ({n_ts/len(y_test_dir)*100:.1f}%)")

    # Compare to pre-tuning baselines
    for mp, dc in [(0.4, 0.55), (0.5, 0.55), (0.3, 0.55)]:
        mask_c = (mag_proba_test >= mp) & (dir_conf_test >= dc)
        n_c = mask_c.sum()
        if n_c > 0:
            acc_c = accuracy_score(y_test_dir[mask_c], dir_pred_test[mask_c])
            print(f"  Pre-tuned (mag>={mp}, dir>={dc}): {acc_c:.4f} ({acc_c*100:.2f}%) on {n_c:,}")

    # Direction-only baseline
    mask_dir = dir_conf_test >= dir_conf_thr
    n_dir = mask_dir.sum()
    acc_dir = accuracy_score(y_test_dir[mask_dir], dir_pred_test[mask_dir])
    print(f"  Direction-only (dir>={dir_conf_thr:.3f}): {acc_dir:.4f} ({acc_dir*100:.2f}%) on {n_dir:,}")

    mask_dir55 = dir_conf_test >= 0.55
    n_dir55 = mask_dir55.sum()
    acc_dir55 = accuracy_score(y_test_dir[mask_dir55], dir_pred_test[mask_dir55])
    print(f"  Direction-only (dir>=0.55): {acc_dir55:.4f} ({acc_dir55*100:.2f}%) on {n_dir55:,}")

    # Save best params for later use
    import pickle
    md = os.path.join(_ROOT, "models")
    os.makedirs(md, exist_ok=True)
    tuned_config = {
        "mag_lgb_params": best,
        "mag_threshold": MAG_THRESHOLD,
        "mag_proba_thr": mag_proba_thr,
        "dir_conf_thr": dir_conf_thr,
    }
    pickle.dump(tuned_config, open(os.path.join(md, "two_stage_config.pkl"), "wb"))
    print(f"\nSaved tuned config to models/two_stage_config.pkl")


if __name__ == "__main__":
    main()
