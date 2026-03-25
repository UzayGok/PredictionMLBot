"""
improve_magnitude.py
Improve the magnitude model with:
  1. New magnitude-specific features (12 new features)
  2. Feature importance ranking + ablation
  3. Class weight balancing
  4. Optuna re-tune with expanded feature set
  5. Final two-stage evaluation on 48k test

Uses sniper profile: mag>=0.5, dir>=0.55 as target.
"""
import os, sys, warnings, pickle
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
from Training.features import (
    calculate_features, make_label, make_label_large_moves,
    FEATURE_COLS, DIRECTION_FEATURES,
)

PCT_THR = 0.0372654293444779
MAG_THRESHOLD = 0.10  # predict |return| > 0.10%

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


def load_full(path):
    df = pd.read_csv(path)
    df = calculate_features(df)
    df["next_return_pct"] = (df["close"].shift(-1) - df["close"]) / df["close"] * 100
    df["next_abs_return"] = df["next_return_pct"].abs()
    df["direction_label"] = make_label(df)
    df["lm_label"] = make_label_large_moves(df, PCT_THR)
    df = df.dropna(subset=FEATURE_COLS + ["direction_label", "next_return_pct"]).reset_index(drop=True)
    return df


def build_direction_model(train_df, scaler_dir):
    """Build B2 stacking direction model."""
    train_lm = train_df.dropna(subset=["lm_label"]).reset_index(drop=True)
    Xtr_dir = scaler_dir.fit_transform(train_lm[DIRECTION_FEATURES])

    sp = int(len(Xtr_dir) * 0.7)
    Xd_base, Xd_meta = Xtr_dir[:sp], Xtr_dir[sp:]
    yd_base = train_lm["lm_label"].iloc[:sp]
    yd_meta = train_lm["lm_label"].iloc[sp:]

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

    # Full-train models
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

    return d_lgb_f, d_xgb_f, d_rf_f, meta_clf


def get_dir_predictions(df, scaler_dir, d_lgb, d_xgb, d_rf, meta_clf):
    """Get direction predictions on a dataframe."""
    Xd = scaler_dir.transform(df[DIRECTION_FEATURES])
    meta = np.column_stack([
        d_lgb.predict_proba(Xd)[:, 1],
        d_xgb.predict_proba(Xd)[:, 1],
        d_rf.predict_proba(Xd)[:, 1],
    ])
    pred = meta_clf.predict(meta)
    proba = meta_clf.predict_proba(meta)
    conf = np.maximum(proba[:, 0], proba[:, 1])
    return pred, conf


def eval_two_stage(mag_proba, mag_thr, dir_pred, dir_conf, dir_thr, y_true, label=""):
    mask = (mag_proba >= mag_thr) & (dir_conf >= dir_thr)
    n = mask.sum()
    if n > 0:
        acc = accuracy_score(y_true[mask], dir_pred[mask])
        print(f"  {label}: {acc:.4f} ({acc*100:.2f}%) on {n:,} ({n/len(y_true)*100:.1f}%)")
        return acc, n
    print(f"  {label}: no trades")
    return 0, 0


def main():
    print("Loading data...")
    full_df = load_full(os.path.join(_ROOT, "data", "btc_candles_100k.csv"))

    test_df = pd.concat([
        load_full(os.path.join(_ROOT, "data", f"btc_candles_set{i}.csv"))
        for i in range(5, 0, -1)
    ] + [
        load_full(os.path.join(_ROOT, "data", "btc_candles_10k.csv")),
    ], ignore_index=True)

    # 80/20 split for tuning
    split = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split].copy().reset_index(drop=True)
    val_df = full_df.iloc[split:].copy().reset_index(drop=True)

    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    print(f"All features: {len(FEATURE_COLS)}")

    # Build direction model
    print("Building direction model (B2)...")
    scaler_dir = StandardScaler()
    d_lgb, d_xgb, d_rf, meta_clf = build_direction_model(train_df, scaler_dir)
    dir_pred_val, dir_conf_val = get_dir_predictions(val_df, scaler_dir, d_lgb, d_xgb, d_rf, meta_clf)
    y_val_dir = val_df["direction_label"].values

    # ================================================================
    # STEP 1: Feature importance for magnitude prediction
    # ================================================================
    print("\n" + "=" * 65)
    print("STEP 1: Feature importance ranking for magnitude model")
    print(f"Using all {len(FEATURE_COLS)} features")
    print("=" * 65)

    all_feats = FEATURE_COLS
    scaler_mag = StandardScaler()
    Xtr_mag = scaler_mag.fit_transform(train_df[all_feats])
    Xval_mag = scaler_mag.transform(val_df[all_feats])
    y_train_mag = (train_df["next_abs_return"] > MAG_THRESHOLD).astype(int).values

    # Train with old (untuned) params first for ranking
    mag_rank_mdl = lgb.LGBMClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.018,
        num_leaves=37, min_child_samples=46,
        subsample=0.8, colsample_bytree=0.76,
        reg_alpha=0.74, reg_lambda=0.07,
        random_state=42, verbose=-1,
    )
    mag_rank_mdl.fit(Xtr_mag, y_train_mag)

    importance = mag_rank_mdl.feature_importances_
    ranking = sorted(zip(all_feats, importance), key=lambda x: -x[1])

    print("\n  Feature ranking for magnitude prediction:")
    for i, (feat, imp) in enumerate(ranking):
        tag = " ** NEW" if feat in [
            "candle_range_pct", "vol_expansion", "big_move_count_12",
            "big_move_count_48", "candles_since_big_move", "volume_acceleration",
            "range_expansion", "hour_of_day", "is_us_session", "is_asia_session",
            "bb_width", "volume_spike_ratio",
        ] else ""
        print(f"    {i+1:>2}. {feat:25s} {imp:5.0f}{tag}")

    ranked_feats = [f for f, _ in ranking]

    # ================================================================
    # STEP 2: Ablation — test two-stage at different feature counts
    # ================================================================
    print("\n" + "=" * 65)
    print("STEP 2: Feature ablation for magnitude model")
    print("=" * 65)

    # Use pre-computed direction predictions on val
    for n_f in [75, 65, 55, 50, 45, 40, 35, 30, 25, 20, 15]:
        if n_f > len(ranked_feats):
            continue
        subset = ranked_feats[:n_f]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(train_df[subset])
        Xval_s = sc.transform(val_df[subset])

        mdl = lgb.LGBMClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.018,
            num_leaves=37, min_child_samples=46,
            subsample=0.8, colsample_bytree=0.76,
            reg_alpha=0.74, reg_lambda=0.07,
            random_state=42, verbose=-1,
        )
        mdl.fit(Xtr_s, y_train_mag)
        mag_proba = mdl.predict_proba(Xval_s)[:, 1]
        eval_two_stage(mag_proba, 0.5, dir_pred_val, dir_conf_val, 0.55, y_val_dir, f"{n_f:>2} feats, sniper")
        eval_two_stage(mag_proba, 0.4, dir_pred_val, dir_conf_val, 0.55, y_val_dir, f"{n_f:>2} feats, balanced")

    # ================================================================
    # STEP 3: Class weights
    # ================================================================
    print("\n" + "=" * 65)
    print("STEP 3: Effect of class weights")
    print("=" * 65)

    best_n = 45  # will try with top-N from ranking after seeing results
    subset_cw = ranked_feats[:best_n]
    sc_cw = StandardScaler()
    Xtr_cw = sc_cw.fit_transform(train_df[subset_cw])
    Xval_cw = sc_cw.transform(val_df[subset_cw])

    for scale_pos in [1.0, 1.5, 2.0, 2.5, 3.0]:
        mdl = lgb.LGBMClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.018,
            num_leaves=37, min_child_samples=46,
            subsample=0.8, colsample_bytree=0.76,
            reg_alpha=0.74, reg_lambda=0.07,
            scale_pos_weight=scale_pos,
            random_state=42, verbose=-1,
        )
        mdl.fit(Xtr_cw, y_train_mag)
        mag_proba = mdl.predict_proba(Xval_cw)[:, 1]
        eval_two_stage(mag_proba, 0.5, dir_pred_val, dir_conf_val, 0.55, y_val_dir, f"weight={scale_pos:.1f} sniper")
        eval_two_stage(mag_proba, 0.4, dir_pred_val, dir_conf_val, 0.55, y_val_dir, f"weight={scale_pos:.1f} balanced")

    # ================================================================
    # STEP 4: Optuna re-tune with expanded features + class weight
    # ================================================================
    print("\n" + "=" * 65)
    print("STEP 4: Optuna re-tune magnitude model (75 trials)")
    print("=" * 65)

    def objective(trial):
        n_feats = trial.suggest_int("n_feats", 20, len(ranked_feats), step=5)
        subset = ranked_feats[:n_feats]

        sc = StandardScaler()
        Xtr_o = sc.fit_transform(train_df[subset])
        Xval_o = sc.transform(val_df[subset])

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 60),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
            "random_state": 42, "verbose": -1,
        }
        mag_proba_thr = trial.suggest_float("mag_proba_thr", 0.35, 0.65)
        dir_conf_thr = trial.suggest_float("dir_conf_thr", 0.52, 0.58)

        mdl = lgb.LGBMClassifier(**params)
        mdl.fit(Xtr_o, y_train_mag)
        mag_proba = mdl.predict_proba(Xval_o)[:, 1]

        mask = (mag_proba >= mag_proba_thr) & (dir_conf_val >= dir_conf_thr)
        n = mask.sum()
        if n < 50:
            return 0.0

        acc = accuracy_score(y_val_dir[mask], dir_pred_val[mask])
        trade_frac = n / len(y_val_dir)

        # Target: high accuracy with at least 1% of candles traded
        if trade_frac < 0.01:
            return acc * (trade_frac / 0.01)
        return acc

    study = optuna.create_study(direction="maximize", study_name="mag_v2")
    study.optimize(objective, n_trials=75, show_progress_bar=True, catch=(KeyboardInterrupt,))

    best = study.best_params
    print(f"\nBest score: {study.best_value:.4f}")
    print(f"Best params: {best}")

    n_feats_best = best.pop("n_feats")
    mag_proba_thr = best.pop("mag_proba_thr")
    dir_conf_thr = best.pop("dir_conf_thr")
    best["random_state"] = 42
    best["verbose"] = -1
    best_feats = ranked_feats[:n_feats_best]

    print(f"\nOptimal: {n_feats_best} features, mag>={mag_proba_thr:.4f}, dir>={dir_conf_thr:.4f}")

    # ================================================================
    # STEP 5: Final OOS evaluation on 48k test
    # ================================================================
    print("\n" + "=" * 65)
    print("STEP 5: FINAL TEST on all 6 OOS files")
    print("=" * 65)

    # Retrain direction model on full 100k
    scaler_dir_full = StandardScaler()
    dl, dx, dr, mc = build_direction_model(full_df, scaler_dir_full)
    dir_pred_test, dir_conf_test = get_dir_predictions(test_df, scaler_dir_full, dl, dx, dr, mc)
    y_test_dir = test_df["direction_label"].values

    # Retrain magnitude model on full 100k with best params
    scaler_mag_full = StandardScaler()
    Xtr_full = scaler_mag_full.fit_transform(full_df[best_feats])
    y_mag_full = (full_df["next_abs_return"] > MAG_THRESHOLD).astype(int).values
    mag_final = lgb.LGBMClassifier(**best)
    mag_final.fit(Xtr_full, y_mag_full)
    Xts_mag = scaler_mag_full.transform(test_df[best_feats])
    mag_proba_test = mag_final.predict_proba(Xts_mag)[:, 1]

    print(f"\nTest samples: {len(test_df):,}")
    print(f"Mag features: {n_feats_best} | Mag threshold: > {MAG_THRESHOLD}%")
    print(f"\n--- Tuned two-stage ---")
    eval_two_stage(mag_proba_test, mag_proba_thr, dir_pred_test, dir_conf_test, dir_conf_thr, y_test_dir, "TUNED")

    print(f"\n--- Comparison grid ---")
    for mp in [0.3, 0.4, 0.5, 0.6]:
        for dc in [0.53, 0.55, 0.57]:
            eval_two_stage(mag_proba_test, mp, dir_pred_test, dir_conf_test, dc, y_test_dir, f"mag>={mp}, dir>={dc}")

    print(f"\n--- Direction-only baselines ---")
    for dc in [0.53, 0.55, 0.57]:
        mask = dir_conf_test >= dc
        n = mask.sum()
        if n > 0:
            acc = accuracy_score(y_test_dir[mask], dir_pred_test[mask])
            print(f"  dir>={dc}: {acc:.4f} ({acc*100:.2f}%) on {n:,}")

    # Save everything
    print("\n--- Saving models ---")
    md = os.path.join(_ROOT, "models")
    os.makedirs(md, exist_ok=True)

    pickle.dump(mag_final, open(os.path.join(md, "mag_model.pkl"), "wb"))
    pickle.dump(scaler_mag_full, open(os.path.join(md, "mag_scaler.pkl"), "wb"))
    pickle.dump(best_feats, open(os.path.join(md, "mag_features.pkl"), "wb"))

    pickle.dump(dl, open(os.path.join(md, "dir_lgb.pkl"), "wb"))
    pickle.dump(dx, open(os.path.join(md, "dir_xgb.pkl"), "wb"))
    pickle.dump(dr, open(os.path.join(md, "dir_rf.pkl"), "wb"))
    pickle.dump(mc, open(os.path.join(md, "dir_meta.pkl"), "wb"))
    pickle.dump(scaler_dir_full, open(os.path.join(md, "dir_scaler.pkl"), "wb"))
    pickle.dump(DIRECTION_FEATURES, open(os.path.join(md, "dir_features.pkl"), "wb"))

    config = {
        "mag_lgb_params": best,
        "mag_features": best_feats,
        "mag_threshold": MAG_THRESHOLD,
        "mag_proba_thr": mag_proba_thr,
        "dir_conf_thr": dir_conf_thr,
        "dir_features": DIRECTION_FEATURES,
    }
    pickle.dump(config, open(os.path.join(md, "two_stage_config.pkl"), "wb"))
    print("Saved all models + config to models/")


if __name__ == "__main__":
    main()
