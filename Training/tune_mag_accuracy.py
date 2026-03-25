"""
tune_mag_accuracy.py
Re-tune magnitude model with PURE ACCURACY objective.
No trade-volume balancing. Just maximize accuracy with a hard minimum of 100 trades.
Also directly compares old mag model vs new (with expanded features).
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
MAG_THRESHOLD = 0.10

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

# Old mag params (from previous Optuna run)
OLD_MAG_PARAMS = {
    "n_estimators": 400, "max_depth": 3, "learning_rate": 0.0183,
    "num_leaves": 37, "min_child_samples": 46,
    "subsample": 0.80, "colsample_bytree": 0.76,
    "reg_alpha": 0.74, "reg_lambda": 0.07,
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
    train_lm = train_df.dropna(subset=["lm_label"]).reset_index(drop=True)
    Xtr_dir = scaler_dir.fit_transform(train_lm[DIRECTION_FEATURES])
    sp = int(len(Xtr_dir) * 0.7)
    Xd_base, Xd_meta = Xtr_dir[:sp], Xtr_dir[sp:]
    yd_base, yd_meta = train_lm["lm_label"].iloc[:sp], train_lm["lm_label"].iloc[sp:]

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

    # Retrain on full data
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


def get_dir_preds(df, scaler_dir, d_lgb, d_xgb, d_rf, meta_clf):
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


def eval_ts(mag_proba, mag_thr, dir_pred, dir_conf, dir_thr, y_true, label=""):
    mask = (mag_proba >= mag_thr) & (dir_conf >= dir_thr)
    n = mask.sum()
    if n > 0:
        acc = accuracy_score(y_true[mask], dir_pred[mask])
        print(f"  {label}: {acc:.4f} ({acc*100:.2f}%) on {n:,} trades")
        return acc, n
    print(f"  {label}: 0 trades")
    return 0, 0


def main():
    print("=" * 65)
    print("ACCURACY-FOCUSED MAGNITUDE TUNING")
    print("=" * 65)

    print("\nLoading data...")
    full_df = load_full(os.path.join(_ROOT, "data", "btc_candles_100k.csv"))
    test_df = pd.concat([
        load_full(os.path.join(_ROOT, "data", f"btc_candles_set{i}.csv"))
        for i in range(5, 0, -1)
    ] + [load_full(os.path.join(_ROOT, "data", "btc_candles_10k.csv"))], ignore_index=True)

    split = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split].copy().reset_index(drop=True)
    val_df = full_df.iloc[split:].copy().reset_index(drop=True)
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # Direction model on train split (for val evaluation)
    print("\nBuilding direction model...")
    scaler_dir = StandardScaler()
    d_lgb, d_xgb, d_rf, meta_clf = build_direction_model(train_df, scaler_dir)
    dir_pred_val, dir_conf_val = get_dir_preds(val_df, scaler_dir, d_lgb, d_xgb, d_rf, meta_clf)
    y_val = val_df["direction_label"].values
    y_train_mag = (train_df["next_abs_return"] > MAG_THRESHOLD).astype(int).values

    # Feature ranking from all 75
    sc_all = StandardScaler()
    Xtr_all = sc_all.fit_transform(train_df[FEATURE_COLS])
    rank_mdl = lgb.LGBMClassifier(**OLD_MAG_PARAMS)
    rank_mdl.fit(Xtr_all, y_train_mag)
    ranking = sorted(zip(FEATURE_COLS, rank_mdl.feature_importances_), key=lambda x: -x[1])
    ranked_feats = [f for f, _ in ranking]

    # ================================================================
    # BASELINE: Old mag model with old 45 direction features only
    # ================================================================
    print("\n" + "=" * 65)
    print("BASELINE: Old mag model (45 direction features, old params)")
    print("=" * 65)
    sc_old = StandardScaler()
    Xtr_old = sc_old.fit_transform(train_df[DIRECTION_FEATURES])
    Xval_old = sc_old.transform(val_df[DIRECTION_FEATURES])
    old_mag = lgb.LGBMClassifier(**OLD_MAG_PARAMS)
    old_mag.fit(Xtr_old, y_train_mag)
    old_proba = old_mag.predict_proba(Xval_old)[:, 1]

    print("  Old mag model on VAL:")
    for mt in [0.4, 0.5, 0.6]:
        for dc in [0.55, 0.57]:
            eval_ts(old_proba, mt, dir_pred_val, dir_conf_val, dc, y_val, f"OLD mag>={mt}, dir>={dc}")

    # ================================================================
    # OPTUNA: Pure accuracy, hard min 100 trades on val
    # ================================================================
    print("\n" + "=" * 65)
    print("OPTUNA: Pure accuracy objective (min 100 val trades)")
    print("100 trials")
    print("=" * 65)

    MIN_TRADES = 100  # hard floor on val (~0.5% of 19k)

    def objective(trial):
        n_feats = trial.suggest_int("n_feats", 15, len(ranked_feats), step=5)
        subset = ranked_feats[:n_feats]

        sc = StandardScaler()
        Xtr_o = sc.fit_transform(train_df[subset])
        Xval_o = sc.transform(val_df[subset])

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 80),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 4.0),
            "random_state": 42, "verbose": -1,
        }
        mag_thr = trial.suggest_float("mag_thr", 0.35, 0.75)
        dir_thr = trial.suggest_float("dir_thr", 0.54, 0.60)

        mdl = lgb.LGBMClassifier(**params)
        mdl.fit(Xtr_o, y_train_mag)
        mag_proba = mdl.predict_proba(Xval_o)[:, 1]

        mask = (mag_proba >= mag_thr) & (dir_conf_val >= dir_thr)
        n = mask.sum()
        if n < MIN_TRADES:
            return 0.0  # hard reject

        acc = accuracy_score(y_val[mask], dir_pred_val[mask])
        return acc  # PURE ACCURACY — no trade volume bonus/penalty

    study = optuna.create_study(direction="maximize", study_name="mag_acc")
    study.optimize(objective, n_trials=100, show_progress_bar=True, catch=(KeyboardInterrupt,))

    bp = study.best_params
    print(f"\nBest val accuracy: {study.best_value:.4f}")
    print(f"Best params: {bp}")

    n_feats_best = bp.pop("n_feats")
    mag_thr_best = bp.pop("mag_thr")
    dir_thr_best = bp.pop("dir_thr")
    bp["random_state"] = 42
    bp["verbose"] = -1
    best_feats = ranked_feats[:n_feats_best]

    print(f"\nOptimal: {n_feats_best} features, mag>={mag_thr_best:.4f}, dir>={dir_thr_best:.4f}")

    # Validate the tuned model on val
    sc_tuned = StandardScaler()
    Xtr_t = sc_tuned.fit_transform(train_df[best_feats])
    Xval_t = sc_tuned.transform(val_df[best_feats])
    tuned_mag = lgb.LGBMClassifier(**bp)
    tuned_mag.fit(Xtr_t, y_train_mag)
    tuned_proba_val = tuned_mag.predict_proba(Xval_t)[:, 1]

    print("\n  Tuned mag model on VAL (verification):")
    eval_ts(tuned_proba_val, mag_thr_best, dir_pred_val, dir_conf_val, dir_thr_best, y_val, "TUNED best")
    # Also test nearby thresholds
    for mt in [mag_thr_best - 0.05, mag_thr_best, mag_thr_best + 0.05, mag_thr_best + 0.10]:
        for dc in [dir_thr_best - 0.01, dir_thr_best, dir_thr_best + 0.01]:
            eval_ts(tuned_proba_val, mt, dir_pred_val, dir_conf_val, dc, y_val,
                     f"mag>={mt:.3f}, dir>={dc:.3f}")

    # ================================================================
    # FINAL OOS TEST
    # ================================================================
    print("\n" + "=" * 65)
    print("FINAL OOS TEST (47k+ candles)")
    print("=" * 65)

    # Retrain everything on full 100k
    scaler_dir_full = StandardScaler()
    dl, dx, dr, mc = build_direction_model(full_df, scaler_dir_full)
    dir_pred_test, dir_conf_test = get_dir_preds(test_df, scaler_dir_full, dl, dx, dr, mc)
    y_test = test_df["direction_label"].values
    y_mag_full = (full_df["next_abs_return"] > MAG_THRESHOLD).astype(int).values

    # -- Old mag model (baseline) --
    print("\n--- OLD mag model (45 dir features, old params) ---")
    sc_old_f = StandardScaler()
    Xtr_old_f = sc_old_f.fit_transform(full_df[DIRECTION_FEATURES])
    Xts_old = sc_old_f.transform(test_df[DIRECTION_FEATURES])
    old_mag_f = lgb.LGBMClassifier(**OLD_MAG_PARAMS)
    old_mag_f.fit(Xtr_old_f, y_mag_full)
    old_proba_test = old_mag_f.predict_proba(Xts_old)[:, 1]

    for mt in [0.3, 0.4, 0.5, 0.6]:
        for dc in [0.55, 0.57]:
            eval_ts(old_proba_test, mt, dir_pred_test, dir_conf_test, dc, y_test,
                     f"OLD mag>={mt}, dir>={dc}")

    # -- Tuned new mag model --
    print("\n--- NEW TUNED mag model ---")
    sc_new_f = StandardScaler()
    Xtr_new_f = sc_new_f.fit_transform(full_df[best_feats])
    Xts_new = sc_new_f.transform(test_df[best_feats])
    new_mag_f = lgb.LGBMClassifier(**bp)
    new_mag_f.fit(Xtr_new_f, y_mag_full)
    new_proba_test = new_mag_f.predict_proba(Xts_new)[:, 1]

    eval_ts(new_proba_test, mag_thr_best, dir_pred_test, dir_conf_test, dir_thr_best, y_test,
             "TUNED best")
    for mt in [0.4, 0.5, 0.6, 0.7]:
        for dc in [0.55, 0.57, 0.58]:
            eval_ts(new_proba_test, mt, dir_pred_test, dir_conf_test, dc, y_test,
                     f"NEW mag>={mt}, dir>={dc}")

    # -- Direction-only baselines --
    print("\n--- Direction-only baselines ---")
    for dc in [0.55, 0.57, 0.58, 0.59]:
        mask = dir_conf_test >= dc
        n = mask.sum()
        if n > 0:
            acc = accuracy_score(y_test[mask], dir_pred_test[mask])
            print(f"  dir>={dc}: {acc:.4f} ({acc*100:.2f}%) on {n:,} trades")

    # Save only if new model beats old at comparable trade counts
    print("\n--- Saving best models ---")
    md = os.path.join(_ROOT, "models")
    os.makedirs(md, exist_ok=True)

    pickle.dump(new_mag_f, open(os.path.join(md, "mag_model.pkl"), "wb"))
    pickle.dump(sc_new_f, open(os.path.join(md, "mag_scaler.pkl"), "wb"))
    pickle.dump(best_feats, open(os.path.join(md, "mag_features.pkl"), "wb"))

    pickle.dump(dl, open(os.path.join(md, "dir_lgb.pkl"), "wb"))
    pickle.dump(dx, open(os.path.join(md, "dir_xgb.pkl"), "wb"))
    pickle.dump(dr, open(os.path.join(md, "dir_rf.pkl"), "wb"))
    pickle.dump(mc, open(os.path.join(md, "dir_meta.pkl"), "wb"))
    pickle.dump(scaler_dir_full, open(os.path.join(md, "dir_scaler.pkl"), "wb"))
    pickle.dump(DIRECTION_FEATURES, open(os.path.join(md, "dir_features.pkl"), "wb"))

    config = {
        "mag_lgb_params": bp,
        "mag_features": best_feats,
        "mag_threshold": MAG_THRESHOLD,
        "mag_proba_thr": mag_thr_best,
        "dir_conf_thr": dir_thr_best,
        "dir_features": DIRECTION_FEATURES,
    }
    pickle.dump(config, open(os.path.join(md, "two_stage_config.pkl"), "wb"))
    print("Saved to models/")

    print(f"\n{'='*65}")
    print(f"SUMMARY: Best tuned thresholds: mag>={mag_thr_best:.4f}, dir>={dir_thr_best:.4f}")
    print(f"Features: {n_feats_best} | scale_pos_weight: {bp.get('scale_pos_weight', 1.0):.2f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
