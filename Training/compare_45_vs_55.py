"""
Compare Current 45 vs Top 55 (stacking-ranked) on ALL test files individually.
"""
import os, sys, warnings
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
    DIRECTION_FEATURES,
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

TOP_55 = [
    "lag_return_1", "high_low_ratio", "return_24h", "range_pos_12",
    "lag_return_5", "body_ratio", "lag_rsi_1", "lag_volume_ratio_2",
    "lag_return_3", "lag_return_2", "trend_strength_24h", "ema_slope_10",
    "upper_wick_ratio", "range_expansion", "lower_wick_ratio",
    "lag_volume_ratio_1", "volume_trend_1h_24h", "macd_diff",
    "lag_return_10", "log_return_5", "rsi_1h", "ema_slope_20",
    "williams_r", "price_vs_vwap_24h", "atr_regime", "volume_ratio",
    "ema_slope_5", "bb_pct", "atr_1h", "rsi_14", "vol_percentile_week",
    "volume_ratio_60", "range_pos_48", "vol_expansion", "num_trades",
    "vol_ratio_10_50", "price_change_pct", "volume_acceleration",
    "lag_volume_ratio_3", "volume", "volatility_50", "volatility_20",
    "return_1h", "gap_pct", "momentum_10", "price_vs_ema_20",
    "lag_rsi_3", "volume_sma", "price_change", "log_return_10",
    "return_4h", "log_return_3", "price_range", "log_return_20",
    "macd_signal",
]


def load_data(path, feats):
    df = pd.read_csv(path)
    df = calculate_features(df)
    df["direction_label"] = make_label(df)
    df["lm_label"] = make_label_large_moves(df, PCT_THR)
    df["next_return_pct"] = (df["close"].shift(-1) - df["close"]) / df["close"] * 100
    df["next_abs_return"] = df["next_return_pct"].abs()
    df = df.dropna(subset=feats + ["direction_label"]).reset_index(drop=True)
    return df


def train_and_eval(train_df, test_df, feats, mag_proba, y_true, label):
    train_lm = train_df.dropna(subset=["lm_label"]).reset_index(drop=True)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(train_lm[feats])
    y_all = train_lm["lm_label"].values

    sp = int(len(X_all) * 0.7)
    X_base, X_meta = X_all[:sp], X_all[sp:]
    y_base, y_meta = y_all[:sp], y_all[sp:]

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

    meta_hold = np.column_stack([
        d_lgb.predict_proba(X_meta)[:, 1],
        d_xgb.predict_proba(X_meta)[:, 1],
        d_rf.predict_proba(X_meta)[:, 1],
    ])
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    meta_clf.fit(meta_hold, y_meta)

    # Retrain on full
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

    X_test = scaler.transform(test_df[feats])
    meta_test = np.column_stack([
        d_lgb2.predict_proba(X_test)[:, 1],
        d_xgb2.predict_proba(X_test)[:, 1],
        d_rf2.predict_proba(X_test)[:, 1],
    ])
    pred = meta_clf.predict(meta_test)
    proba = meta_clf.predict_proba(meta_test)
    conf = np.maximum(proba[:, 0], proba[:, 1])
    return pred, conf


def main():
    # All features superset (need TOP_55 which includes extras)
    all_feats = list(set(DIRECTION_FEATURES + TOP_55))

    print("Loading training data...")
    train_df = load_data(os.path.join(_ROOT, "data", "btc_candles_100k.csv"), all_feats)
    print(f"  Train: {len(train_df):,}")

    test_files = [f"btc_candles_set{i}.csv" for i in range(1, 6)] + ["btc_candles_10k.csv"]

    # Mag model (always uses DIRECTION_FEATURES)
    print("Training mag model...")
    mag_scaler = StandardScaler()
    X_mag_tr = mag_scaler.fit_transform(train_df[DIRECTION_FEATURES])
    y_mag_tr = (train_df["next_abs_return"] > MAG_THRESHOLD).astype(int).values
    mag_model = lgb.LGBMClassifier(**MAG_PARAMS)
    mag_model.fit(X_mag_tr, y_mag_tr)

    # Train both direction configs once
    print("Training Current 45 direction model...")
    sys.stdout.flush()

    results = []

    for tf in test_files:
        path = os.path.join(_ROOT, "data", tf)
        tdf = load_data(path, all_feats)
        y_true = tdf["direction_label"].values
        mag_proba = mag_model.predict_proba(mag_scaler.transform(tdf[DIRECTION_FEATURES]))[:, 1]

        row = {"file": tf, "n": len(tdf)}

        for feat_label, feats in [("Current 45", DIRECTION_FEATURES), ("Top 55", TOP_55)]:
            pred, conf = train_and_eval(train_df, tdf, feats, mag_proba, y_true, feat_label)
            for mt in [0.5]:
                for dc in [0.53, 0.55, 0.57]:
                    m = (mag_proba >= mt) & (conf >= dc)
                    nn = m.sum()
                    acc = accuracy_score(y_true[m], pred[m]) if nn > 0 else 0
                    row[f"{feat_label}_m50_d{int(dc*100)}"] = (acc, nn)

        results.append(row)
        print(f"  Done: {tf}")
        sys.stdout.flush()

    # Print results
    print("\n" + "=" * 80)
    print("COMPARISON: Current 45 vs Top 55 (stacking-ranked)")
    print("=" * 80)

    for dc in [0.53, 0.55, 0.57]:
        dc_str = f"d{int(dc*100)}"
        print(f"\n--- mag>=0.50, dir>={dc:.2f} ---")
        print(f"{'File':<25} {'Current 45':>18} {'Top 55':>18} {'Delta':>8}")
        print("-" * 72)
        tot_45_correct = tot_45_n = tot_55_correct = tot_55_n = 0
        for row in results:
            a45, n45 = row[f"Current 45_m50_{dc_str}"]
            a55, n55 = row[f"Top 55_m50_{dc_str}"]
            delta = (a55 - a45) * 100 if n45 > 0 and n55 > 0 else 0
            s45 = f"{a45*100:.2f}% ({n45:>3})" if n45 > 0 else "  N/A"
            s55 = f"{a55*100:.2f}% ({n55:>3})" if n55 > 0 else "  N/A"
            sign = "+" if delta > 0 else ""
            print(f"{row['file']:<25} {s45:>18} {s55:>18} {sign}{delta:>6.2f}%")
            if n45 > 0:
                tot_45_correct += round(a45 * n45)
                tot_45_n += n45
            if n55 > 0:
                tot_55_correct += round(a55 * n55)
                tot_55_n += n55
        if tot_45_n > 0 and tot_55_n > 0:
            avg45 = tot_45_correct / tot_45_n * 100
            avg55 = tot_55_correct / tot_55_n * 100
            d = avg55 - avg45
            sign = "+" if d > 0 else ""
            print(f"{'COMBINED':<25} {avg45:>11.2f}% ({tot_45_n:>3}) {avg55:>11.2f}% ({tot_55_n:>3}) {sign}{d:>6.2f}%")


if __name__ == "__main__":
    main()
