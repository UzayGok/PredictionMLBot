"""
Evaluate saved two-stage production models on test data.
Usage:
  python eval_10k.py           -> test on btc_candles_10k.csv
  python eval_10k.py sets      -> test on combined set1-5
  python eval_10k.py all       -> test on set1-5 individually + 10k + combined
"""
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
mag_feats   = pickle.load(open(os.path.join(MD, "mag_features.pkl"), "rb"))  # 45
dir_feats   = pickle.load(open(os.path.join(MD, "features.pkl"), "rb"))      # 55
config      = pickle.load(open(os.path.join(MD, "two_stage_config.pkl"), "rb"))

MAG_PROBA_THR = config["mag_proba_thr"]
DIR_CONF_THR  = config["dir_conf_thr"]
all_feats = list(set(mag_feats + dir_feats))


def load_file(fname):
    df = pd.read_csv(os.path.join(_ROOT, "data", fname))
    df = calculate_features(df)
    df["direction_label"] = make_label(df)
    df = df.dropna(subset=all_feats + ["direction_label"]).reset_index(drop=True)
    return df


def run_eval(df, label):
    y_true = df["direction_label"].values

    # Stage 1: Magnitude
    mag_proba = mag_model.predict_proba(mag_scaler.transform(df[mag_feats]))[:, 1]

    # Stage 2: Direction (B2 stacking)
    Xd = dir_scaler.transform(df[dir_feats])
    meta_in = np.column_stack([
        dir_lgb.predict_proba(Xd)[:, 1],
        dir_xgb.predict_proba(Xd)[:, 1],
        dir_rf.predict_proba(Xd)[:, 1],
    ])
    dir_pred  = dir_meta.predict(meta_in)
    dir_proba = dir_meta.predict_proba(meta_in)
    dir_conf  = np.maximum(dir_proba[:, 0], dir_proba[:, 1])

    print("=" * 60)
    print(f"TWO-STAGE RESULTS  --  {label}")
    print("=" * 60)
    print(f"  Samples: {len(y_true):,}")

    print(f"\nProduction thresholds: mag >= {MAG_PROBA_THR}, dir >= {DIR_CONF_THR}")
    mask = (mag_proba >= MAG_PROBA_THR) & (dir_conf >= DIR_CONF_THR)
    n = mask.sum()
    if n:
        acc = accuracy_score(y_true[mask], dir_pred[mask])
        up  = (dir_pred[mask] == 1).sum()
        dn  = (dir_pred[mask] == 0).sum()
        print(f"  Accuracy: {acc*100:.2f}%")
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

    print(f"\nDirection-only baselines (no mag filter):")
    for dc in [0.53, 0.55, 0.57, 0.59]:
        m = dir_conf >= dc
        nn = m.sum()
        if nn > 0:
            a = accuracy_score(y_true[m], dir_pred[m])
            print(f"  dir>={dc:.2f}: {a*100:.2f}% on {nn:,}")

    base = max(y_true.mean(), 1 - y_true.mean())
    print(f"\nBaseline (always majority): {base*100:.2f}%")
    print()


mode = sys.argv[1] if len(sys.argv) > 1 else "10k"

if mode == "10k":
    print("Loading btc_candles_10k.csv ...")
    run_eval(load_file("btc_candles_10k.csv"), "btc_candles_10k.csv")

elif mode == "sets":
    print("Loading btc_candles_set1-5 ...")
    dfs = [load_file(f"btc_candles_set{i}.csv") for i in range(1, 6)]
    combined = pd.concat(dfs, ignore_index=True)
    run_eval(combined, "Combined set1-5")

elif mode == "all":
    # Individual set files
    for i in range(1, 6):
        fname = f"btc_candles_set{i}.csv"
        print(f"Loading {fname} ...")
        run_eval(load_file(fname), fname)
    # 10k
    print("Loading btc_candles_10k.csv ...")
    run_eval(load_file("btc_candles_10k.csv"), "btc_candles_10k.csv")
    # Combined sets
    print("Loading combined set1-5 ...")
    dfs = [load_file(f"btc_candles_set{i}.csv") for i in range(1, 6)]
    run_eval(pd.concat(dfs, ignore_index=True), "Combined set1-5")

else:
    print(f"Unknown mode '{mode}'. Use: 10k | sets | all")
