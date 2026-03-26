"""
predict_live.py
Two-stage live prediction:
  Stage 1: Magnitude model — is a big move (>0.10%) coming?
  Stage 2: B2 stacking direction model — which direction?
Only signals a trade when both stages pass their thresholds.
"""

import datetime
import os
import pickle
import sys

import numpy as np
import pandas as pd

from .data_fetcher import fetch_candles, current_boundary_utc

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from Training.features import calculate_features


# ---------------------------------------------------------------------------
# Two-stage prediction
# ---------------------------------------------------------------------------

MAG_PROBA_THR = 0.50
DIR_CONF_THR = 0.55
LIVE_CANDLE_LIMIT = 2500


def predict_two_stage(df, mag_model, mag_scaler, mag_features,
                       dir_models, dir_scaler, dir_features):
    """
    Two-stage prediction on the last row.
    Returns dict with signal, mag_proba, dir_conf, trade flag.
    """
    # Stage 1: magnitude (45 DIRECTION_FEATURES)
    mag_scaled = mag_scaler.transform(df[mag_features].iloc[[-1]])
    mag_proba = mag_model.predict_proba(mag_scaled)[0][1]  # P(big move)

    # Stage 2: direction (55 STACKING_FEATURES)
    dir_scaled = dir_scaler.transform(df[dir_features].iloc[[-1]])
    d_lgb, d_xgb, d_rf, meta_clf = dir_models
    meta = np.array([[
        d_lgb.predict_proba(dir_scaled)[0][1],
        d_xgb.predict_proba(dir_scaled)[0][1],
        d_rf.predict_proba(dir_scaled)[0][1],
    ]])
    dir_proba = meta_clf.predict_proba(meta)[0]
    prob_up = dir_proba[1]
    prob_down = dir_proba[0]
    dir_conf = max(prob_up, prob_down)
    signal = "UP" if prob_up >= prob_down else "DOWN"

    trade = (mag_proba >= MAG_PROBA_THR) and (dir_conf >= DIR_CONF_THR)

    return {
        "signal": signal,
        "mag_proba": mag_proba,
        "dir_conf": dir_conf,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "trade": trade,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    models_dir = os.path.join(_ROOT, "models")

    # Load magnitude model
    mag_model = pickle.load(open(os.path.join(models_dir, "mag_model.pkl"), "rb"))
    mag_scaler = pickle.load(open(os.path.join(models_dir, "mag_scaler.pkl"), "rb"))

    # Load direction B2 stacking models
    d_lgb = pickle.load(open(os.path.join(models_dir, "dir_lgb.pkl"), "rb"))
    d_xgb = pickle.load(open(os.path.join(models_dir, "dir_xgb.pkl"), "rb"))
    d_rf = pickle.load(open(os.path.join(models_dir, "dir_rf.pkl"), "rb"))
    meta_clf = pickle.load(open(os.path.join(models_dir, "dir_meta.pkl"), "rb"))
    dir_scaler = pickle.load(open(os.path.join(models_dir, "dir_scaler.pkl"), "rb"))
    mag_features = pickle.load(open(os.path.join(models_dir, "mag_features.pkl"), "rb"))
    dir_features = pickle.load(open(os.path.join(models_dir, "features.pkl"), "rb"))

    dir_models = (d_lgb, d_xgb, d_rf, meta_clf)

    # Fetch enough history to satisfy the longest rolling features.
    df = fetch_candles(limit=LIVE_CANDLE_LIMIT)
    df = calculate_features(df)
    required = list(set(mag_features + dir_features))
    df = df.dropna(subset=required).reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "No valid rows after feature calculation. Try increasing LIVE_CANDLE_LIMIT."
        )

    # Candle timing
    last_candle_open = df["timestamp"].iloc[-1]
    candle_close = last_candle_open + datetime.timedelta(minutes=5)
    predicting_for_open = candle_close
    predicting_for_close = predicting_for_open + datetime.timedelta(minutes=5)

    # Two-stage prediction
    result = predict_two_stage(
        df, mag_model, mag_scaler, mag_features,
        dir_models, dir_scaler, dir_features
    )

    # Display
    now_utc = datetime.datetime.utcnow().strftime("%H:%M:%S")
    last_open_str = last_candle_open.strftime("%H:%M")
    last_close_str = candle_close.strftime("%H:%M")
    pred_open_str = predicting_for_open.strftime("%H:%M")
    pred_close_str = predicting_for_close.strftime("%H:%M")

    trade_str = ">>> TRADE <<<" if result["trade"] else "SKIP (no trade)"

    print()
    print(f"  Time (UTC):          {now_utc}")
    print()
    print(f"  Last candle:         {last_open_str} - {last_close_str} UTC")
    print(f"  Last candle close:   ${df['close'].iloc[-1]:,.2f}")
    print(f"  Predicting candle:   {pred_open_str} - {pred_close_str} UTC")
    print()
    print(f"  Stage 1 — Big move:  {result['mag_proba']:.1%}  (thr: {MAG_PROBA_THR:.0%})")
    print(f"  Stage 2 — Direction: {result['signal']}  conf: {result['dir_conf']:.1%}  (thr: {DIR_CONF_THR:.0%})")
    print(f"  P(UP):               {result['prob_up']:.1%}")
    print(f"  P(DOWN):             {result['prob_down']:.1%}")
    print()
    print(f"  Decision:            {trade_str}")
    if result["trade"]:
        print(f"  Signal:              {result['signal']}")
    print()


if __name__ == "__main__":
    main()
