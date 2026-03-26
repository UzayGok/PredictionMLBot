"""
run_scheduler.py
Runs the two-stage BTC prediction every 5 minutes, aligned to candle closes.
Sends Telegram and/or email alerts when a trade signal fires.

Usage:
    python run_scheduler.py
"""

import datetime
import os
import pickle
import socket
import smtplib
import sys
import time
import traceback
import atexit
from email.mime.text import MIMEText

from dotenv import load_dotenv
import requests

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

load_dotenv(os.path.join(_ROOT, ".env"))

from Predict.data_fetcher import fetch_candles, current_boundary_utc
from Training.features import calculate_features
from Predict.predict_live import predict_two_stage, MAG_PROBA_THR, DIR_CONF_THR, LIVE_CANDLE_LIMIT

# Notification config from .env
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
ALLOWED_TELEGRAM_CHAT_IDS = {TELEGRAM_CHAT_ID} if TELEGRAM_CHAT_ID else set()

EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").strip().lower() == "true"
EMAIL_TO = os.getenv("EMAIL_TO", "").strip()
EMAIL_FROM = os.getenv("EMAIL_FROM", "").strip()
SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
SMTP_USER = os.getenv("EMAIL_SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("EMAIL_SMTP_PASSWORD", "").strip()

LOG_DIR = os.path.join(_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "scheduler.log")
LOCK_HOST = "127.0.0.1"
LOCK_PORT = 49555
_INSTANCE_SOCKET = None


class TeeStream:
    def __init__(self, file_stream, console_stream=None):
        self.file_stream = file_stream
        self.console_stream = console_stream

    def write(self, data):
        if not data:
            return
        self.file_stream.write(data)
        self.file_stream.flush()
        if self.console_stream is not None:
            self.console_stream.write(data)
            self.console_stream.flush()

    def flush(self):
        self.file_stream.flush()
        if self.console_stream is not None:
            self.console_stream.flush()


def configure_logging() -> None:
    """Mirror stdout/stderr to logs/scheduler.log, including pythonw runs."""
    os.makedirs(LOG_DIR, exist_ok=True)
    file_stream = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    atexit.register(file_stream.close)
    sys.stdout = TeeStream(file_stream, sys.stdout)
    sys.stderr = TeeStream(file_stream, sys.stderr)


def acquire_single_instance_lock() -> bool:
    """Allow only one scheduler process per machine/user session."""
    global _INSTANCE_SOCKET

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)

    try:
        sock.bind((LOCK_HOST, LOCK_PORT))
        sock.listen(1)
    except OSError:
        sock.close()
        return False

    _INSTANCE_SOCKET = sock
    return True


def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def email_enabled() -> bool:
    return EMAIL_ENABLED and all([EMAIL_TO, EMAIL_FROM, SMTP_HOST, SMTP_USER, SMTP_PASSWORD])


def get_telegram_chat(chat_id: str) -> dict:
    """Fetch Telegram chat metadata for validation."""
    response = requests.get(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getChat",
        params={"chat_id": chat_id},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise ValueError(f"Telegram getChat failed: {payload}")
    return payload["result"]


def validate_telegram_target(chat_id: str) -> None:
    """Allow alerts only to the configured private chat."""
    if chat_id not in ALLOWED_TELEGRAM_CHAT_IDS:
        raise ValueError(f"Refusing to send Telegram alert to non-whitelisted chat: {chat_id}")

    chat = get_telegram_chat(chat_id)
    if chat.get("type") != "private":
        raise ValueError(
            f"Refusing to send Telegram alert to non-private chat {chat_id} (type={chat.get('type')})"
        )


def format_signal_message(
    now_utc: str,
    last_candle_open: datetime.datetime,
    candle_close: datetime.datetime,
    pred_open: datetime.datetime,
    pred_close: datetime.datetime,
    price: float,
    result: dict,
) -> str:
    return (
        f"Trade Signal: BTC {result['signal']}\n"
        f"\n"
        f"Time (UTC):           {now_utc}\n"
        f"Last candle:          {last_candle_open.strftime('%Y-%m-%d %H:%M')} - {candle_close.strftime('%H:%M')} UTC\n"
        f"Last candle close:    ${price:,.2f}\n"
        f"Predicting candle:    {pred_open.strftime('%H:%M')} - {pred_close.strftime('%H:%M')} UTC\n"
        f"\n"
        f"Stage 1 — Big move:   {result['mag_proba']:.1%}  (threshold: {MAG_PROBA_THR:.0%})\n"
        f"Stage 2 — Direction:  {result['signal']}  confidence: {result['dir_conf']:.1%}  (threshold: {DIR_CONF_THR:.0%})\n"
        f"P(UP):                {result['prob_up']:.1%}\n"
        f"P(DOWN):              {result['prob_down']:.1%}\n"
        f"\n"
        f"Decision:             TRADE {result['signal']}\n"
    )


def send_telegram_message(subject: str, body: str, chat_id: str = TELEGRAM_CHAT_ID) -> None:
    """Send a plain-text Telegram message via the Bot API."""
    validate_telegram_target(chat_id)
    response = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": f"{subject}\n\n{body}",
        },
        timeout=30,
    )
    response.raise_for_status()


def send_telegram_notification(subject: str, body: str) -> None:
    """Send a Telegram notification if Telegram is configured."""
    if not telegram_enabled():
        return
    send_telegram_message(subject, body)


def send_error_notification(exc: Exception) -> None:
    """Send a concise Telegram alert when the scheduler hits an error."""
    if not telegram_enabled():
        return

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    error_type = type(exc).__name__
    body = (
        f"Time (UTC):           {timestamp}\n"
        f"Error type:           {error_type}\n"
        f"Error message:        {exc}\n"
        f"\n"
        f"PredictionMLBot scheduler hit an error and will retry next cycle."
    )
    try:
        send_telegram_notification("PredictionMLBot Error", body)
    except Exception as notify_exc:
        print(f"  [ERROR] Failed to send Telegram error notification: {notify_exc}")


def send_email(subject: str, body: str) -> None:
    """Send a plain-text email via SMTP/TLS."""
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())


def load_models():
    """Load all production models once."""
    models_dir = os.path.join(_ROOT, "models")

    mag_model = pickle.load(open(os.path.join(models_dir, "mag_model.pkl"), "rb"))
    mag_scaler = pickle.load(open(os.path.join(models_dir, "mag_scaler.pkl"), "rb"))

    d_lgb = pickle.load(open(os.path.join(models_dir, "dir_lgb.pkl"), "rb"))
    d_xgb = pickle.load(open(os.path.join(models_dir, "dir_xgb.pkl"), "rb"))
    d_rf = pickle.load(open(os.path.join(models_dir, "dir_rf.pkl"), "rb"))
    meta_clf = pickle.load(open(os.path.join(models_dir, "dir_meta.pkl"), "rb"))
    dir_scaler = pickle.load(open(os.path.join(models_dir, "dir_scaler.pkl"), "rb"))
    mag_features = pickle.load(open(os.path.join(models_dir, "mag_features.pkl"), "rb"))
    dir_features = pickle.load(open(os.path.join(models_dir, "features.pkl"), "rb"))

    return {
        "mag_model": mag_model,
        "mag_scaler": mag_scaler,
        "dir_models": (d_lgb, d_xgb, d_rf, meta_clf),
        "dir_scaler": dir_scaler,
        "mag_features": mag_features,
        "dir_features": dir_features,
    }


def run_prediction(m: dict) -> None:
    """Fetch candles, predict, print, and notify if trade."""
    df = fetch_candles(limit=LIVE_CANDLE_LIMIT)
    df = calculate_features(df)
    required = list(set(m["mag_features"] + m["dir_features"]))
    df = df.dropna(subset=required).reset_index(drop=True)

    if df.empty:
        print("  [WARN] No valid rows after features — skipping this cycle.")
        return

    last_candle_open = df["timestamp"].iloc[-1]
    candle_close = last_candle_open + datetime.timedelta(minutes=5)
    pred_open = candle_close
    pred_close = pred_open + datetime.timedelta(minutes=5)
    price = df["close"].iloc[-1]

    result = predict_two_stage(
        df,
        m["mag_model"], m["mag_scaler"], m["mag_features"],
        m["dir_models"], m["dir_scaler"], m["dir_features"],
    )

    now_utc = datetime.datetime.utcnow().strftime("%H:%M:%S")
    decision = ">>> TRADE <<<" if result["trade"] else "SKIP"

    print(f"  [{now_utc}]  Candle {last_candle_open.strftime('%H:%M')}-{candle_close.strftime('%H:%M')}  "
          f"Close ${price:,.2f}  |  Mag {result['mag_proba']:.1%}  Dir {result['signal']} {result['dir_conf']:.1%}  "
          f"|  {decision}")

    if result["trade"]:
        subject = f"BTC going {result['signal']}"
        body = format_signal_message(
            now_utc,
            last_candle_open,
            candle_close,
            pred_open,
            pred_close,
            price,
            result,
        )

        if telegram_enabled():
            try:
                send_telegram_notification(subject, body)
                print(f"           Telegram sent → chat {TELEGRAM_CHAT_ID}")
            except Exception as exc:
                print(f"           [ERROR] Telegram failed: {exc}")

        if email_enabled():
            try:
                send_email(subject, body)
                print(f"           Email sent → {EMAIL_TO}")
            except Exception as exc:
                print(f"           [ERROR] Email failed: {exc}")
    else:
        if telegram_enabled():
            try:
                send_telegram_notification(
                    f"SKIP, {result['signal']}, {result['dir_conf']:.2%}",
                    "",
                )
                print(f"           Telegram skip sent → chat {TELEGRAM_CHAT_ID}")
            except Exception as exc:
                print(f"           [ERROR] Telegram skip failed: {exc}")


def seconds_until_next_boundary() -> float:
    """Seconds until the next Binance 5-minute boundary plus a small buffer."""
    now = datetime.datetime.now(datetime.timezone.utc)
    target = current_boundary_utc() + datetime.timedelta(minutes=5, seconds=5)
    wait = (target - now).total_seconds()
    return max(wait, 1)


def main():
    configure_logging()

    if not acquire_single_instance_lock():
        print("Another PredictionMLBot scheduler instance is already running. Exiting.")
        return

    if not (telegram_enabled() or email_enabled()):
        raise ValueError(
            "No notification channel configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, "
            "or provide the EMAIL_* settings in .env."
        )

    if telegram_enabled():
        validate_telegram_target(TELEGRAM_CHAT_ID)

    print("=" * 70)
    print("  BTC 5-min Prediction Scheduler")
    if telegram_enabled():
        print(f"  Telegram alerts → chat {TELEGRAM_CHAT_ID}")
    if email_enabled():
        print(f"  Email alerts → {EMAIL_TO}")
    print(f"  Thresholds: mag >= {MAG_PROBA_THR:.0%}, dir >= {DIR_CONF_THR:.0%}")
    print("=" * 70)
    print()

    print("Loading models...")
    m = load_models()
    print("Models loaded. Waiting for next 5-min boundary...\n")

    if telegram_enabled():
        try:
            send_telegram_notification(
                "PredictionMLBot Started",
                (
                    f"Time (UTC):           {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Telegram chat:        {TELEGRAM_CHAT_ID}\n"
                    f"Thresholds:           mag >= {MAG_PROBA_THR:.0%}, dir >= {DIR_CONF_THR:.0%}\n"
                    f"Status:               Scheduler started and waiting for the next 5-minute boundary."
                ),
            )
            print("Startup Telegram notification sent.")
        except Exception as exc:
            print(f"  [ERROR] Failed to send startup Telegram notification: {exc}")

    while True:
        wait = seconds_until_next_boundary()
        next_run = datetime.datetime.utcnow() + datetime.timedelta(seconds=wait)
        print(f"  Next run at ~{next_run.strftime('%H:%M:%S')} UTC  (sleeping {wait:.0f}s)")
        time.sleep(wait)

        try:
            run_prediction(m)
        except Exception as exc:
            traceback.print_exc()
            send_error_notification(exc)
            print("  [ERROR] Prediction failed — will retry next cycle.\n")


if __name__ == "__main__":
    main()
