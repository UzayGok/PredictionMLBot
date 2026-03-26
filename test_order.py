"""One-shot test: place an order and send Telegram notification."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Trade.market import get_btc_5m_market
from Trade.auth import get_clob_client
from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions
from run_scheduler import send_telegram_notification, telegram_enabled

market = get_btc_5m_market()  # auto-picks next upcoming market
print(f"Market: {market['title']}")

client = get_clob_client()
order_args = OrderArgs(token_id=market["token_up"], price=0.01, size=5.0, side="BUY")
options = PartialCreateOrderOptions(tick_size=str(market["tick_size"]), neg_risk=market["neg_risk"])

try:
    resp = client.create_and_post_order(order_args, options)
    print(f"Order response: {resp}")
    if telegram_enabled():
        body = (
            f"Direction:            UP\n"
            f"Price:                $0.01\n"
            f"Shares:               5\n"
            f"Market:               {market['slug']}\n"
            f"Order ID:             {resp.get('orderID', 'N/A')}\n"
            f"Status:               {resp.get('status', 'N/A')}"
        )
        send_telegram_notification("Order Placed", body)
        print("Telegram sent: Order Placed")
except Exception as exc:
    print(f"Order failed: {exc}")
    if telegram_enabled():
        send_telegram_notification("Order FAILED", str(exc))
        print("Telegram sent: Order FAILED")
