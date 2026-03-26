"""
Trade/order.py
Place limit orders on Polymarket BTC Up/Down 5-minute markets.
"""

from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions

from Trade.auth import get_clob_client
from Trade.market import get_btc_5m_market


# Defaults (can be overridden per call)
DEFAULT_PRICE = 0.50   # 50 cents per share
DEFAULT_SIZE = 5       # 5 shares

# Cached client — created once, reused across orders
_cached_client = None


def _get_client():
    """Return a cached ClobClient, creating it on first call."""
    global _cached_client
    if _cached_client is None:
        _cached_client = get_clob_client()
    return _cached_client


def _reset_client():
    """Force re-creation of the CLOB client (e.g. after auth failure)."""
    global _cached_client
    _cached_client = None


def warm_client():
    """Pre-create the CLOB client at startup so orders are instant."""
    _get_client()


def place_btc_5m_order(
    direction: str,
    price: float = DEFAULT_PRICE,
    size: float = DEFAULT_SIZE,
    boundary_epoch: int | None = None,
) -> dict:
    """
    Place a limit BUY order on the BTC 5-min Up/Down market.

    Parameters
    ----------
    direction : str
        "UP" or "DOWN" — which outcome to buy.
    price : float
        Limit price per share (0.01–0.99).
    size : float
        Number of shares to buy.
    boundary_epoch : int or None
        Unix epoch of the 5-min window start. If None, uses system time.

    Returns
    -------
    dict with keys: success, orderID, status, etc. from the CLOB API.
    """
    direction = direction.upper()
    if direction not in ("UP", "DOWN"):
        raise ValueError(f"direction must be 'UP' or 'DOWN', got '{direction}'")

    market = get_btc_5m_market(boundary_epoch=boundary_epoch)

    token_id = market["token_up"] if direction == "UP" else market["token_down"]
    tick_size = str(market["tick_size"])
    neg_risk = market["neg_risk"]

    order_args = OrderArgs(
        token_id=token_id,
        price=price,
        size=size,
        side="BUY",
    )

    options = PartialCreateOrderOptions(
        tick_size=tick_size,
        neg_risk=neg_risk,
    )

    # Try with cached client; on auth failure, rebuild client and retry once
    try:
        client = _get_client()
        resp = client.create_and_post_order(order_args, options)
    except Exception as exc:
        if "401" in str(exc) or "403" in str(exc) or "invalid signature" in str(exc).lower():
            print("           [WARN] Auth error, re-creating CLOB client...")
            _reset_client()
            client = _get_client()
            resp = client.create_and_post_order(order_args, options)
        else:
            raise

    return {
        "response": resp,
        "market_slug": market["slug"],
        "market_title": market["title"],
        "direction": direction,
        "token_id": token_id,
        "price": price,
        "size": size,
    }


if __name__ == "__main__":
    print("This module places real orders. Import and call place_btc_5m_order().")
    print("Example: place_btc_5m_order('UP', price=0.50, size=5)")
