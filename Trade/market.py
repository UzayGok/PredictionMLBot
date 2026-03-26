"""
Trade/market.py
Look up the current BTC Up/Down 5-minute market on Polymarket
and return the token IDs for the UP and DOWN outcomes.
"""

import datetime
import requests

GAMMA_API = "https://gamma-api.polymarket.com"


def _current_boundary_epoch() -> int:
    """Unix timestamp of the current 5-minute boundary (floored)."""
    now = datetime.datetime.now(datetime.timezone.utc)
    epoch = int(now.timestamp())
    return (epoch // 300) * 300


def get_btc_5m_market(boundary_epoch: int | None = None) -> dict:
    """
    Fetch the Polymarket BTC Up/Down 5-min event for the given boundary.

    If boundary_epoch is None, uses the *next* 5-minute boundary
    (the candle the bot is predicting).

    Returns dict with keys:
        slug, title, condition_id, neg_risk, tick_size,
        token_up, token_down, end_date
    """
    if boundary_epoch is None:
        boundary_epoch = _current_boundary_epoch()  # fallback; prefer explicit epoch

    slug = f"btc-updown-5m-{boundary_epoch}"

    resp = requests.get(
        f"{GAMMA_API}/events",
        params={"slug": slug},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    if not data:
        raise LookupError(f"No Polymarket event found for slug: {slug}")

    event = data[0]
    market = event["markets"][0]

    outcomes = market["outcomes"]  # e.g. '["Up", "Down"]'
    if isinstance(outcomes, str):
        import json
        outcomes = json.loads(outcomes)

    token_ids = market["clobTokenIds"]
    if isinstance(token_ids, str):
        import json
        token_ids = json.loads(token_ids)

    # Map outcome names to token IDs
    outcome_map = dict(zip(outcomes, token_ids))

    return {
        "slug": slug,
        "title": event["title"],
        "condition_id": market["conditionId"],
        "neg_risk": event.get("negRisk", False),
        "tick_size": market.get("orderPriceMinTickSize", "0.01"),
        "token_up": outcome_map["Up"],
        "token_down": outcome_map["Down"],
        "end_date": market["endDate"],
    }


if __name__ == "__main__":
    info = get_btc_5m_market()
    for k, v in info.items():
        print(f"  {k}: {v}")
