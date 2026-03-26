"""
Trade/auth.py
Polymarket CLOB authentication — L1 (private key) and L2 (API credentials).

Uses the official py-clob-client SDK.
API credentials are derived once and cached to disk so they survive restarts.
"""

import json
import os

from dotenv import load_dotenv
from py_clob_client.client import ClobClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet

POLY_PRIVATE_KEY = os.getenv("POLY_PRIVATE_KEY", "").strip()
POLY_FUNDER_ADDRESS = os.getenv("POLY_FUNDER_ADDRESS", "").strip()
POLY_SIGNATURE_TYPE = int(os.getenv("POLY_SIGNATURE_TYPE", "2"))

_CREDS_FILE = os.path.join(_ROOT, "Trade", ".poly_creds.json")


def _save_creds(creds) -> None:
    """Persist API credentials to a local JSON file."""
    data = {
        "apiKey": creds.api_key,
        "secret": creds.api_secret,
        "passphrase": creds.api_passphrase,
    }
    with open(_CREDS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_creds() -> dict | None:
    """Load cached API credentials if they exist."""
    if not os.path.isfile(_CREDS_FILE):
        return None
    with open(_CREDS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if all(k in data for k in ("apiKey", "secret", "passphrase")):
        return data
    return None


def _validate_env() -> None:
    """Ensure required environment variables are set."""
    if not POLY_PRIVATE_KEY:
        raise ValueError("POLY_PRIVATE_KEY is not set in .env")
    if not POLY_FUNDER_ADDRESS:
        raise ValueError("POLY_FUNDER_ADDRESS is not set in .env")


def derive_api_credentials(force: bool = False) -> dict:
    """
    Derive (or load cached) Polymarket API credentials via L1 auth.

    Returns dict with keys: apiKey, secret, passphrase.
    Set force=True to re-derive even if cached credentials exist.
    """
    _validate_env()

    if not force:
        cached = _load_creds()
        if cached:
            return cached

    # L1-only client — no API creds yet
    client = ClobClient(
        CLOB_HOST,
        key=POLY_PRIVATE_KEY,
        chain_id=CHAIN_ID,
        funder=POLY_FUNDER_ADDRESS,
        signature_type=POLY_SIGNATURE_TYPE,
    )

    creds = client.create_or_derive_api_creds()
    _save_creds(creds)
    return _load_creds()


def get_clob_client() -> ClobClient:
    """
    Return a fully authenticated ClobClient (L2) ready for trading.
    Derives or loads API credentials automatically.
    """
    _validate_env()
    creds = derive_api_credentials()

    from py_clob_client.clob_types import ApiCreds
    api_creds = ApiCreds(
        api_key=creds["apiKey"],
        api_secret=creds["secret"],
        api_passphrase=creds["passphrase"],
    )
    client = ClobClient(
        CLOB_HOST,
        key=POLY_PRIVATE_KEY,
        chain_id=CHAIN_ID,
        creds=api_creds,
        funder=POLY_FUNDER_ADDRESS,
        signature_type=POLY_SIGNATURE_TYPE,
    )
    return client


if __name__ == "__main__":
    print("Deriving Polymarket API credentials...")
    credentials = derive_api_credentials(force=True)
    print(f"  apiKey:     {credentials['apiKey']}")
    print(f"  secret:     {credentials['secret'][:8]}...")
    print(f"  passphrase: {credentials['passphrase'][:8]}...")
    print("Credentials cached to Trade/.poly_creds.json")

    print("\nTesting L2 client...")
    clob = get_clob_client()
    print(f"  Client OK — host: {CLOB_HOST}")
