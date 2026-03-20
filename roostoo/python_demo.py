#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dotenv import load_dotenv
import os
import time
import hmac
import hashlib
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, Any

import requests

load_dotenv()

API_KEY = os.getenv("API_KEY")
SECRET = os.getenv("SECRET")

BASE_URL = "https://mock-api.roostoo.com"
SESSION = requests.Session()


# =========================================================
# BASIC HELPERS
# =========================================================
def _ts_ms() -> str:
    return str(int(time.time() * 1000))


def _sign_payload(payload: dict) -> str:
    query_string = "&".join(f"{k}={payload[k]}" for k in sorted(payload.keys()))
    return hmac.new(
        SECRET.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _signed_headers(payload: dict, is_post: bool = False) -> dict:
    headers = {
        "RST-API-KEY": API_KEY,
        "MSG-SIGNATURE": _sign_payload(payload),
    }
    if is_post:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    return headers

def _post_form(url: str, payload: dict) -> Optional[dict]:
    headers = _signed_headers(payload, is_post=True)
    total_params = "&".join(f"{k}={payload[k]}" for k in sorted(payload.keys()))
    try:
        resp = SESSION.post(url, headers=headers, data=total_params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"POST error: {e}")
        if e.response is not None:
            print(f"Response text: {e.response.text}")
        return None


def _get_signed(url: str, payload: dict) -> Optional[dict]:
    headers = _signed_headers(payload, is_post=False)
    try:
        resp = SESSION.get(url, headers=headers, params=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"GET error: {e}")
        if e.response is not None:
            print(f"Response text: {e.response.text}")
        return None


def _normalize_pair(pair_or_coin: str) -> str:
    return pair_or_coin if "/" in pair_or_coin else f"{pair_or_coin}/USD"


def _round_down(value: float, decimals: int) -> float:
    q = Decimal("1").scaleb(-decimals)  # e.g. decimals=2 -> Decimal('0.01')
    return float(Decimal(str(value)).quantize(q, rounding=ROUND_DOWN))


# =========================================================
# PUBLIC ENDPOINTS
# =========================================================
def get_server_time() -> Optional[dict]:
    try:
        resp = SESSION.get(f"{BASE_URL}/v3/serverTime", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"Error getting server time: {e}")
        return None


def get_ex_info() -> Optional[dict]:
    try:
        resp = SESSION.get(f"{BASE_URL}/v3/exchangeInfo", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"Error getting exchange info: {e}")
        return None


def get_ticker(pair: Optional[str] = None) -> Optional[dict]:
    payload = {"timestamp": _ts_ms()}
    if pair:
        payload["pair"] = _normalize_pair(pair)

    try:
        resp = SESSION.get(f"{BASE_URL}/v3/ticker", params=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"Error getting ticker: {e}")
        if e.response is not None:
            print(f"Response text: {e.response.text}")
        return None


# =========================================================
# SIGNED ENDPOINTS
# =========================================================
def get_balance() -> Optional[dict]:
    payload = {"timestamp": _ts_ms()}
    return _get_signed(f"{BASE_URL}/v3/balance", payload)


def pending_count() -> Optional[dict]:
    payload = {"timestamp": _ts_ms()}
    return _get_signed(f"{BASE_URL}/v3/pending_count", payload)


def query_order(
    order_id: Optional[str] = None,
    pair: Optional[str] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    pending_only: Optional[bool] = None,
) -> Optional[dict]:
    payload = {"timestamp": _ts_ms()}

    if order_id is not None:
        payload["order_id"] = str(order_id)
    else:
        if pair is not None:
            payload["pair"] = _normalize_pair(pair)
        if offset is not None:
            payload["offset"] = str(offset)
        if limit is not None:
            payload["limit"] = str(limit)
        if pending_only is not None:
            payload["pending_only"] = "TRUE" if pending_only else "FALSE"

    return _post_form(f"{BASE_URL}/v3/query_order", payload)


def cancel_order(order_id: Optional[str] = None, pair: Optional[str] = None) -> Optional[dict]:
    payload = {"timestamp": _ts_ms()}

    if order_id is not None and pair is not None:
        raise ValueError("Only one of order_id or pair may be sent.")

    if order_id is not None:
        payload["order_id"] = str(order_id)
    elif pair is not None:
        payload["pair"] = _normalize_pair(pair)

    return _post_form(f"{BASE_URL}/v3/cancel_order", payload)


def _get_pair_rules(pair: str) -> Dict[str, Any]:
    info = get_ex_info()
    if not info:
        raise RuntimeError("Failed to fetch exchangeInfo")

    trade_pairs = info.get("TradePairs", {})
    if pair not in trade_pairs:
        raise ValueError(f"Pair not found in exchange info: {pair}")

    return trade_pairs[pair]


def place_order(
    pair_or_coin: str,
    side: str,
    quantity: float,
    price: Optional[float] = None,
    order_type: Optional[str] = None,
) -> Optional[dict]:
    pair = _normalize_pair(pair_or_coin)
    side = side.upper()

    if order_type is None:
        order_type = "LIMIT" if price is not None else "MARKET"
    order_type = order_type.upper()

    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    if order_type not in {"MARKET", "LIMIT"}:
        raise ValueError("order_type must be MARKET or LIMIT")
    if order_type == "LIMIT" and price is None:
        raise ValueError("LIMIT order requires price")

    rules = _get_pair_rules(pair)
    amount_precision = int(rules["AmountPrecision"])
    price_precision = int(rules["PricePrecision"])
    min_order = float(rules["MiniOrder"])

    quantity = _round_down(quantity, amount_precision)
    if quantity <= 0:
        raise ValueError(f"Quantity rounded to zero for {pair}")

    payload = {
        "timestamp": _ts_ms(),
        "pair": pair,
        "side": side,
        "type": order_type,
        "quantity": str(quantity),
    }

    if order_type == "LIMIT":
        price = _round_down(float(price), price_precision)
        if price <= 0:
            raise ValueError(f"Price rounded to zero for {pair}")
        payload["price"] = str(price)
        order_value = quantity * price
    else:
        ticker = get_ticker(pair)
        if not ticker or not ticker.get("Success"):
            raise RuntimeError(f"Could not get ticker for {pair}")
        last_price = float(ticker["Data"][pair]["LastPrice"])
        order_value = quantity * last_price

    if order_value < min_order:
        raise ValueError(
            f"Order value {order_value:.8f} below MiniOrder {min_order} for {pair}"
        )

    return _post_form(f"{BASE_URL}/v3/place_order", payload)


def close_all_positions(coins=None, min_value_usd: float = 1.0):
    if coins is None:
        coins = ["BTC", "ETH", "BNB"]

    print("=== CLOSING ALL POSITIONS ===")

    # 1) cancel all pending orders
    try:
        pending = query_order(pending_only=True)
        orders = pending.get("OrderMatched", []) if pending and pending.get("Success") else []

        for order in orders:
            if order.get("Status") == "PENDING":
                oid = order.get("OrderID")
                resp = cancel_order(order_id=oid)
                print(f"Cancel {oid}: {resp}")
    except Exception as e:
        print(f"Cancel orders error: {e}")

    # 2) market sell balances
    try:
        bal = get_balance()
        if not bal or not bal.get("Success"):
            print("Could not fetch balance.")
            return

        wallet = bal.get("Wallet", {})
        for coin in coins:
            qty = float(wallet.get(coin, {}).get("Free", 0.0))
            if qty <= 0:
                continue

            pair = f"{coin}/USD"
            ticker = get_ticker(pair)
            if not ticker or not ticker.get("Success"):
                continue

            price = float(ticker["Data"][pair]["LastPrice"])
            if qty * price < min_value_usd:
                continue

            print(f"Selling {qty} {coin}")
            resp = place_order(pair, "SELL", qty, order_type="MARKET")
            print(resp)
    except Exception as e:
        print(f"Sell error: {e}")

    print("=== DONE ===")


if __name__ == "__main__":
    print(get_server_time())
    print(get_ex_info())
    print(get_ticker("BTC/USD"))
    print(get_balance())