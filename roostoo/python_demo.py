#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dotenv import load_dotenv
import os
import requests
import hashlib
import hmac
import time


load_dotenv()

API_KEY = os.getenv("API_KEY")
SECRET = os.getenv("SECRET")

BASE_URL = "https://mock-api.roostoo.com"
SESSION = requests.Session()


def generate_signature(params):
    query_string = '&'.join(
        ["{}={}".format(k, params[k]) for k in sorted(params.keys())]
    )
    us = SECRET.encode('utf-8')
    m = hmac.new(us, query_string.encode('utf-8'), hashlib.sha256)
    return m.hexdigest()


def _auth_headers():
    return {
        "RST-API-KEY": API_KEY,
    }


def _signed_headers(params):
    headers = _auth_headers()
    headers["MSG-SIGNATURE"] = generate_signature(params)
    return headers


def _print_response(r):
    try:
        print(r.status_code, r.text)
    except Exception:
        pass


def get_server_time():
    try:
        r = SESSION.get(
            BASE_URL + "/v3/serverTime",
            timeout=10,
        )
        _print_response(r)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print("get_server_time error:", e)
        return None


def get_ex_info():
    try:
        r = SESSION.get(
            BASE_URL + "/v3/exchangeInfo",
            timeout=10,
        )
        _print_response(r)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print("get_ex_info error:", e)
        return None


def get_ticker(pair=None):
    payload = {
        "timestamp": int(time.time() * 1000),
    }
    if pair:
        payload["pair"] = pair

    try:
        r = SESSION.get(
            BASE_URL + "/v3/ticker",
            params=payload,
            timeout=10,
        )
        _print_response(r)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print("get_ticker error:", e)
        if getattr(e, "response", None) is not None:
            print("Response text:", e.response.text)
        return None


def get_balance():
    payload = {
        "timestamp": int(time.time()) * 1000,
    }

    try:
        r = SESSION.get(
            BASE_URL + "/v3/balance",
            params=payload,
            headers=_signed_headers(payload),
            timeout=10,
        )
        _print_response(r)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print("get_balance error:", e)
        if getattr(e, "response", None) is not None:
            print("Response text:", e.response.text)
        return None


def place_order(coin, side, qty, price=None):
    payload = {
        "timestamp": int(time.time()) * 1000,
        "pair": coin + "/USD",
        "side": side,
        "quantity": qty,
    }

    if price is None:
        payload["type"] = "MARKET"
    else:
        payload["type"] = "LIMIT"
        payload["price"] = price

    try:
        r = SESSION.post(
            BASE_URL + "/v3/place_order",
            data=payload,
            headers=_signed_headers(payload),
            timeout=10,
        )
        _print_response(r)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print("place_order error:", e)
        if getattr(e, "response", None) is not None:
            print("Response text:", e.response.text)
        return None


def cancel_order(order_id=None, pair="BTC/USD"):
    payload = {
        "timestamp": int(time.time()) * 1000,
    }

    if order_id is not None:
        payload["order_id"] = order_id
    elif pair is not None:
        payload["pair"] = pair

    try:
        r = SESSION.post(
            BASE_URL + "/v3/cancel_order",
            data=payload,
            headers=_signed_headers(payload),
            timeout=10,
        )
        _print_response(r)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print("cancel_order error:", e)
        if getattr(e, "response", None) is not None:
            print("Response text:", e.response.text)
        return None


def query_order(order_id=None, pair=None, pending_only=None):
    payload = {
        "timestamp": int(time.time()) * 1000,
    }

    if order_id is not None:
        payload["order_id"] = order_id
    if pair is not None:
        payload["pair"] = pair
    if pending_only is not None:
        payload["pending_only"] = pending_only

    try:
        r = SESSION.post(
            BASE_URL + "/v3/query_order",
            data=payload,
            headers=_signed_headers(payload),
            timeout=10,
        )
        _print_response(r)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print("query_order error:", e)
        if getattr(e, "response", None) is not None:
            print("Response text:", e.response.text)
        return None


def pending_count():
    payload = {
        "timestamp": int(time.time()) * 1000,
    }

    try:
        r = SESSION.get(
            BASE_URL + "/v3/pending_count",
            params=payload,
            headers=_signed_headers(payload),
            timeout=10,
        )
        _print_response(r)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print("pending_count error:", e)
        if getattr(e, "response", None) is not None:
            print("Response text:", e.response.text)
        return None


if __name__ == '__main__':
    # print(get_server_time())
    # print(get_ex_info())
    print(get_ticker())
    # print(get_balance())
    # print(place_order("BNB", "BUY", 200000))
    # print(cancel_order())
    # print(query_order())
    # print(pending_count())