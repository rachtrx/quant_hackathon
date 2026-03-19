#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import hashlib
import hmac
import time
import python_demo
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

# Add parent directory to path to import features and data_retrieval
sys.path.append('..')
from features import add_features
from data_retrieval import fetch_recent_klines_rest

# Strategy parameters
PAIR = "BTC/USD"
COIN = "BTC"
TIMEFRAME_MIN = 30
VOL_FILTER_THRESH = 1.1
BASE_SIZE_PCT = 0.005  # 0.5%
SL_DEV = -2.5
TP_DEV = 0.0  # mean
WINDOW = 1  # for rolling mean/std (start after first 30min candle)
TARGET_HORIZON = 5  # since models are for 5min

MODEL_DIR = "../models/rf"
DATA_DIR = "../user_data/data"
INTERVAL = "1m"
TARGET_HORIZON = 5

if __name__ == '__main__':
    # Load model and feature columns
    model_path = os.path.join(MODEL_DIR, "BTCUSDT__h5_model.joblib")
    features_path = os.path.join(MODEL_DIR, "BTCUSDT__h5_feature_cols.json")
    
    if not os.path.exists(model_path):
        print("Model file not found. Please ensure models are trained.")
        exit(1)
    
    model = joblib.load(model_path)
    
    with open(features_path, "r") as f:
        feature_cols = json.load(f)
    
    # Initialize dataframe with historical data
    data_path = os.path.join(DATA_DIR, "BTCUSDT_1m.parquet")
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path).tail(2000).reset_index(drop=True)
    else:
        df = pd.DataFrame()  # Start empty if no data
    
    while True:
        try:
            # Fetch recent data
            print("Fetching recent data...")
            recent_df = fetch_recent_klines_rest("BTCUSDT", INTERVAL, limit=100)  # Fetch last 100 1m bars
            if not recent_df.empty:
                # Append to df, drop duplicates by open_time
                df = pd.concat([df, recent_df]).drop_duplicates(subset=['open_time']).tail(2000).reset_index(drop=True)
            
            if df.empty or len(df) < 200:
                print("Not enough data, waiting...")
                time.sleep(60)
                continue
            
            # Add features
            df_features, _ = add_features(df, TARGET_HORIZON)
            
            # Resample to 30min for mean/std calculation
            df_30 = df.set_index('open_time').resample('30min').agg({'close': 'last'}).dropna()
            df_30['mean'] = df_30['close'].rolling(WINDOW).mean()
            df_30['std'] = df_30['close'].rolling(WINDOW).std()
            
            # Ensure we have enough data
            if len(df_30) < WINDOW:
                print("Not enough 30min data, waiting...")
                time.sleep(60)
                continue
            
            # Current values
            current_price = df_30['close'].iloc[-1]
            current_mean = df_30['mean'].iloc[-1]
            current_std = df_30['std'].iloc[-1]
            dev = (current_price - current_mean) / current_std
            
            print(f"Current price: {current_price}, Mean: {current_mean}, Std: {current_std}, Dev: {dev}")
            
            # Check for signal
            if dev <= -2:
                # Volatility filter
                vol_5 = df['close'].pct_change().rolling(5).std().iloc[-1]
                vol_30 = df_30['close'].pct_change().rolling(30).std().iloc[-1]
                vol_ratio = vol_5 / vol_30
                
                print(f"Vol ratio: {vol_ratio}")
                
                if vol_ratio < VOL_FILTER_THRESH:
                    # Get prediction
                    latest_features = df_features.iloc[-1][feature_cols]
                    pred = model.predict(latest_features.values.reshape(1, -1))[0]
                    
                    print(f"Prediction: {pred}")
                    
                    if pred > 0:
                        # Predicted price deviation
                        pred_price = current_price * (1 + pred)
                        pred_dev = (pred_price - current_mean) / current_std
                        
                        print(f"Pred dev: {pred_dev}")
                        
                        # Determine position size
                        if -2 <= pred_dev < -1:
                            size_pct = BASE_SIZE_PCT
                        elif -1 <= pred_dev < 0:
                            size_pct = 0.01
                        elif pred_dev >= 0:
                            size_pct = 0.02
                        else:
                            size_pct = 0
                        
                        if size_pct > 0:
                            # Get balance
                            balance = python_demo.get_balance()
                            wallet = balance.get('Wallet') or balance.get('SpotWallet', {})
                            usd_balance = wallet.get('USD', {}).get('Free', 0)
                            
                            if usd_balance > 0:
                                # Apply market order fee (0.1%) to buy
                                qty = (usd_balance * size_pct * (1 - 0.001)) / current_price

                                # Ensure minimum order value (MiniOrder = 1.0 for BTC/USD)
                                min_order_value = 1.0
                                if qty * current_price <= min_order_value:
                                    qty = (min_order_value + 0.1) / current_price  # Slightly above minimum

                                print(f"Placing BUY order: {qty} {COIN} at market price (fee 0.1%)")
                                buy_resp = python_demo.place_order(COIN, "BUY", qty)
                                
                                # Place take profit order
                                tp_price = current_mean
                                # Ensure TP order meets minimum
                                if qty * tp_price <= min_order_value:
                                    qty = (min_order_value + 0.1) / tp_price  # Adjust qty if needed

                                print(f"Placing TP order: SELL {qty} {COIN} at {tp_price} (fee 0.05%)")
                                tp_resp = python_demo.place_order(COIN, "SELL", qty, tp_price)

                                # Start monitoring
                                sl_price = current_mean + SL_DEV * current_std
                                position = {
                                    'qty': qty,
                                    'entry_price': current_price,
                                    'sl_price': sl_price,
                                    'tp_price': tp_price,
                                    'tp_order_id': tp_resp.get('order_id')  # assuming API returns this
                                }
                                monitor_position(position)
                            else:
                                print("No USD balance available")
                        else:
                            print("Prediction deviation too low")
                    else:
                        print("Prediction not positive")
                else:
                    print("Volatility filter not met")
            else:
                print("No buy signal: deviation not <= -2")
            
            # Save updated df to disk
            df.to_parquet(os.path.join(DATA_DIR, "BTCUSDT_1m_live.parquet"), index=False)
            
            # Sleep before next cycle
            time.sleep(60)  # Check every minute
        
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)


def monitor_position(position):
    import time
    print(f"Starting position monitoring. SL: {position['sl_price']}, TP: {position['tp_price']}")
    
    while True:
        # Get current price
        ticker = python_demo.get_ticker(PAIR)
        current_price = ticker.get('Data', {}).get(PAIR, {}).get('LastPrice', 0)
        
        if current_price == 0:
            print("Failed to get current price")
            time.sleep(60)
            continue
        
        print(f"Monitoring: Current price {current_price}")
        
        # Check stop loss
        if current_price <= position['sl_price']:
            print(f"Stop loss triggered at {current_price}")
            python_demo.place_order(COIN, "SELL", position['qty'])
            break
        
        # Check if TP order is filled
        if position.get('tp_order_id'):
            # Query TP order status
            # Assuming query_order can take order_id
            # For mock, perhaps modify python_demo or assume
            # For simplicity, keep price check
            pass
        if current_price >= position['tp_price']:
            print(f"Take profit order likely filled at {current_price}")
            break
        
        # Sleep for 1 minute
        time.sleep(60)
    
    print("Position closed")
    
    while True:
        # Get current price
        ticker = python_demo.get_ticker(PAIR)
        current_price = float(ticker.get('last', ticker.get('price', 0)))  # adjust based on API response
        
        print(f"Current price: {current_price}")
        
        # Check if TP order is filled (in real implementation, query order status)
        # For mock, simulate: if current_price >= tp_price, assume filled
        if current_price >= position['tp_price']:
            print("TP hit, position closed")
            break
        
        # Check stop loss
        if current_price <= position['sl_price']:
            print("SL hit, selling position")
            # Cancel TP order if still open
            if position['tp_order_id']:
                python_demo.cancel_order(order_id=position['tp_order_id'], pair=PAIR)
            # Place market sell
            python_demo.place_order(COIN, "SELL", position['qty'])
            break
        
        # Sleep before next check
        time.sleep(60)  # Check every minute
