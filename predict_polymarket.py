#!/usr/bin/env python3
"""
BTC Polymarket Predictor - Production Ready
Predicts UP/DOWN for next 5-min bar
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"

def get_latest_data():
    """Get recent 5m data from Binance"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "5m", "limit": 500}
    
    data = requests.get(url, params=params).json()
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def add_features(df):
    """Add features for prediction"""
    df = df.copy()
    
    for lag in [1, 2, 3, 5]:
        df[f'ret_{lag}'] = df['close'].pct_change(lag).shift(1)
    
    df['body'] = (df['close'] - df['open']) / df['open']
    df['range'] = (df['high'] - df['low']) / df['close']
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(7).mean()
    loss = delta.where(delta < 0, 0).rolling(7).mean()
    df['rsi_7'] = (100 - (100 / (1 + gain / (loss + 1e-10)))).shift(1)
    
    for p in [5, 10, 20]:
        sma = df['close'].rolling(p).mean().shift(1)
        df[f'vsma_{p}'] = (df['close'] - sma) / sma
    
    df['volatility'] = df['ret_1'].rolling(10).std()
    
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    return df

def train_model(df):
    """Train model on available data"""
    features = ['ret_1', 'ret_2', 'ret_3', 'ret_5', 'body', 'range', 'rsi_7', 
                'vsma_5', 'vsma_10', 'vsma_20', 'volatility']
    
    df = df.dropna(subset=features).reset_index(drop=True)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].values)
    
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna(subset=['target']).reset_index(drop=True)
    X = scaler.transform(df[features].values)
    
    model = GradientBoostingClassifier(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.7,
        random_state=42
    )
    model.fit(X, df['target'].values)
    
    return model, scaler, features

def predict_current(model, scaler, features, df):
    """Generate prediction for current bar"""
    df = df.dropna(subset=features).reset_index(drop=True)
    
    last_row = df.iloc[-1:]
    X = scaler.transform(last_row[features].values)
    
    prob = model.predict_proba(X)[0]
    pred = "UP" if prob[1] > 0.5 else "DOWN"
    
    return pred, prob

def main():
    print("=" * 50)
    print("BTC Polymarket Predictor")
    print("=" * 50)
    
    # Try load local data first, else fetch
    local_file = f"{DATA_DIR}/btc_5m_2023_2025.csv"
    try:
        df = pd.read_csv(local_file, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        print(f"Loaded local data: {len(df)} bars")
    except:
        print("Fetching fresh data...")
        df = get_latest_data()
    
    # Add features
    df = add_features(df)
    
    # Train model
    print("Training model...")
    model, scaler, features = train_model(df)
    
    # Get fresh data for prediction
    print("Fetching latest data...")
    fresh = get_latest_data()
    fresh = add_features(fresh)
    
    # Predict
    pred, prob = predict_current(model, scaler, features, fresh)
    
    print(f"\n=== SIGNAL ===")
    print(f"Time: {fresh['timestamp'].iloc[-1]}")
    print(f"Price: ${fresh['close'].iloc[-1]:,.2f}")
    print(f"Prediction: {pred}")
    print(f"Confidence: UP={prob[1]:.1%}, DOWN={prob[0]:.1%}")
    
    # Should we bet?
    if prob[1] > 0.58:
        print(f"\n>>> BET UP (confidence {prob[1]:.1%})")
    elif prob[0] > 0.58:
        print(f"\n>>> BET DOWN (confidence {prob[0]:.1%})")
    else:
        print(f"\n>>> NO BET (low confidence)")

if __name__ == "__main__":
    main()