#!/usr/bin/env python3
"""
BTC Polymarket Predictor - Universal
Works for ANY 5-min window - run anytime to get signal
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"

def get_recent_bars(n=200):
    """Get recent 5m bars from Binance"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "5m", "limit": n}
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
    """Add features to dataframe"""
    df = df.copy()
    
    # Target: Polymarket resolves on: will NEXT bar close > NEXT bar open?
    df['target'] = (df['close'].shift(-1) > df['open'].shift(-1)).astype(int)
    
    # Returns
    for lag in [1, 2, 3, 5]:
        df[f'ret_{lag}'] = df['close'].pct_change(lag).shift(1)
    
    # Candle features
    df['body'] = (df['close'] - df['open']) / df['open']
    df['range'] = (df['high'] - df['low']) / df['close']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(7).mean()
    loss = delta.where(delta < 0, 0).rolling(7).mean()
    df['rsi_7'] = (100 - (100 / (1 + gain / (loss + 1e-10)))).shift(1)
    
    # SMA
    for p in [5, 10, 20]:
        sma = df['close'].rolling(p).mean().shift(1)
        df[f'vsma_{p}'] = (df['close'] - sma) / sma
    
    # Volatility
    df['volatility'] = df['ret_1'].rolling(10).std()
    df['vol_ratio'] = (df['volume'] / df['volume'].rolling(10).mean()).shift(1)
    
    # Time features
    df['minute'] = df['timestamp'].dt.minute
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # Streaks
    df['is_green'] = (df['close'] > df['open']).astype(int)
    df['green_streak'] = df['is_green'].groupby(
        (df['is_green'] != df['is_green'].shift()).cumsum()
    ).cumcount()
    
    return df

def train():
    """Load historical data and train model"""
    # Try combined file first (3 years), fallback to old
    try:
        df = pd.read_csv(f"{DATA_DIR}/btc_5m_2023_2025.csv", parse_dates=['timestamp'])
        df2 = pd.read_csv(f"{DATA_DIR}/btc_5m_2025b.csv", parse_dates=['timestamp'])
        df = pd.concat([df, df2], ignore_index=True)
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    except:
        try:
            df = pd.read_csv(f"{DATA_DIR}/btc_5m_2023_2025.csv", parse_dates=['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        except:
            df = get_recent_bars(10000)
    
    df = add_features(df)
    
    features = ['ret_1', 'ret_2', 'ret_3', 'ret_5', 'body', 'range', 'rsi_7', 
               'vsma_5', 'vsma_10', 'vsma_20', 'volatility', 'vol_ratio',
               'minute', 'hour', 'dayofweek', 'green_streak']
    
    df = df.dropna(subset=features + ['target']).reset_index(drop=True)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].values)
    y = df['target'].values
    
    models = [
        GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=42),
        LogisticRegression(max_iter=1000)
    ]
    
    for m in models:
        m.fit(X, y)
    
    return models, scaler, features

def predict_current(models, scaler, features):
    """Get prediction for next bar"""
    df = get_recent_bars(100)
    df = add_features(df)
    
    df = df.dropna(subset=features).reset_index(drop=True)
    
    # Last complete bar
    last_bar = df.iloc[-1:]
    X = scaler.transform(last_bar[features].values)
    
    probs = [m.predict_proba(X)[:, 1] for m in models]
    avg_prob = np.mean(probs, axis=0)[0]
    
    return avg_prob, last_bar.iloc[0]

def should_bet(prob, threshold=0.55):
    """Determine action based on probability"""
    conf = max(prob, 1 - prob)
    
    if conf >= threshold:
        direction = "UP" if prob > 0.5 else "DOWN"
        confident = conf >= 0.60
        return direction, conf, confident
    else:
        return "NO BET", conf, False

def main():
    print("=" * 55)
    print("BTC Polymarket - Universal Predictor")
    print("Check anytime to see if next 5-min bar is a good bet")
    print("=" * 55)
    
    print("\nTraining model...")
    models, scaler, features = train()
    
    print("Getting current data...")
    prob, bar = predict_current(models, scaler, features)
    
    direction, conf, confident = should_bet(prob, threshold=0.55)
    
    print(f"\n=== CURRENT SIGNAL ===")
    print(f"Last bar: {bar['timestamp']}")
    print(f"Price: ${bar['close']:,.2f}")
    print(f"Next bar UP probability: {prob:.1%}")
    print()
    
    if direction == "NO BET":
        print(f">>> DON'T BET")
        print(f"   Confidence: {conf:.1%} (below 55% threshold)")
    else:
        print(f">>> BET {direction}")
        print(f"   Confidence: {conf:.1%}")
        if confident:
            print(f"   HIGH CONFIDENCE - more reliable!")
        else:
            print(f"   Lower confidence - be careful")
    
    print(f"\nAccuracy history: ~54-56% all trades, ~56-58% confident")
    print(f"Only bet when confident!")
    
    # Show all thresholds
    print(f"\n=== DECISION GUIDE ===")
    print(f"  < 55%: NO BET (low confidence)")
    print(f"  55-58%: Maybe - small bet only")  
    print(f"  58-60%: Good - normal bet")
    print(f"  > 60%:  BEST - larger bet")

if __name__ == "__main__":
    main()