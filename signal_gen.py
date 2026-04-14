#!/usr/bin/env python3
"""
BTC Polymarket Signal Generator
- Real-time data fetching
- Signal generation with confidence thresholds
- Telegram-style CLI for checking signals
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
import json
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Model features
FEATURES = ['r1','r3','r5','r10','body','rng','rsi7','rsi14','vsma10','vsma30',
            'vol5','vol15','vr','macd','macdh','bbp','hour','dow','rl1','rl2','rl3']

def get_latest_data(minutes=60):
    """Get latest N minutes of data from Binance"""
    url = "https://api.binance.com/api/v3/klines"
    end = datetime.now()
    start = end - timedelta(minutes=minutes+10)
    
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000),
        "limit": 1000
    }
    
    data = requests.get(url, params=params).json()
    
    df = pd.DataFrame(data)
    df = df.iloc[:, :6]
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    
    return df

def create_features(df):
    """Create features from price data"""
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    for lag in [1, 3, 5, 10]:
        df[f'r{lag}'] = df['close'].pct_change(lag)
    
    df['body'] = (df['close'] - df['open']) / df['close']
    df['rng'] = (df['high'] - df['low']) / df['close']
    
    for p in [7, 14]:
        d = df['close'].diff()
        g = d.where(d>0,0).rolling(p).mean()
        l = (-d.where(d<0,0)).rolling(p).mean()
        df[f'rsi{p}'] = 100 - (100/(1 + g/(l+1e-10)))
    
    for p in [10, 30]:
        sma = df['close'].rolling(p).mean()
        df[f'vsma{p}'] = (df['close'] - sma) / sma
    
    df['vol5'] = df['r1'].rolling(5).std()
    df['vol15'] = df['r1'].rolling(15).std()
    
    df['vma'] = df['volume'].rolling(20).mean()
    df['vr'] = df['volume'] / (df['vma'] + 1e-10)
    
    e12 = df['close'].ewm(span=12, adjust=False).mean()
    e26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = e12 - e26
    df['macdh'] = df['macd'] - df['macd'].ewm(span=9, adjust=False).mean()
    
    bb = df['close'].rolling(20)
    df['bbp'] = (df['close'] - (bb.mean() - 2*bb.std())) / (4*bb.std() + 1e-10)
    
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    
    for lag in [1, 2, 3]:
        df[f'rl{lag}'] = df['r1'].shift(lag)
    
    return df

def load_historical_and_train():
    """Load historical data and train model"""
    # Try to load cached data
    cache_file = f"{DATA_DIR}/model_cache.pkl"
    
    # Check if we have recent data
    data_file = f"{DATA_DIR}/btc_1m_recent.csv"
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, parse_dates=['timestamp'])
        # Check if data is fresh (within 1 hour)
        if (datetime.now() - df['timestamp'].max()).total_seconds() > 3600:
            print("Cached data is stale, fetching new data...")
            os.remove(data_file)
            df = None
    else:
        df = None
    
    if df is None:
        print("Downloading 60 days of historical data...")
        df = get_latest_data(minutes=60*24*60)  # 60 days
        df.to_csv(data_file, index=False)
    
    print(f"Training on {len(df)} candles...")
    
    # Create features
    df = create_features(df)
    
    # Target: 5min direction
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    df = df.dropna(subset=FEATURES + ['target']).copy()
    
    for f in FEATURES:
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES)
    
    X = df[FEATURES].values
    y = df['target'].values
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42, subsample=0.7)
    model.fit(X_s, y)
    
    print("Model trained!")
    return model, scaler, df['close'].iloc[-1], df['timestamp'].iloc[-1]

def get_signal(model, scaler):
    """Get current signal"""
    # Get latest data
    df = get_latest_data(minutes=120)  # Get extra for features
    df = create_features(df)
    
    # Clean features
    latest = df[FEATURES].iloc[-1:].copy()
    for f in FEATURES:
        latest[f] = latest[f].replace([np.inf, -np.inf], np.nan)
    
    if latest.isna().any().any():
        print("Warning: Missing features")
        return None
    
    X = latest.values
    X_s = scaler.transform(X)
    
    pred = model.predict(X_s)[0]
    prob = model.predict_proba(X_s)[0]
    
    return {
        'time': df['timestamp'].iloc[-1],
        'price': df['close'].iloc[-1],
        'prediction': 'UP' if pred == 1 else 'DOWN',
        'confidence_up': prob[1],
        'confidence_down': prob[0]
    }

def print_signal(signal):
    """Pretty print signal"""
    if signal is None:
        print("No signal available")
        return
    
    print("\n" + "="*50)
    print(f"BTC PRICE PREDICTION SIGNAL")
    print("="*50)
    print(f"Time:    {signal['time']}")
    print(f"Price:   ${signal['price']:,.2f}")
    print(f"Pred:    {signal['prediction']}")
    print(f"Conf:    UP {signal['confidence_up']:.1%}  |  DOWN {signal['confidence_down']:.1%}")
    print("-"*50)
    
    # EV check
    THRESHOLD = 0.52  # Need >52% to beat Polymarket fees
    
    if signal['confidence_up'] >= THRESHOLD:
        bet = "UP"
        conf = signal['confidence_up']
        print(f">>> BET {bet} <<< (confidence {conf:.1%} >= {THRESHOLD:.0%})")
    elif signal['confidence_down'] >= THRESHOLD:
        bet = "DOWN"
        conf = signal['confidence_down']
        print(f">>> BET {bet} <<< (confidence {conf:.1%} >= {THRESHOLD:.0%})")
    else:
        print(f">>> NO BET <<< (confidence below {THRESHOLD:.0%} threshold)")
    
    print("="*50 + "\n")

def main():
    print("Loading model and generating signal...")
    model, scaler, last_price, last_time = load_historical_and_train()
    
    print(f"Model trained on data up to {last_time}")
    
    while True:
        signal = get_signal(model, scaler)
        print_signal(signal)
        
        print("Press Enter to refresh, 'q' to quit...")
        cmd = input()
        if cmd.lower() == 'q':
            break

if __name__ == "__main__":
    main()