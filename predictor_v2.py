#!/usr/bin/env python3
"""BTC Polymarket Predictor - Complete Working Version"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_binance_1m(days=90):
    """Download ~90 days of 1m data"""
    f = f"{DATA_DIR}/btc_1m_{days}d.csv"
    if os.path.exists(f):
        print(f"Loading {f}...")
        return pd.read_csv(f, parse_dates=['timestamp'])
    
    print(f"Downloading {days} days of 1m data...")
    url = "https://api.binance.com/api/v3/klines"
    data = []
    
    end = datetime.now()
    start = end - timedelta(days=days)
    start_ts = int(start.timestamp() * 1000)
    end_ts = int(end.timestamp() * 1000)
    
    while start_ts < end_ts:
        r = requests.get(url, params={"symbol": "BTCUSDT", "interval": "1m", "startTime": start_ts, "endTime": end_ts, "limit": 1000}).json()
        if not r: break
        data.extend(r)
        start_ts = r[-1][0] + 1
        print(f"  {len(data)} candles")
    
    df = pd.DataFrame(data)
    df = df.iloc[:, :6]
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    df.to_csv(f, index=False)
    print(f"Saved {len(df)} rows")
    return df

def create_features(df):
    """Create prediction features"""
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Returns
    for lag in [1, 3, 5, 10]:
        df[f'r{lag}'] = df['close'].pct_change(lag)
    
    # Candle
    df['body'] = (df['close'] - df['open']) / df['close']
    df['rng'] = (df['high'] - df['low']) / df['close']
    
    # RSI
    for p in [7, 14]:
        d = df['close'].diff()
        g = d.where(d>0,0).rolling(p).mean()
        l = (-d.where(d<0,0)).rolling(p).mean()
        df[f'rsi{p}'] = 100 - (100/(1 + g/(l+1e-10)))
    
    # Price vs SMA
    for p in [10, 30]:
        sma = df['close'].rolling(p).mean()
        df[f'vsma{p}'] = (df['close'] - sma) / sma
    
    # Volatility
    df['vol5'] = df['r1'].rolling(5).std()
    df['vol15'] = df['r1'].rolling(15).std()
    
    # Volume
    df['vma'] = df['volume'].rolling(20).mean()
    df['vr'] = df['volume'] / (df['vma'] + 1e-10)
    
    # MACD
    e12 = df['close'].ewm(span=12, adjust=False).mean()
    e26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = e12 - e26
    df['macdh'] = df['macd'] - df['macd'].ewm(span=9, adjust=False).mean()
    
    # BB position
    bb = df['close'].rolling(20)
    df['bbp'] = (df['close'] - (bb.mean() - 2*bb.std())) / (4*bb.std() + 1e-10)
    
    # Time
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    
    # Lags
    for lag in [1, 2, 3]:
        df[f'rl{lag}'] = df['r1'].shift(lag)
    
    return df

FEATURES = ['r1','r3','r5','r10','body','rng','rsi7','rsi14','vsma10','vsma30',
            'vol5','vol15','vr','macd','macdh','bbp','hour','dow','rl1','rl2','rl3']

def walk_forward(df, target_col, n_splits=8):
    """Walk-forward validation with proper train/test separation"""
    df = df.dropna(subset=FEATURES + [target_col]).copy()
    
    for f in FEATURES:
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES)
    df = df.reset_index(drop=True)
    
    n = len(df)
    train_size = int(n * 0.8)
    
    results = []
    
    for i in range(n_splits):
        # Sliding window: each test period is 20% of data
        test_start = train_size + (i * (n - train_size) // n_splits)
        test_end = min(test_start + (n - train_size) // n_splits, n)
        
        if test_end - test_start < 1000:
            continue
        
        train_df = df.iloc[:test_start]
        test_df = df.iloc[test_start:test_end]
        
        X_train = train_df[FEATURES].values
        y_train = train_df[target_col].values
        X_test = test_df[FEATURES].values
        y_test = test_df[target_col].values
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42, subsample=0.7)
        model.fit(X_train_s, y_train)
        
        acc = (model.predict(X_test_s) == y_test).mean()
        baseline = y_test.mean()
        
        results.append({
            'period': i+1,
            'test_start': test_start,
            'test_end': test_end,
            'accuracy': acc,
            'baseline': baseline,
            'n_test': len(test_df)
        })
        
        print(f"Period {i+1}: test samples {test_start}-{test_end}, accuracy: {acc:.4f}, baseline: {baseline:.4f}")
    
    return pd.DataFrame(results)

def train_and_signal(df, target_col):
    """Train on all data and generate current signal"""
    df = df.dropna(subset=FEATURES + [target_col]).copy()
    for f in FEATURES:
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES)
    
    X = df[FEATURES].values
    y = df[target_col].values
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42, subsample=0.7)
    model.fit(X_s, y)
    
    # Get last row features
    latest = df[FEATURES].iloc[-1:].values
    latest_s = scaler.transform(latest)
    
    pred = model.predict(latest_s)[0]
    prob = model.predict_proba(latest_s)[0]
    
    return pred, prob, df['close'].iloc[-1], df['timestamp'].iloc[-1]

def main():
    print("="*55)
    print("BTC Polymarket Predictor")
    print("="*55)
    
    # Download data
    df = download_binance_1m(days=60)
    print(f"\nData: {len(df)} candles")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
    
    # Create features
    print("Creating features...")
    df = create_features(df)
    
    # Target: predict 5min direction (Polymarket interval)
    target_col = 'target_5'
    df[target_col] = (df['close'].shift(-5) > df['close']).astype(int)
    
    # Walk-forward
    print("\n" + "="*55)
    print("WALK-FORWARD VALIDATION")
    print("="*55)
    print("Target: predict UP in next 5 minutes\n")
    
    results = walk_forward(df, target_col, n_splits=8)
    
    print("\n--- SUMMARY ---")
    print(f"Mean accuracy: {results['accuracy'].mean():.4f} ({results['accuracy'].mean()*100:.2f}%)")
    print(f"Std: {results['accuracy'].std():.4f}")
    print(f"Baseline (always UP): {results['baseline'].mean():.4f}")
    beats = (results['accuracy'] > results['baseline']).sum()
    print(f"Beats baseline: {beats}/{len(results)} periods")
    
    # Final model and signal
    print("\n" + "="*55)
    print("CURRENT SIGNAL")
    print("="*55)
    
    pred, prob, price, time = train_and_signal(df, target_col)
    
    print(f"Time: {time}")
    print(f"Price: ${price:,.2f}")
    print(f"Prediction: {'UP' if pred == 1 else 'DOWN'}")
    print(f"Confidence: UP {prob[1]:.2%}, DOWN {prob[0]:.2%}")
    
    # EV check (Polymarket fees ~2%)
    if prob[1] > 0.52:
        print(f"\n>>> BET UP: {prob[1]:.2%} > 52% (threshold for +EV)")
    elif prob[0] > 0.52:
        print(f"\n>>> BET DOWN: {prob[0]:.2%} > 52% (threshold for +EV)")
    else:
        print(f"\n>>> No bet: confidence below 52% threshold")
    
    # Save
    results.to_csv(f"{DATA_DIR}/results.csv", index=False)
    print(f"\nResults saved to {DATA_DIR}/results.csv")

if __name__ == "__main__":
    main()