#!/usr/bin/env python3
"""
BTC Predictor - Optimized version
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_data():
    """Download 1m and 5m data"""
    # 1m data - 2024 only (faster)
    f1m = f"{DATA_DIR}/btc_1m_2024.csv"
    if not os.path.exists(f1m):
        print("Downloading 1m data...")
        url = "https://api.binance.com/api/v3/klines"
        data = []
        start = int(datetime(2024, 1, 1).timestamp() * 1000)
        end = int(datetime(2025, 1, 1).timestamp() * 1000)
        while start < end:
            d = requests.get(url, params={"symbol": "BTCUSDT", "interval": "1m", "startTime": start, "endTime": end, "limit": 1000}).json()
            if not d: break
            data.extend(d)
            start = d[-1][0] + 1
            if len(data) % 50000 == 0: print(f"  {len(data)}...")
        df = pd.DataFrame(data[:500000], columns=['timestamp','open','high','low','close','volume','close_time','qv','trades','tb','tbo','i'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
        df.to_csv(f1m, index=False)
        print(f"Saved {len(df)} rows")
    else:
        print("Loading 1m data...")
    
    # 5m data
    f5m = f"{DATA_DIR}/btc_5m_2024.csv"
    if not os.path.exists(f5m):
        print("Downloading 5m data...")
        url = "https://api.binance.com/api/v3/klines"
        data = []
        start = int(datetime(2024, 1, 1).timestamp() * 1000)
        end = int(datetime(2025, 1, 1).timestamp() * 1000)
        while start < end:
            d = requests.get(url, params={"symbol": "BTCUSDT", "interval": "5m", "startTime": start, "endTime": end, "limit": 1000}).json()
            if not d: break
            data.extend(d)
            start = d[-1][0] + 1
        df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume','close_time','qv','trades','tb','tbo','i'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
        df.to_csv(f5m, index=False)
        print(f"Saved {len(df)} rows")
    else:
        print("Loading 5m data...")
    
    return pd.read_csv(f1m, parse_dates=['timestamp']), pd.read_csv(f5m, parse_dates=['timestamp'])

def make_features(df_1m):
    """Create features from 1m data"""
    df = df_1m.sort_values('timestamp').copy()
    
    # Returns
    for lag in [1, 3, 5, 10, 15]:
        df[f'r{lag}'] = df['close'].pct_change(lag)
    
    # Candle
    df['body'] = (df['close'] - df['open']) / df['open']
    df['rng'] = (df['high'] - df['low']) / df['close']
    
    # RSI
    for p in [7, 14]:
        d = df['close'].diff()
        g = d.where(d>0,0).rolling(p).mean()
        l = (-d.where(d<0,0)).rolling(p).mean()
        df[f'rsi{p}'] = 100 - (100/(1 + g/(l+1e-10)))
    
    # SMA
    for p in [10, 30]:
        sma = df['close'].rolling(p).mean()
        df[f'vsma{p}'] = (df['close'] - sma) / sma
    
    # Vol
    df['vol5'] = df['r1'].rolling(5).std()
    df['vol15'] = df['r1'].rolling(15).std()
    
    # Volume
    df['vma'] = df['volume'].rolling(30).mean()
    df['vr'] = df['volume'] / (df['vma']+1e-10)
    
    # MACD
    e12, e26 = df['close'].ewm(span=12, adjust=False).mean(), df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = e12 - e26
    df['macdh'] = df['macd'] - df['macd'].ewm(span=9, adjust=False).mean()
    
    # BB
    bb = df['close'].rolling(20)
    df['bbp'] = (df['close'] - (bb.mean() - 2*bb.std())) / (4*bb.std() + 1e-10)
    
    # Time
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    
    # Lags
    for lag in [1, 2, 3]:
        df[f'rl{lag}'] = df['r1'].shift(lag)
    
    return df

def add_5m_features(df_1m, df_5m):
    """Add 5m features to 1m data - compute 5m features first"""
    df_5m = df_5m.sort_values('timestamp').copy()
    
    # Compute features on 5m
    df_5m['r1'] = df_5m['close'].pct_change(1)
    d = df_5m['close'].diff()
    g = d.where(d>0,0).rolling(7).mean()
    l = (-d.where(d<0,0)).rolling(7).mean()
    df_5m['rsi7'] = 100 - (100/(1 + g/(l+1e-10)))
    sma = df_5m['close'].rolling(10).mean()
    df_5m['vsma10'] = (df_5m['close'] - sma) / sma
    e12 = df_5m['close'].ewm(span=12, adjust=False).mean()
    e26 = df_5m['close'].ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    df_5m['macdh'] = macd - macd.ewm(span=9, adjust=False).mean()
    
    f5 = df_5m[['timestamp', 'r1', 'rsi7', 'vsma10', 'macdh']].copy()
    f5.columns = ['ts5', 'r5_1', 'r5_rsi7', 'r5_vsma10', 'r5_macdh']
    
    df = df_1m.merge(f5, left_on='timestamp', right_on='ts5', how='left').drop('ts5', axis=1)
    df = df.sort_values('timestamp').reset_index(drop=True)
    for c in ['r5_1', 'r5_rsi7', 'r5_vsma10', 'r5_macdh']:
        df[c] = df[c].ffill()
    return df

def run_walkforward(df, target_col, n_test=5):
    """Walk-forward validation"""
    feats = ['r1', 'r3', 'r5', 'r10', 'r15', 'body', 'rng', 'rsi7', 'rsi14', 'vsma10', 'vsma30',
             'vol5', 'vol15', 'vr', 'macdh', 'bbp', 'hour', 'dow', 'rl1', 'rl2', 'rl3',
             'r5_1', 'r5_rsi7', 'r5_vsma10', 'r5_macdh']
    feats = [f for f in feats if f in df.columns]
    
    df = df.dropna(subset=feats + [target_col]).copy()
    df = df.reset_index(drop=True)
    
    # Replace inf
    for c in feats:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feats)
    
    n = len(df)
    split = int(n * 0.85)
    
    X_train = df.iloc[:split][feats].values
    y_train = df.iloc[:split][target_col].values
    X_test = df.iloc[split:][feats].values
    y_test = df.iloc[split:][target_col].values
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[split]}")
    print(f"Test period: {df['timestamp'].iloc[split]} to {df['timestamp'].iloc[-1]}")
    
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42, subsample=0.7)
    model.fit(X_train_s, y_train)
    
    train_acc = (model.predict(X_train_s) == y_train).mean()
    test_acc = (model.predict(X_test_s) == y_test).mean()
    
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Baseline:       {y_test.mean():.4f}")
    
    # Feature importance
    imp = pd.DataFrame({'f': feats, 'i': model.feature_importances_}).sort_values('i', ascending=False)
    print("\nTop features:")
    print(imp.head(10).to_string(index=False))
    
    return test_acc

def main():
    print("="*50)
    print("BTC Polymarket Predictor")
    print("="*50)
    
    df_1m, df_5m = download_data()
    print(f"\nData: {len(df_1m)} 1m, {len(df_5m)} 5m")
    
    print("\nCreating features...")
    df = make_features(df_1m)
    df = add_5m_features(df, df_5m)
    
    # Targets for different horizons
    print("\n=== Experiments ===\n")
    
    for horizon in [1, 3, 5, 10]:
        print(f"--- Predict {horizon}min direction ---")
        df[f'tgt{horizon}'] = (df['close'].shift(-horizon) > df['close']).astype(int)
        run_walkforward(df, f'tgt{horizon}')
        print()

if __name__ == "__main__":
    main()