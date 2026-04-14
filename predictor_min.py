#!/usr/bin/env python3
"""BTC Predictor - Minimal fast version"""
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

def get_data():
    f = f"{DATA_DIR}/btc_1m.csv"
    if not os.path.exists(f):
        print("Downloading...")
        d = []
        start = int(datetime(2024, 1, 1).timestamp() * 1000)
        end = int(datetime(2024, 12, 31).timestamp() * 1000)
        while start < end:
            r = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": "BTCUSDT", "interval": "1m", "startTime": start, "endTime": end, "limit": 1000}).json()
            d.extend(r)
            start = r[-1][0] + 1
            print(f"  {len(d)}")
            if len(d) > 200000: break
        df = pd.DataFrame(d[:200000])
    else:
        print("Loading...")
        df = pd.read_csv(f, nrows=200000)
    df = df.iloc[:, :6]
    df.columns = ['t','o','h','l','c','v']
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    for c in 'ohlcv': df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna()

def features(df):
    df = df.sort_values('t').reset_index(drop=True)
    df['ret'] = df['c'].pct_change()
    for lag in [1,3,5]:
        df[f'r{lag}'] = df['ret'].shift(lag)
    df['body'] = (df['c'] - df['o']) / df['c']
    df['rng'] = (df['h'] - df['l']) / df['c']
    
    d = df['c'].diff()
    g = d.where(d>0,0).rolling(14).mean()
    l = (-d.where(d<0,0)).rolling(14).mean()
    df['rsi'] = 100 - (100/(1 + g/(l+1e-10)))
    
    sma = df['c'].rolling(20).mean()
    df['vsma'] = (df['c'] - sma) / sma
    df['vol'] = df['ret'].rolling(10).std()
    df['vrat'] = df['v'] / (df['v'].rolling(20).mean() + 1e-10)
    df['hour'] = df['t'].dt.hour
    
    return df.dropna()

def test():
    df = get_data()
    print(f"Data: {len(df)}")
    df = features(df)
    
    # Target: 5min direction (Polymarket)
    df['tgt'] = (df['c'].shift(-5) > df['c']).astype(int)
    df = df.dropna(subset=['tgt'])
    
    feats = ['r1','r3','r5','body','rng','rsi','vsma','vol','vrat','hour']
    X = df[feats].values
    y = df['tgt'].values
    
    # Split 80/20 (last 20% is test)
    split = int(len(X)*0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    
    print(f"Train: {len(Xtr)}, Test: {len(Xte)}")
    print(f"Train period: {df['t'].iloc[0]} to {df['t'].iloc[split]}")
    print(f"Test period: {df['t'].iloc[split]} to {df['t'].iloc[-1]}")
    
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    
    m = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
    m.fit(Xtr_s, ytr)
    
    tr_acc = (m.predict(Xtr_s) == ytr).mean()
    te_acc = (m.predict(Xte_s) == yte).mean()
    
    print(f"\nTrain accuracy: {tr_acc:.4f}")
    print(f"Test accuracy:  {te_acc:.4f}")
    print(f"Baseline:       {yte.mean():.4f}")
    
    # Top features
    for f,i in sorted(zip(feats, m.feature_importances_), key=lambda x: -x[1])[:5]:
        print(f"  {f}: {i:.3f}")

if __name__ == "__main__":
    test()