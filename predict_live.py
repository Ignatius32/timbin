#!/usr/bin/env python3
"""
BTC Polymarket Predictor - LIVE Training
Trains on FRESH API data (not old files)
"""

import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def get_data(n=500):
    """Get recent 5m data from Binance"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "5m", "limit": n}
    d = requests.get(url, params=params).json()
    
    df = pd.DataFrame(d)[[0,1,2,3,4,5]].astype(float)
    df.columns = ['timestamp','open','high','low','close','volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

print("=" * 55)
print("BTC Polymarket - LIVE Training Version")
print("=" * 55)

# Get LOTS of fresh data
print("\n1. Fetching fresh data from Binance...")
df = get_data(1000)  # 1000 bars = ~3.5 days
print(f"   Got {len(df)} bars")

# Create features
print("2. Creating features...")
for l in [1,2,3,5]:
    df[f'r{l}'] = df.close.pct_change(l).shift(1)

df['body'] = (df.close - df.open) / df.open

delta = df.close.diff()
g = delta.where(delta > 0, 0).rolling(7).mean()
l = delta.where(delta < 0, 0).rolling(7).mean()
df['rsi'] = (100 - (100/(1 + g/(l+1e-10)))).shift(1)

for p in [5,10,20]:
    sma = df.close.rolling(p).mean().shift(1)
    df[f'vsma{p}'] = (df.close - sma) / sma

df['vol'] = df.r1.rolling(10).std()
df['vol_ratio'] = df.volume / df.volume.rolling(10).mean()

# Target: next bar UP
df['target'] = (df.close.shift(-1) > df.close).astype(int)

# Features
features = ['r1','r2','r3','r5','body','rsi','vsma5','vsma10','vsma20','vol','vol_ratio']

# Clean
df = df.dropna(subset=features + ['target'])
print(f"   Clean data: {len(df)} bars")

# Train
print("3. Training model...")
X = df[features].values
y = df['target'].values

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

m1 = GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=42)
m2 = LogisticRegression(max_iter=1000)
m1.fit(X_s, y)
m2.fit(X_s, y)

# Predict
print("4. Predicting...")
X_cur = scaler.transform(df[features].values[-1:])
p = (m1.predict_proba(X_cur)[0][1] + m2.predict_proba(X_cur)[0][1]) / 2

print(f"\n=== CURRENT SIGNAL ===")
print(f"Last bar: {df.iloc[-1]['timestamp']}")
print(f"Price: ${df.iloc[-1]['close']:,.2f}")
print(f"UP probability: {p:.1%}")

if p > 0.58:
    print(f"\n>>> BET UP! (confidence {p:.0%})")
elif p < 0.42:
    print(f"\n>>> BET DOWN! (confidence {1-p:.0%})")
elif p > 0.55:
    print(f"\n>>> small UP (confidence {p:.0%})")
elif p < 0.45:
    print(f"\n>>> small DOWN (confidence {1-p:.0%})")
else:
    print(f"\n>>> NO BET (confidence below 55%)")

print(f"\nTrained on LAST {len(df)} fresh bars (~3.5 days)")
print("=" * 55)