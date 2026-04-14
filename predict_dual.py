#!/usr/bin/env python3
"""
Dual predictor: 
- TimesFM (fine-tuned or original)
- XGBoost 1m recursive
Shows both predictions
"""
import numpy as np
import pandas as pd
import requests
import pickle
import torch
from transformers import TimesFm2_5ModelForPrediction
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

def get_btc_1m(n=500):
    df = pd.DataFrame(
        requests.get("https://api.binance.com/api/v3/klines", 
                   params={"symbol":"BTCUSDT","interval":"1m","limit":n}).json(),
        columns=['t','o','h','l','c','v','ct','qv','tr','tbb','tbq','i']
    )
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    for c in ['o','h','l','c','v']: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def load_xgb():
    with open("cache/1m_models.pkl", 'rb') as f:
        return pickle.load(f)

def feats(df):
    df = df.copy()
    for lag in [1, 2, 3, 5, 10]:
        df[f'r{lag}'] = df['c'].pct_change(lag)
    df['body'] = (df['c'] - df['o']) / df['o']
    df['range'] = (df['h'] - df['l']) / df['c']
    d = df['c'].diff()
    g = d.where(d>0,0).rolling(7).mean()
    l = d.where(d<0,0).rolling(7).mean()
    df['rsi7'] = (100 - (100/(1+g/(l+1e-10)))).shift(1)
    for p in [5, 10, 20]:
        df[f'sma{p}'] = (df['c'] - df['c'].rolling(p).mean().shift(1)) / df['c'].rolling(p).mean().shift(1)
    df['vol'] = df['r1'].rolling(10).std()
    df['vr'] = df['v'] / df['v'].rolling(10).mean().shift(1)
    df['hr'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    df['is_green'] = (df['c'] > df['o']).astype(int)
    df['streak'] = df['is_green'].groupby((df['is_green'] != df['is_green'].shift()).cumsum()).cumcount()
    df['macd'] = (df['c'].ewm(span=12).mean() - df['c'].ewm(span=26).mean()).shift(1)
    df['macds'] = df['macd'].ewm(span=9).mean().shift(1)
    df['macdh'] = df['macd'] - df['macds']
    return df

features = ['r1','r2','r3','r5','r10','body','range','rsi7','sma5','sma10','sma20','vol','vr','hr','dow','streak','macd','macds','macdh']

# Load models
print("Loading XGBoost model...")
d1 = load_xgb()

print("Loading TimesFM model...")
import os
if os.path.exists("cache/timesfm_btc_ft"):
    tfm = TimesFm2_5ModelForPrediction.from_pretrained("cache/timesfm_btc_ft", device_map="cuda")
    print("Using FINE-TUNED TimesFM")
else:
    tfm = TimesFm2_5ModelForPrediction.from_pretrained("google/timesfm-2.5-200m-transformers", device_map="cuda")
    print("Using original TimesFM")
tfm.eval()

print("\n=== DUAL PREDICTOR (XGBoost + TimesFM) ===")
print()

while True:
    try:
        now = datetime.now()
        minute = now.minute
        bars_left = 5 - (minute % 5)
        
        df1 = get_btc_1m(200)
        closes = df1['c'].values
        
        # === XGBoost 1m recursive ===
        df_feat = feats(df1)
        df_clean = df_feat.dropna(subset=features).reset_index(drop=True)
        X = d1['scaler'].transform(df_clean[features].iloc[-1:])
        
        probs = []
        for _ in range(bars_left):
            p = d1['xgb'].predict_proba(X)[:,1][0]
            probs.append(p)
        xgb_prob = np.mean(probs)
        xgb_conf = max(xgb_prob, 1-xgb_prob)
        xgb_dir = "UP" if xgb_prob > 0.5 else "DOWN"
        
        # === TimesFM prediction ===
        open_5m = closes[-5]
        
        with torch.no_grad():
            inp = torch.tensor(closes[-128:], dtype=torch.float32).cuda().unsqueeze(0)
            out = tfm(past_values=inp, return_dict=True)
            pred = out.mean_predictions[0].cpu().numpy()
        
        pred_5m_close = pred[4]
        tfm_dir = "UP" if pred_5m_close > open_5m else "DOWN"
        
        # Colors
        cx = GREEN if xgb_conf >= 0.60 else YELLOW if xgb_conf >= 0.55 else RED
        
        print(f"{now.strftime('%H:%M:%S')} | Bars left: {bars_left}")
        print(f"  Open: {open_5m:.2f} | Current: {closes[-1]:.2f}")
        print()
        print(f"[XGBoost 1m Recursive]")
        print(f"  Prob UP: {xgb_prob:.1%} | {cx}>>> BET {xgb_dir} ({xgb_conf:.1%}){RESET}")
        print()
        print(f"[TimesFM]")
        print(f"  Predicted: {pred_5m_close:.2f} | >>> BET {tfm_dir}")
        print()
        print("-" * 40)
        
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(60)