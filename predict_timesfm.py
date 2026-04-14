#!/usr/bin/env python3
"""
Predictor using fine-tuned TimesFM model
Works like the 1m predictor: predicts 5min window closing direction
"""
import numpy as np
import pandas as pd
import requests
import torch
from transformers import TimesFm2_5ModelForPrediction
import warnings
warnings.filterwarnings('ignore')

def get_btc_1m(n=500):
    df = pd.DataFrame(
        requests.get("https://api.binance.com/api/v3/klines", 
                   params={"symbol":"BTCUSDT","interval":"1m","limit":n}).json(),
        columns=['t','o','h','l','c','v','ct','qv','tr','tbb','tbq','i']
    )
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    for c in ['o','h','l','c','v']: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

MODEL_PATH = "cache/timesfm_btc_ft"

def load_model():
    """Load fine-tuned TimesFM model"""
    import os
    if os.path.exists(MODEL_PATH):
        print(f"Loading fine-tuned TimesFM from {MODEL_PATH}...")
        return TimesFm2_5ModelForPrediction.from_pretrained(MODEL_PATH, device_map="cuda")
    else:
        print("No fine-tuned model found, using original TimesFM...")
        return TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers", 
            device_map="cuda"
        )

print("Loading TimesFM model...")
model = load_model()
model.eval()

print("\n=== TimesFM Predictor for 5min Window ===")
print("Predicting: will 5min bar close > its open?")
print()

import time
from datetime import datetime

while True:
    try:
        now = datetime.now()
        minute = now.minute
        
        # Position in 5min window
        pos_in_5min = minute % 5
        bars_left = 5 - pos_in_5min
        
        # Get current data
        df = get_btc_1m(200)
        closes = df['c'].values
        
        # Current 5min window open (5 bars ago)
        open_5m = closes[-5]
        current_price = closes[-1]
        
        with torch.no_grad():
            # Forecast next 5 bars
            forecast_input = torch.tensor(closes[-128:], dtype=torch.float32).cuda().unsqueeze(0)
            outputs = model(past_values=forecast_input, return_dict=True)
            pred = outputs.mean_predictions[0].cpu().numpy()
        
        # Predicted price at end of 5min window
        predicted_5m_close = pred[4]
        
        # Direction prediction
        will_up = predicted_5m_close > open_5m
        
        # Confidence (based on how far the prediction is from current)
        change_pct = abs(predicted_5m_close - current_price) / current_price
        confidence = min(change_pct * 100, 1.0)  # Scale to 0-1
        
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        
        c = GREEN if confidence > 0.6 else YELLOW if confidence > 0.3 else RESET
        
        print(f"{now.strftime('%H:%M:%S')} | Bars left: {bars_left}")
        print(f"  Open: {open_5m:.2f} | Current: {current_price:.2f}")
        print(f"  Predicted 5m close: {predicted_5m_close:.2f}")
        if will_up:
            print(f"  {c}>>> BET UP ({confidence:.1%}){RESET}")
        else:
            print(f"  {c}>>> BET DOWN ({confidence:.1%}){RESET}")
        
        print()
        
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(60)