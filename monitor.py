#!/usr/bin/env python3
"""
BTC Polymarket Monitor - runs every minute
"""
import os
import time
import sys

# ANSI colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'

def clear():
    os.system('clear' if os.name == 'posix' else 'cls')

def run_prediction():
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
        df = df.copy()
        
        # Target: Next bar close > next bar open (Polymarket resolution)
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

    sys.stdout.write("Loading data... ")
    sys.stdout.flush()
    
    # Load and train
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

    sys.stdout.write(f"({len(df)} bars) ")
    sys.stdout.flush()
    
    df = add_features(df)

    features = ['ret_1', 'ret_2', 'ret_3', 'ret_5', 'body', 'range', 'rsi_7', 
               'vsma_5', 'vsma_10', 'vsma_20', 'volatility', 'vol_ratio',
               'minute', 'hour', 'dayofweek', 'green_streak']

    df = df.dropna(subset=features + ['target']).reset_index(drop=True)

    sys.stdout.write("Training... ")
    sys.stdout.flush()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].values)
    y = df['target'].values

    models = [
        GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=42),
        LogisticRegression(max_iter=1000)
    ]

    for m in models:
        m.fit(X, y)
    
    sys.stdout.write("Predicting... ")
    sys.stdout.flush()

    # Predict
    df = get_recent_bars(100)
    df = add_features(df)
    df = df.dropna(subset=features).reset_index(drop=True)
    
    last_bar = df.iloc[-1:]
    X = scaler.transform(last_bar[features].values)
    
    probs = [m.predict_proba(X)[:, 1] for m in models]
    avg_prob = np.mean(probs, axis=0)[0]
    
    return avg_prob, last_bar.iloc[0]['timestamp'], last_bar.iloc[0]['close']

def main():
    print(f"{BOLD}BTC Polymarket Monitor - checking every minute...{RESET}")
    print("Press Ctrl+C to stop\n")
    sys.stdout.flush()
    
    while True:
        try:
            prob, ts, price = run_prediction()
            
            clear()
            now = datetime.now().strftime("%H:%M:%S")
            
            # Color based on confidence
            conf = max(prob, 1 - prob)
            if conf >= 0.60:
                color = GREEN
                emoji = "🟢"
            elif conf >= 0.55:
                color = YELLOW
                emoji = "🟡"
            else:
                color = RED
                emoji = "🔴"
            
            direction = "UP" if prob > 0.5 else "DOWN"
            
            print(f"{BOLD}{'='*50}{RESET}")
            print(f"{BOLD}BTC POLYMARKET MONITOR{RESET} | {now}")
            print(f"{'='*50}")
            print(f"Last bar: {ts} | ${price:,.2f}")
            print()
            print(f"{color}{BOLD}UP: {prob:.1%}{RESET} | DOWN: {1-prob:.1%}")
            print()
            
            if conf >= 0.60:
                print(f"{color}{BOLD}{emoji} BET {direction} - HIGH CONFIDENCE!{RESET}")
            elif conf >= 0.55:
                print(f"{color}{BOLD}{emoji} BET {direction} - marginal (55-60%){RESET}")
            else:
                print(f"{color}{BOLD}{emoji} NO BET - below 55% threshold{RESET}")
            
            print()
            print(f"Next check in 60 seconds...")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"Error: {e}")
            sys.stdout.flush()
        
        time.sleep(60)

if __name__ == "__main__":
    from datetime import datetime
    main()