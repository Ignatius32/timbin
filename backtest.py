import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os

DATA_DIR = "data"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

os.makedirs(DATA_DIR, exist_ok=True)

def download_binance_klines():
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    
    start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
    end_ts = int(datetime(2024, 12, 31).timestamp() * 1000)
    
    while start_ts < end_ts:
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        start_ts = data[-1][0] + 1
        print(f"Downloaded {len(all_data)} candles...")
    
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df.to_csv(f"{DATA_DIR}/btc_5m.csv", index=False)
    print(f"Saved {len(df)} candles to {DATA_DIR}/btc_5m.csv")
    return df

def add_features(df):
    df = df.copy()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['next_return'] = df['returns'].shift(-1)
    df['direction'] = (df['next_return'] > 0).astype(int)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # SMA
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['price_vs_sma'] = (df['close'] - df['sma_20']) / df['sma_20']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Volume ratio
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Momentum
    df['momentum_5'] = df['close'].pct_change(5)
    
    # Hour of day (for pattern detection)
    df['hour'] = df['timestamp'].dt.hour
    
    return df

def backtest_strategy(df, strategy_name, entry_cond, exit_cond=None):
    df = df.copy()
    df['signal'] = 0
    
    if entry_cond == 'rsi_oversold':
        df.loc[df['rsi'] < 30, 'signal'] = 1
    elif entry_cond == 'rsi_overbought':
        df.loc[df['rsi'] > 70, 'signal'] = -1
    elif entry_cond == 'sma_cross':
        df['sma_20_prev'] = df['sma_20'].shift(1)
        df['sma_50_prev'] = df['sma_50'].shift(1)
        df.loc[(df['sma_20'] > df['sma_50']) & (df['sma_20_prev'] <= df['sma_50_prev']), 'signal'] = 1
        df.loc[(df['sma_20'] < df['sma_50']) & (df['sma_20_prev'] >= df['sma_50_prev']), 'signal'] = -1
    elif entry_cond == 'volume_spike':
        df.loc[df['volume_ratio'] > 2, 'signal'] = 1
    elif entry_cond == 'momentum_pos':
        df.loc[df['momentum_5'] > 0, 'signal'] = 1
    elif entry_cond == 'volatility_low':
        df.loc[df['volatility'] < df['volatility'].quantile(0.2), 'signal'] = 1
    
    # Remove NaN
    valid = df.dropna(subset=['direction', 'signal']).copy()
    
    # Calculate accuracy
    trades = valid[valid['signal'] != 0]
    if len(trades) == 0:
        return None, 0
    
    correct = (trades['signal'] == 1) & (trades['direction'] == 1)
    correct += (trades['signal'] == -1) & (trades['direction'] == 0)
    accuracy = correct.sum() / len(trades)
    
    return accuracy, len(trades)

def main():
    # Download data
    csv_path = f"{DATA_DIR}/btc_5m.csv"
    if os.path.exists(csv_path):
        print("Loading existing data...")
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    else:
        print("Downloading Binance data...")
        df = download_binance_klines()
    
    print(f"Loaded {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Add features
    df = add_features(df)
    
    print("\n=== BACKTEST RESULTS ===\n")
    print(f"Baseline (random guess): 50.0%\n")
    
    strategies = [
        ("RSI Oversold (buy when RSI < 30)", "rsi_oversold"),
        ("RSI Overbought (sell when RSI > 70)", "rsi_overbought"),
        ("SMA Crossover (20 vs 50)", "sma_cross"),
        ("Volume Spike (vol > 2x avg)", "volume_spike"),
        ("Positive Momentum (5min > 0)", "momentum_pos"),
        ("Low Volatility", "volatility_low"),
    ]
    
    results = []
    for name, cond in strategies:
        acc, n_trades = backtest_strategy(df, name, cond)
        if acc:
            results.append((name, acc, n_trades))
            print(f"{name}")
            print(f"  Accuracy: {acc*100:.2f}% ({n_trades} trades)")
            print()
    
    # Best strategy
    if results:
        best = max(results, key=lambda x: x[1])
        print(f"Best: {best[0]} with {best[1]*100:.2f}%")

if __name__ == "__main__":
    main()