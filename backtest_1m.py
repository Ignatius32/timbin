import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os

DATA_DIR = "data"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

os.makedirs(DATA_DIR, exist_ok=True)

def download_binance_klines():
    csv_path = f"{DATA_DIR}/btc_1m.csv"
    if os.path.exists(csv_path):
        print("Loading existing data...")
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        return df
    
    print("Downloading 1m Binance data (this will take a while)...")
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
        if len(all_data) % 10000 == 0:
            print(f"Downloaded {len(all_data)} candles...")
    
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} candles to {csv_path}")
    return df

def add_features(df):
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Target: next 1-minute direction
    df['next_close'] = df['close'].shift(-1)
    df['target'] = (df['next_close'] > df['close']).astype(int)
    
    # Returns
    df['returns_1'] = df['close'].pct_change(1)
    df['returns_3'] = df['close'].pct_change(3)
    df['returns_5'] = df['close'].pct_change(5)
    df['returns_15'] = df['close'].pct_change(15)
    
    # Price position in candle
    df['candle_body'] = (df['close'] - df['open']) / df['open']
    df['candle_range'] = (df['high'] - df['low']) / df['low']
    
    # RSI (multiple windows)
    for window in [7, 14, 30]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    # Moving averages
    for window in [5, 10, 20, 50, 100]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'price_vs_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']
    
    # EMA
    for window in [5, 20]:
        df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
    
    # Volatility
    for window in [5, 15, 30]:
        df[f'volatility_{window}'] = df['returns_1'].rolling(window=window).std()
    
    # Volume
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
    df['volume_change'] = df['volume'].pct_change()
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # Momentum
    for window in [5, 10, 20]:
        df[f'momentum_{window}'] = df['close'].pct_change(window)
    
    # High/Low positions
    for window in [10, 30, 60]:
        roll_high = df['high'].rolling(window=window).max()
        roll_low = df['low'].rolling(window=window).min()
        df[f'high_low_pos_{window}'] = (df['close'] - roll_low) / (roll_high - roll_low + 1e-10)
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['minute'] = df['timestamp'].dt.minute
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f'returns_lag_{lag}'] = df['returns_1'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].pct_change().shift(lag)
    
    return df

def walk_forward_validation(df, feature_cols, n_train_days=60, n_test_days=7):
    """
    Walk-forward validation to avoid lookahead bias and overfitting.
    Train on past data, test on future data (no data leakage).
    """
    df = df.copy()
    df = df.dropna(subset=feature_cols + ['target']).copy()
    df = df.reset_index(drop=True)
    
    results = []
    
    # Calculate test periods
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    
    current_date = min_date + pd.Timedelta(days=n_train_days)
    test_count = 0
    
    while current_date + pd.Timedelta(days=n_test_days) <= max_date:
        train_end = current_date
        test_end = current_date + pd.Timedelta(days=n_test_days)
        
        train_mask = df['timestamp'] < current_date
        test_mask = (df['timestamp'] >= current_date) & (df['timestamp'] < test_end)
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) < 1000 or len(test_df) < 100:
            current_date += pd.Timedelta(days=n_test_days)
            continue
        
        # Simple logistic-like scoring (no model to avoid overfitting)
        # Score each feature and combine
        
        # RSI under 30 -> up
        train_rsi_median = train_df['rsi_14'].median()
        rsi_signal = (test_df['rsi_14'] < 30).astype(float) - (test_df['rsi_14'] > 70).astype(float)
        
        # Volatility regime
        train_vol_median = train_df['volatility_15'].median()
        vol_signal = (test_df['volatility_15'] < train_vol_median).astype(float) * 2 - 1
        
        # Price vs SMA
        sma_signal = (test_df['price_vs_sma_20'] > 0).astype(float) * 2 - 1
        
        # MACD
        macd_signal = (test_df['macd_hist'] > 0).astype(float) * 2 - 1
        
        # Combine signals (simple ensemble)
        combined = (rsi_signal + vol_signal + sma_signal + macd_signal) / 4
        predicted = (combined > 0).astype(int)
        
        actual = test_df['target'].values
        correct = (predicted == actual).mean()
        
        results.append({
            'train_end': train_end,
            'test_start': current_date,
            'test_end': test_end,
            'n_trades': len(test_df),
            'accuracy': correct,
            'predicted_up_pct': predicted.mean()
        })
        
        test_count += 1
        current_date += pd.Timedelta(days=n_test_days)
        
        if test_count % 10 == 0:
            print(f"Test {test_count}: {current_date.date()}, accuracy: {correct:.2%}")
    
    return pd.DataFrame(results)

def main():
    df = download_binance_klines()
    print(f"Loaded {len(df)} candles")
    
    print("Adding features...")
    df = add_features(df)
    print(f"Features added. Shape: {df.shape}")
    
    feature_cols = [
        'returns_1', 'returns_3', 'returns_5', 'returns_15',
        'candle_body', 'candle_range',
        'rsi_7', 'rsi_14', 'rsi_30',
        'price_vs_sma_5', 'price_vs_sma_20', 'price_vs_sma_50',
        'volatility_5', 'volatility_15', 'volatility_30',
        'volume_ratio', 'volume_change',
        'macd', 'macd_signal', 'macd_hist',
        'bb_position',
        'momentum_5', 'momentum_20',
        'high_low_pos_10', 'high_low_pos_30', 'high_low_pos_60',
        'hour', 'dayofweek',
        'returns_lag_1', 'returns_lag_2', 'returns_lag_3'
    ]
    
    print("\n=== Walk-Forward Validation ===")
    print("Train: 60 days, Test: 7 days\n")
    
    results = walk_forward_validation(df, feature_cols, n_train_days=60, n_test_days=7)
    
    if len(results) > 0:
        print("\n=== RESULTS ===")
        print(f"Total test periods: {len(results)}")
        print(f"Mean accuracy: {results['accuracy'].mean():.4f} ({results['accuracy'].mean()*100:.2f}%)")
        print(f"Std accuracy: {results['accuracy'].std():.4f}")
        print(f"Min accuracy: {results['accuracy'].min():.4f}")
        print(f"Max accuracy: {results['accuracy'].max():.4f}")
        
        # How many periods beat 50%
        beats_50 = (results['accuracy'] > 0.5).sum()
        print(f"Periods beating 50%: {beats_50}/{len(results)} ({beats_50/len(results)*100:.1f}%)")
        
        # Save results
        results.to_csv(f"{DATA_DIR}/walkforward_results.csv", index=False)
        print(f"\nResults saved to {DATA_DIR}/walkforward_results.csv")

if __name__ == "__main__":
    main()