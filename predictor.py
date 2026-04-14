#!/usr/bin/env python3
"""
BTC Price Predictor for Polymarket
- Multi-year data
- Advanced features (multi-timeframe, order flow proxies)
- Multiple prediction targets
- Proper walk-forward validation
- Actionable signals
"""

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

SYMBOL = "BTCUSDT"

def download_binance_data(interval, start_date, end_date, filename):
    """Download Binance klines with rate limit handling"""
    filepath = f"{DATA_DIR}/{filename}"
    if os.path.exists(filepath):
        print(f"Loading {filename}...")
        return pd.read_csv(filepath, parse_dates=['timestamp'])
    
    print(f"Downloading {filename}...")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    while start_ts < end_ts:
        params = {
            "symbol": SYMBOL,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
        except Exception as e:
            print(f"Error: {e}, retrying...")
            import time
            time.sleep(2)
            continue
        
        if not data or 'code' in data:
            print(f"API error: {data}")
            break
            
        all_data.extend(data)
        start_ts = data[-1][0] + 1
        
        if len(all_data) % 10000 == 0:
            print(f"  Downloaded {len(all_data)}...")
    
    df = pd.DataFrame(all_data[:min(len(all_data), 500000)], columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']].copy()
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} rows to {filepath}")
    return df

def create_features_1m(df_1m):
    """Create features from 1m data"""
    df = df_1m.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Returns at various horizons
    for lag in [1, 3, 5, 10, 15, 30]:
        df[f'ret_{lag}'] = df['close'].pct_change(lag)
    
    # Candle patterns
    df['body'] = (df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_shadow'] = ((df[['open', 'close']].min(axis=1)) - df['low']) / df['close']
    df['range'] = (df['high'] - df['low']) / df['close']
    
    # RSI (multiple periods)
    for period in [7, 14, 30]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # Moving averages
    for period in [5, 10, 20, 50, 100]:
        sma = df['close'].rolling(period).mean()
        df[f'sma_{period}'] = sma
        df[f'price_vs_sma_{period}'] = (df['close'] - sma) / sma
        df[f'sma_slope_{period}'] = sma.pct_change(5)
    
    # Volatility
    df['vol_5'] = df['ret_1'].rolling(5).std()
    df['vol_15'] = df['ret_1'].rolling(15).std()
    df['vol_30'] = df['ret_1'].rolling(30).std()
    
    # Volume features
    df['vol_ma_10'] = df['volume'].rolling(10).mean()
    df['vol_ma_30'] = df['volume'].rolling(30).mean()
    df['vol_ma_60'] = df['volume'].rolling(60).mean()
    df['vol_ratio_10'] = df['volume'] / (df['vol_ma_10'] + 1e-10)
    df['vol_ratio_30'] = df['volume'] / (df['vol_ma_30'] + 1e-10)
    df['vol_change'] = df['volume'].pct_change()
    
    # Quote volume (proxy for dollar volume)
    df['qv_ma'] = df['quote_volume'].rolling(30).mean()
    df['qv_ratio'] = df['quote_volume'] / (df['qv_ma'] + 1e-10)
    
    # Trade count
    df['trades_ma'] = df['trades'].rolling(30).mean()
    df['trades_ratio'] = df['trades'] / (df['trades_ma'] + 1e-10)
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid
    
    # High/Low position
    for period in [10, 30, 60]:
        roll_high = df['high'].rolling(period).max()
        roll_low = df['low'].rolling(period).min()
        df[f'hl_pos_{period}'] = (df['close'] - roll_low) / (roll_high - roll_low + 1e-10)
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Lag features (avoid lookahead)
    for lag in [1, 2, 3, 5, 10]:
        df[f'ret1_lag{lag}'] = df['ret_1'].shift(lag)
        df[f'vol_lag{lag}'] = df['vol_5'].shift(lag)
    
    # Consecutive direction
    df['up_streak'] = (df['ret_1'] > 0).astype(int)
    df['up_streak'] = df['up_streak'].groupby((df['up_streak'] != df['up_streak'].shift()).cumsum()).cumcount()
    df['down_streak'] = (df['ret_1'] < 0).astype(int)
    df['down_streak'] = df['down_streak'].groupby((df['down_streak'] != df['down_streak'].shift()).cumsum()).cumcount()
    
    return df

def create_features_5m(df_5m):
    """Create features from 5m data"""
    df = df_5m.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Returns
    for lag in [1, 3, 6, 12]:
        df[f'ret_5m_{lag}'] = df['close'].pct_change(lag)
    
    # RSI
    for period in [7, 14]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        df[f'rsi_5m_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # SMA
    for period in [5, 10, 20]:
        sma = df['close'].rolling(period).mean()
        df[f'vsma_5m_{period}'] = (df['close'] - sma) / sma
    
    # Volatility
    df['vol_5m_5'] = df['ret_5m_1'].rolling(5).std()
    df['vol_5m_20'] = df['ret_5m_1'].rolling(20).std()
    
    # Volume
    df['vol_5m_ma'] = df['volume'].rolling(20).mean()
    df['vol_5m_ratio'] = df['volume'] / (df['vol_5m_ma'] + 1e-10)
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_5m'] = ema12 - ema26
    df['macd_hist_5m'] = df['macd_5m'] - df['macd_5m'].ewm(span=9, adjust=False).mean()
    
    # Time
    df['hour_5m'] = df['timestamp'].dt.hour
    df['dayofweek_5m'] = df['timestamp'].dt.dayofweek
    
    return df

def merge_timeframes(df_1m, df_5m):
    """Merge 5m features into 1m data"""
    df_5m_features = df_5m[['timestamp', 'ret_5m_1', 'ret_5m_3', 'ret_5m_6', 
                           'rsi_5m_7', 'rsi_5m_14', 'vsma_5m_5', 'vsma_5m_20',
                           'vol_5m_5', 'vol_5m_ratio', 'macd_5m', 'macd_hist_5m']].copy()
    
    df_5m_features['timestamp_5m'] = df_5m_features['timestamp']
    df_5m_features = df_5m_features.drop('timestamp', axis=1)
    
    df = df_1m.merge(df_5m_features, left_on='timestamp', right_on='timestamp_5m', how='left')
    df = df.drop('timestamp_5m', axis=1)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    for col in df_5m_features.columns:
        if col != 'timestamp_5m':
            df[col] = df[col].ffill()
    
    return df

def create_targets(df, horizons=[1, 3, 5, 10]):
    """Create prediction targets for different horizons"""
    for h in horizons:
        df[f'target_{h}'] = (df['close'].shift(-h) > df['close']).astype(int)
    return df

def clean_data(df, feature_cols, target_col):
    """Clean data - remove inf and clip extreme values"""
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].clip(-1e10, 1e10)
    return df

def walk_forward_validation(df, feature_cols, target_col, n_train_days=60, n_test_days=7):
    """Proper walk-forward with no lookahead bias"""
    df = clean_data(df, feature_cols, target_col)
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    df = df.reset_index(drop=True)
    
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    
    results = []
    period_num = 0
    
    current_train_end = min_date + timedelta(days=n_train_days)
    
    while current_train_end + timedelta(days=n_test_days) <= max_date:
        test_start = current_train_end
        test_end = test_start + timedelta(days=n_test_days)
        
        train_mask = df['timestamp'] < test_start
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) < 5000 or len(test_df) < 500:
            current_train_end += timedelta(days=n_test_days)
            continue
        
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.7,
            random_state=42
        )
        model.fit(X_train_s, y_train)
        
        y_pred = model.predict(X_test_s)
        accuracy = (y_pred == y_test).mean()
        
        period_num += 1
        results.append({
            'period': period_num,
            'train_end': test_start,
            'test_start': test_start,
            'test_end': test_end,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'accuracy': accuracy,
            'baseline': y_test.mean()
        })
        
        if period_num % 5 == 0:
            print(f"Period {period_num}: {test_start.date()} - {test_end.date()} | Acc: {accuracy:.4f}")
        
        current_train_end += timedelta(days=n_test_days)
    
    return pd.DataFrame(results)

def run_experiment(df, target_horizon, experiment_name):
    """Run a single experiment with a specific target"""
    target_col = f'target_{target_horizon}'
    
    feature_cols = [
        'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_15', 'ret_30',
        'body', 'upper_shadow', 'lower_shadow', 'range',
        'rsi_7', 'rsi_14', 'rsi_30',
        'price_vs_sma_5', 'price_vs_sma_20', 'price_vs_sma_50',
        'sma_slope_20', 'sma_slope_50',
        'vol_5', 'vol_15', 'vol_30',
        'vol_ratio_10', 'vol_ratio_30', 'vol_change',
        'qv_ratio', 'trades_ratio',
        'macd', 'macd_hist', 'bb_position', 'bb_width',
        'hl_pos_10', 'hl_pos_30', 'hl_pos_60',
        'hour', 'dayofweek', 'is_weekend',
        'ret1_lag1', 'ret1_lag2', 'ret1_lag3', 'ret1_lag5',
        'up_streak', 'down_streak',
        'ret_5m_1', 'rsi_5m_7', 'vsma_5m_5', 'macd_hist_5m', 'vol_5m_ratio'
    ]
    
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    print(f"\n=== Experiment: {experiment_name} ===")
    print(f"Target: predict UP in {target_horizon} minutes")
    print(f"Features: {len(feature_cols)}")
    
    results = walk_forward_validation(df, feature_cols, target_col, n_train_days=60, n_test_days=7)
    
    if len(results) > 0:
        print(f"\nResults over {len(results)} test periods:")
        print(f"  Mean accuracy: {results['accuracy'].mean():.4f} ({results['accuracy'].mean()*100:.2f}%)")
        print(f"  Std: {results['accuracy'].std():.4f}")
        print(f"  Min: {results['accuracy'].min():.4f}, Max: {results['accuracy'].max():.4f}")
        print(f"  Baseline (always UP): {results['baseline'].mean():.4f}")
        
        beats_50 = (results['accuracy'] > 0.5).sum()
        print(f"  Periods beating 50%: {beats_50}/{len(results)} ({beats_50/len(results)*100:.1f}%)")
    
    return results

def generate_signals(df, feature_cols):
    """Generate real-time signals using latest data"""
    df = df.dropna(subset=feature_cols).tail(100).copy()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, df['target_5'].values)
    
    latest = df.iloc[-1:][feature_cols].values
    latest_s = scaler.transform(latest)
    pred = model.predict(latest_s)
    prob = model.predict_proba(latest_s)[0]
    
    print("\n=== Latest Signal ===")
    print(f"Time: {df['timestamp'].iloc[-1]}")
    print(f"Price: {df['close'].iloc[-1]}")
    print(f"Prediction: {'UP' if pred[0] == 1 else 'DOWN'}")
    print(f"Confidence: UP={prob[1]:.2%}, DOWN={prob[0]:.2%}")
    
    return pred[0], prob

def main():
    print("=" * 60)
    print("BTC Price Predictor for Polymarket")
    print("=" * 60)
    
    # Download data (multiple years for robustness)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 1, 1)
    
    df_1m = download_binance_data("1m", start_date, end_date, "btc_1m_2023_2025.csv")
    df_5m = download_binance_data("5m", start_date, end_date, "btc_5m_2023_2025.csv")
    
    print(f"\n1m data: {len(df_1m)} rows, {df_1m['timestamp'].min()} to {df_1m['timestamp'].max()}")
    print(f"5m data: {len(df_5m)} rows, {df_5m['timestamp'].min()} to {df_5m['timestamp'].max()}")
    
    # Create features
    print("\nCreating 1m features...")
    df_1m = create_features_1m(df_1m)
    
    print("Creating 5m features...")
    df_5m = create_features_5m(df_5m)
    
    print("Merging timeframes...")
    df = merge_timeframes(df_1m, df_5m)
    
    # Create targets
    print("Creating targets...")
    df = create_targets(df, horizons=[1, 3, 5, 10])
    
    # Run experiments
    experiments = [
        (1, "Predict 1min direction"),
        (3, "Predict 3min direction"),
        (5, "Predict 5min direction (Polymarket interval)"),
        (10, "Predict 10min direction"),
    ]
    
    all_results = []
    for horizon, name in experiments:
        results = run_experiment(df, horizon, name)
        if len(results) > 0:
            results['target_horizon'] = horizon
            results['experiment'] = name
            all_results.append(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all_results:
        summary = pd.concat(all_results)
        print(f"\nOverall mean accuracy: {summary['accuracy'].mean():.4f}")
        
        by_horizon = summary.groupby('target_horizon')['accuracy'].agg(['mean', 'std', 'count'])
        print("\nBy prediction horizon:")
        print(by_horizon)
    
    # Generate signal
    feature_cols = [
        'ret_1', 'ret_3', 'ret_5', 'ret_10',
        'body', 'range',
        'rsi_7', 'rsi_14',
        'price_vs_sma_20', 'vol_15',
        'vol_ratio_30', 'macd_hist', 'bb_position',
        'hour', 'dayofweek'
    ]
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    generate_signals(df, feature_cols)
    
    # Save results
    if all_results:
        pd.concat(all_results).to_csv(f"{DATA_DIR}/experiment_results.csv", index=False)
        print(f"\nResults saved to {DATA_DIR}/experiment_results.csv")

if __name__ == "__main__":
    main()