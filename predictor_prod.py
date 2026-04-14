#!/usr/bin/env python3
"""
BTC Polymarket Predictor - Production Version
- Proper walk-forward validation (no lookahead bias)
- Multiple timeframes
- Real-time signals
- Performance tracking
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

def get_1m_data():
    """Get 1-minute BTC data"""
    f = f"{DATA_DIR}/btc_1m_full.csv"
    if os.path.exists(f):
        print("Loading 1m data...")
        return pd.read_csv(f, parse_dates=['timestamp'])
    
    print("Downloading 1m data...")
    d = []
    start = int(datetime(2023, 1, 1).timestamp() * 1000)
    end = int(datetime(2025, 1, 1).timestamp() * 1000)
    url = "https://api.binance.com/api/v3/klines"
    
    while start < end:
        r = requests.get(url, params={"symbol": "BTCUSDT", "interval": "1m", "startTime": start, "endTime": end, "limit": 1000}).json()
        if not r: break
        d.extend(r)
        start = r[-1][0] + 1
        if len(d) % 100000 == 0: print(f"  {len(d)}")
        if len(d) >= 700000: break
    
    df = pd.DataFrame(d[:700000])
    df = df.iloc[:, :6]
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.to_csv(f, index=False)
    print(f"Saved {len(df)} rows")
    return df.dropna()

def make_features(df):
    """Create all features"""
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Returns
    for lag in [1, 3, 5, 10, 15, 30]:
        df[f'ret_{lag}'] = df['close'].pct_change(lag)
    
    # Candle features
    df['body'] = (df['close'] - df['open']) / df['close']
    df['range'] = (df['high'] - df['low']) / df['close']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_shadow'] = ((df[['open', 'close']].min(axis=1)) - df['low']) / df['close']
    
    # RSI (multiple periods)
    for period in [7, 14, 30]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        df[f'rsi_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        sma = df['close'].rolling(period).mean()
        df[f'vsma_{period}'] = (df['close'] - sma) / sma
        df[f'sma_slope_{period}'] = sma.pct_change(3)
    
    # Volatility
    for window in [5, 15, 30]:
        df[f'vol_{window}'] = df['ret_1'].rolling(window).std()
    
    # Volume
    df['vma_10'] = df['volume'].rolling(10).mean()
    df['vma_30'] = df['volume'].rolling(30).mean()
    df['vol_ratio'] = df['volume'] / (df['vma_30'] + 1e-10)
    df['vol_change'] = df['volume'].pct_change()
    
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
    df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # Lags
    for lag in [1, 2, 3, 5]:
        df[f'ret_lag_{lag}'] = df['ret_1'].shift(lag)
    
    # Streaks
    df['up_streak'] = (df['ret_1'] > 0).astype(int)
    df['up_streak'] = df['up_streak'].groupby((df['up_streak'] != df['up_streak'].shift()).cumsum()).cumcount()
    df['down_streak'] = (df['ret_1'] < 0).astype(int)
    df['down_streak'] = df['down_streak'].groupby((df['down_streak'] != df['down_streak'].shift()).cumsum()).cumcount()
    
    return df

FEATURES = [
    'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_15',
    'body', 'range', 'upper_shadow', 'lower_shadow',
    'rsi_7', 'rsi_14', 'rsi_30',
    'vsma_5', 'vsma_20', 'vsma_50', 'sma_slope_20',
    'vol_5', 'vol_15', 'vol_30',
    'vol_ratio', 'vol_change',
    'macd', 'macd_hist', 'bb_pos',
    'hour', 'dayofweek',
    'ret_lag_1', 'ret_lag_2', 'ret_lag_3',
    'up_streak', 'down_streak'
]

def walk_forward_test(df, target_col, n_periods=10):
    """
    Proper walk-forward: train on past, test on future
    No data leakage - each test period is completely unseen during training
    """
    df = df.dropna(subset=FEATURES + [target_col]).copy()
    df = df.reset_index(drop=True)
    
    # Replace inf
    for f in FEATURES:
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES)
    
    n = len(df)
    dates = df['timestamp']
    min_date = dates.min()
    max_date = dates.max()
    total_days = (max_date - min_date).days
    
    # Each period: 60 days train, 14 days test
    train_days = 60
    test_days = 14
    
    results = []
    
    for i in range(n_periods):
        train_end_day = (i + 1) * (total_days // (n_periods + 2)) + train_days
        test_start_day = train_end_day
        test_end_day = test_start_day + test_days
        
        train_end = min_date + pd.Timedelta(days=train_end_day)
        test_start = min_date + pd.Timedelta(days=test_start_day)
        test_end = min_date + pd.Timedelta(days=test_end_day)
        
        train_mask = dates < train_end
        test_mask = (dates >= test_start) & (dates < test_end)
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) < 10000 or len(test_df) < 1000:
            continue
        
        X_train = train_df[FEATURES].values
        y_train = train_df[target_col].values
        X_test = test_df[FEATURES].values
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
        y_prob = model.predict_proba(X_test_s)[:, 1]
        
        accuracy = (y_pred == y_test).mean()
        baseline = y_test.mean()
        
        # Average probability when UP
        avg_prob_up = y_prob.mean()
        
        results.append({
            'period': i + 1,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'n_train': len(train_df),
            'n_test': len(test_df),
            'accuracy': accuracy,
            'baseline': baseline,
            'avg_prob_up': avg_prob_up
        })
        
        print(f"Period {i+1}: {test_start.date()} - {test_end.date()} | Acc: {accuracy:.4f} | Baseline: {baseline:.4f}")
    
    return pd.DataFrame(results)

def train_final_model(df, target_col):
    """Train final model on all data for real-time predictions"""
    df = df.dropna(subset=FEATURES + [target_col]).copy()
    
    for f in FEATURES:
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES)
    
    X = df[FEATURES].values
    y = df[target_col].values
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.7,
        random_state=42
    )
    model.fit(X_s, y)
    
    return model, scaler

def generate_signal(df, model, scaler):
    """Generate current prediction signal"""
    latest = df.tail(1).copy()
    
    for f in FEATURES:
        latest[f] = latest[f].replace([np.inf, -np.inf], np.nan)
    
    if latest[FEATURES].isna().any().any():
        print("Warning: Missing features in latest data")
        return None
    
    X = latest[FEATURES].values
    X_s = scaler.transform(X)
    
    pred = model.predict(X_s)[0]
    prob = model.predict_proba(X_s)[0]
    
    return {
        'timestamp': latest['timestamp'].values[0],
        'price': latest['close'].values[0],
        'prediction': 'UP' if pred == 1 else 'DOWN',
        'confidence_up': prob[1],
        'confidence_down': prob[0]
    }

def main():
    print("=" * 60)
    print("BTC Polymarket Predictor - Production")
    print("=" * 60 + "\n")
    
    # Load data
    df = get_1m_data()
    print(f"Data: {len(df)} candles, {df['timestamp'].min()} to {df['timestamp'].max()}\n")
    
    # Create features
    print("Creating features...")
    df = make_features(df)
    print(f"Features created. Shape: {df.shape}\n")
    
    # Target: predict 5-minute direction (matches Polymarket)
    target_col = 'target_5'
    df[target_col] = (df['close'].shift(-5) > df['close']).astype(int)
    
    # Walk-forward validation
    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)
    print("Testing: predict UP in next 5 minutes\n")
    
    results = walk_forward_test(df, target_col, n_periods=12)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Test periods: {len(results)}")
    print(f"Mean accuracy: {results['accuracy'].mean():.4f} ({results['accuracy'].mean()*100:.2f}%)")
    print(f"Std accuracy: {results['accuracy'].std():.4f}")
    print(f"Min: {results['accuracy'].min():.4f}, Max: {results['accuracy'].max():.4f}")
    print(f"Mean baseline: {results['baseline'].mean():.4f}")
    
    beats_baseline = (results['accuracy'] > results['baseline']).sum()
    print(f"Periods beating baseline: {beats_baseline}/{len(results)} ({beats_baseline/len(results)*100:.1f}%)")
    
    # Train final model
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)
    
    model, scaler = train_final_model(df, target_col)
    print("Model trained on all data\n")
    
    # Generate signal
    print("=" * 60)
    print("CURRENT SIGNAL")
    print("=" * 60)
    
    signal = generate_signal(df, model, scaler)
    if signal:
        print(f"Time: {signal['timestamp']}")
        print(f"Price: ${signal['price']:,.2f}")
        print(f"Prediction: {signal['prediction']}")
        print(f"Confidence: UP {signal['confidence_up']:.2%}, DOWN {signal['confidence_down']:.2%}")
        
        # Expected value calculation
        # Polymarket typically: odds ~ 0.96 for each side (2% fee)
        polymarket_fee = 0.02
        fair_odds = 0.5
        actual_odds = 1 - polymarket_fee
        
        # If confidence > 52%, might have +EV
        if signal['confidence_up'] > 0.52:
            print(f"\n*** POTENTIAL +EV: Confidence {signal['confidence_up']:.2%} > 52% ***")
        elif signal['confidence_down'] > 0.52:
            print(f"\n*** POTENTIAL +EV: Confidence {signal['confidence_down']:.2%} > 52% ***")
    
    # Save results
    results.to_csv(f"{DATA_DIR}/walkforward_results.csv", index=False)
    print(f"\nResults saved to {DATA_DIR}/walkforward_results.csv")

if __name__ == "__main__":
    main()