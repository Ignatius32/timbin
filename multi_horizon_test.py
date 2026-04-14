#!/usr/bin/env python3
"""
BTC Polymarket Predictor - Multi-horizon test
Tests prediction at different horizons (1 bar, 2 bars, etc.)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"

def load_data():
    df = pd.read_csv(f"{DATA_DIR}/btc_5m_2023_2025.csv", parse_dates=['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)

def add_features(df, horizon=1):
    """Add features for specific prediction horizon"""
    df = df.copy()
    
    # Target: UP at horizon (shift by horizon)
    df['target'] = (df['close'].shift(-horizon) > df['close']).astype(int)
    
    # Features (past only)
    for lag in [1, 2, 3, 5, 10]:
        df[f'ret_{lag}'] = df['close'].pct_change(lag).shift(1)
    
    df['body'] = (df['close'] - df['open']) / df['open']
    df['range'] = (df['high'] - df['low']) / df['close']
    df['is_green'] = (df['close'] > df['open']).astype(int)
    
    for period in [7, 14]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        df[f'rsi_{period}'] = (100 - (100 / (1 + gain / (loss + 1e-10)))).shift(1)
    
    for period in [5, 10, 20]:
        sma = df['close'].rolling(period).mean().shift(1)
        df[f'vsma_{period}'] = (df['close'] - sma) / sma
    
    df['volatility'] = df['ret_1'].rolling(10).std()
    df['vol_ratio'] = (df['volume'] / df['volume'].rolling(10).mean()).shift(1)
    
    df['green_streak'] = df['is_green']
    df['green_streak'] = df['green_streak'].groupby(
        (df['green_streak'] != df['green_streak'].shift()).cumsum()
    ).cumcount()
    
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    return df

def walk_forward_test(df, feature_cols, n_periods=30):
    df = df.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
    
    min_date = df['timestamp'].min()
    results = []
    
    test_start = min_date + pd.Timedelta(days=30)
    
    for period in range(n_periods):
        test_end = test_start + pd.Timedelta(days=7)
        if test_end > df['timestamp'].max():
            break
        
        train_mask = df['timestamp'] < test_start
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) < 10000 or len(test_df) < 500:
            test_start = test_end
            continue
        
        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.7, random_state=42
        )
        model.fit(X_train_s, y_train)
        
        y_pred = model.predict(X_test_s)
        acc = (y_pred == y_test).mean()
        baseline = y_test.mean()
        
        results.append({
            'period': period + 1,
            'accuracy': acc,
            'baseline': baseline
        })
        
        test_start = test_end
    
    return pd.DataFrame(results)

def main():
    print("=" * 60)
    print("BTC Polymarket - Multi-Horizon Test")
    print("=" * 60 + "\n")
    
    df = load_data()
    print(f"Data: {len(df)} bars, {df['timestamp'].min()} to {df['timestamp'].max()}\n")
    
    feature_cols = [
        'ret_1', 'ret_2', 'ret_3', 'ret_5',
        'body', 'range',
        'rsi_7', 'rsi_14',
        'vsma_5', 'vsma_10', 'vsma_20',
        'volatility', 'vol_ratio',
        'green_streak',
        'hour', 'dayofweek'
    ]
    
    print("Testing different prediction horizons:\n")
    print(f"{'Horizon':>10} | {'Accuracy':>10} | {'Baseline':>10} | {'Edge':>10}")
    print("-" * 45)
    
    all_results = []
    
    for horizon in [1, 2, 3, 5]:
        df_feat = add_features(df.copy(), horizon=horizon)
        results = walk_forward_test(df_feat, feature_cols, n_periods=25)
        
        if not results.empty:
            acc = results['accuracy'].mean()
            base = results['baseline'].mean()
            edge = acc - base
            print(f"{horizon:>10} bar | {acc:>10.3f} | {base:>10.3f} | {edge:>+10.3f}")
            
            all_results.append({
                'horizon': horizon,
                'accuracy': acc,
                'baseline': base,
                'edge': edge
            })
    
    # Best horizon
    best = max(all_results, key=lambda x: x['edge'])
    print(f"\n*** Best: {best['horizon']} bar horizon with {best['edge']:+.3f} edge ***")

if __name__ == "__main__":
    main()