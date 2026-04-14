#!/usr/bin/env python3
"""
BTC 5-min ML Strategy Tester
Tests ML-based prediction vs baseline strategies
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"

def load_data():
    """Load 5m data"""
    df = pd.read_csv(f"{DATA_DIR}/btc_5m_2023_2025.csv", parse_dates=['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)

def add_features(df):
    """Add features using only past data (no leakage)"""
    df = df.copy()
    
    # Target: direction in NEXT 5-min bar (shift -1 to avoid current bar)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)  # Predict at bar N, result at N+1
    
    # Returns (past only)
    for lag in [1, 2, 3, 5, 10]:
        df[f'ret_{lag}'] = df['close'].pct_change(lag).shift(1)
    
    # Current candle (no future info)
    df['body'] = (df['close'] - df['open']) / df['open']
    df['range'] = (df['high'] - df['low']) / df['close']
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_wick'] = ((df[['open', 'close']].min(axis=1)) - df['low']) / df['close']
    df['is_green'] = (df['close'] > df['open']).astype(int)
    
    # RSI
    for period in [7, 14]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        df[f'rsi_{period}'] = (100 - (100 / (1 + gain / (loss + 1e-10)))).shift(1)
    
    # SMA
    for period in [5, 10, 20]:
        sma = df['close'].rolling(period).mean().shift(1)
        df[f'vsma_{period}'] = (df['close'] - sma) / sma
    
    # Volatility
    df['volatility'] = df['ret_1'].rolling(10).std()
    
    # Volume
    df['vol_ratio'] = (df['volume'] / df['volume'].rolling(10).mean()).shift(1)
    
    # Streaks
    df['green_streak'] = df['is_green']
    df['green_streak'] = df['green_streak'].groupby(
        (df['green_streak'] != df['green_streak'].shift()).cumsum()
    ).cumcount()
    
    # Time
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    return df

def ml_walk_forward(df, feature_cols, n_periods=30):
    """Proper ML walk-forward: train on past, test on future"""
    df = df.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
    
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    
    results = []
    
    # Each period: ~7 days test
    test_start = min_date + pd.Timedelta(days=30)
    
    for period in range(n_periods):
        test_end = test_start + pd.Timedelta(days=7)
        
        if test_end > max_date:
            break
        
        # Train on all data before test period
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
        
        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Train ML model
        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.7,
            random_state=42
        )
        model.fit(X_train_s, y_train)
        
        # Predict
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]
        
        acc = (y_pred == y_test).mean()
        baseline = y_test.mean()
        
        # Only trade when confident
        confident_trades = (y_prob > 0.55) | (y_prob < 0.45)
        if confident_trades.sum() > 100:
            conf_correct = (y_pred[confident_trades] == y_test[confident_trades])
            conf_acc = conf_correct.mean()
        else:
            conf_acc = acc
        
        results.append({
            'period': period + 1,
            'test_start': test_start,
            'test_end': test_end,
            'accuracy': acc,
            'baseline': baseline,
            'confident_accuracy': conf_acc,
            'n_trades': len(test_df),
            'confident_trades': confident_trades.sum()
        })
        
        test_start = test_end
    
    return pd.DataFrame(results)

def simple_baseline_walk_forward(df):
    """Test simple baselines"""
    df = df.dropna(subset=['target', 'ret_1']).reset_index(drop=True)
    
    min_date = df['timestamp'].min()
    test_start = min_date + pd.Timedelta(days=30)
    
    results = []
    
    for period in range(30):
        test_end = test_start + pd.Timedelta(days=7)
        
        if test_end > df['timestamp'].max():
            break
        
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)
        test_df = df[test_mask]
        
        if len(test_df) < 500:
            test_start = test_end
            continue
        
        baseline = test_df['target'].mean()
        
        results.append({
            'period': period + 1,
            'test_start': test_start,
            'baseline': baseline,
            'n_trades': len(test_df)
        })
        
        test_start = test_end
    
    return pd.DataFrame(results)

def main():
    print("=" * 60)
    print("BTC 5-min ML Strategy Tester")
    print("=" * 60 + "\n")
    
    df = load_data()
    print(f"Loaded {len(df)} bars")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
    
    print("Adding features...")
    df = add_features(df)
    
    feature_cols = [
        'ret_1', 'ret_2', 'ret_3', 'ret_5',
        'body', 'range',
        'rsi_7', 'rsi_14',
        'vsma_5', 'vsma_10', 'vsma_20',
        'volatility', 'vol_ratio',
        'green_streak',
        'hour', 'dayofweek'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"Features: {len(feature_cols)}\n")
    
    # Test baselines
    print("=== Testing Baselines ===\n")
    baselines = simple_baseline_walk_forward(df)
    if not baselines.empty:
        print(f"Always UP: {baselines['baseline'].mean():.3f}")
        print(f"Random: 0.500")
    
    # ML walk-forward
    print("\n=== ML Walk-Forward ===\n")
    results = ml_walk_forward(df, feature_cols, n_periods=30)
    
    if not results.empty:
        print(f"Periods tested: {len(results)}")
        print(f"Mean accuracy: {results['accuracy'].mean():.4f}")
        print(f"Std: {results['accuracy'].std():.4f}")
        print(f"Baseline mean: {results['baseline'].mean():.4f}")
        
        beats_baseline = (results['accuracy'] > results['baseline']).sum()
        print(f"\nBeating baseline: {beats_baseline}/{len(results)} ({beats_baseline/len(results)*100:.1f}%)")
        
        # Confident trades only
        print(f"\nWhen confident (>55% prob):")
        print(f"Mean accuracy: {results['confident_accuracy'].mean():.4f}")
        
        # Per period
        print("\nPer-period results:")
        for _, row in results.iterrows():
            print(f"  {row['test_start'].date()}: acc={row['accuracy']:.3f}, base={row['baseline']:.3f}")
    
    results.to_csv(f"{DATA_DIR}/ml_strategy_results.csv", index=False)
    print(f"\nResults saved to {DATA_DIR}/ml_strategy_results.csv")

if __name__ == "__main__":
    main()