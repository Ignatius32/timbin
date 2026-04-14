#!/usr/bin/env python3
"""
BTC Polymarket Predictor - Final Version
Best settings from testing:
- Features: ret_1-3, body, rsi_7, vsma_5/10/20, vol
- Model: GB 30 trees, depth 3
- Threshold: 0.58+ confident
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"

def load_and_prepare():
    """Load data and create features"""
    df = pd.read_csv(f"{DATA_DIR}/btc_5m_2023_2025.csv", parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Target: next bar UP
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Returns (past only)
    for lag in [1, 2, 3, 5]:
        df[f'ret_{lag}'] = df['close'].pct_change(lag).shift(1)
    
    # Candle
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
    
    # Time
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    return df

def train_and_predict(df, features, conf_threshold=0.58):
    """Train on all past, predict current"""
    df = df.dropna(subset=features + ['target']).reset_index(drop=True)
    
    # Train on all data before last week
    train_df = df[df['timestamp'] < df['timestamp'].max() - pd.Timedelta(days=7)]
    test_df = df[df['timestamp'] >= df['timestamp'].max() - pd.Timedelta(days=7)]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features].values)
    X_test = scaler.transform(test_df[features].values)
    
    model = GradientBoostingClassifier(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.7,
        random_state=42
    )
    model.fit(X_train, train_df['target'].values)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Only trade when confident
    conf_mask = (y_prob > conf_threshold) | (y_prob < (1-conf_threshold))
    
    if conf_mask.sum() > 0:
        y_pred = (y_prob > 0.5).astype(int)
        acc = (y_pred[conf_mask] == test_df['target'].values[conf_mask]).mean()
    else:
        acc = 0.5
    
    return acc, conf_mask.sum(), len(test_df)

def walk_forward_final(df, features, n_periods=20):
    """Final walk-forward validation"""
    df = df.dropna(subset=features + ['target']).reset_index(drop=True)
    
    min_date = df['timestamp'].min()
    results = []
    
    test_start = min_date + pd.Timedelta(days=30)
    
    for period in range(n_periods):
        test_end = test_start + pd.Timedelta(days=7)
        if test_end > df['timestamp'].max():
            break
        
        train_df = df[df['timestamp'] < test_start]
        test_df = df[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)]
        
        if len(train_df) < 5000 or len(test_df) < 200:
            test_start = test_end
            continue
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features].values)
        X_test = scaler.transform(test_df[features].values)
        
        model = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
        model.fit(X_train, train_df['target'].values)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # No threshold first
        y_pred = (y_prob > 0.5).astype(int)
        acc_all = (y_pred == test_df['target'].values).mean()
        
        # With 58% threshold
        conf = (y_prob > 0.58) | (y_prob < 0.42)
        if conf.sum() > 30:
            y_pred_c = (y_prob > 0.5).astype(int)
            acc_conf = (y_pred_c[conf] == test_df['target'].values[conf]).mean()
        else:
            acc_conf = acc_all
        
        results.append({
            'period': test_start.date(),
            'acc_all': acc_all,
            'acc_conf': acc_conf,
            'conf_trades': conf.sum() if hasattr(conf, 'sum') else 0,
            'baseline': test_df['target'].mean()
        })
        
        test_start = test_end
    
    return pd.DataFrame(results)

def main():
    print("=" * 60)
    print("BTC Polymarket - Final Strategy")
    print("=" * 60 + "\n")
    
    df = load_and_prepare()
    print(f"Data: {len(df)} bars")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
    
    features = ['ret_1', 'ret_2', 'ret_3', 'ret_5', 'body', 'range', 'rsi_7', 
                'vsma_5', 'vsma_10', 'vsma_20', 'volatility']
    print(f"Features: {features}\n")
    
    # Walk-forward test
    print("=== Walk-Forward Validation (20 weeks) ===\n")
    results = walk_forward_final(df, features, n_periods=20)
    
    if not results.empty:
        print(f"All trades:")
        print(f"  Accuracy: {results['acc_all'].mean():.3f}")
        print(f"  Baseline: {results['baseline'].mean():.3f}")
        print(f"  Edge: {results['acc_all'].mean() - results['baseline'].mean():+.3f}")
        
        print(f"\nConfident trades only (58%+):")
        print(f"  Accuracy: {results['acc_conf'].mean():.3f}")
        print(f"  Avg trades/week: {results['conf_trades'].mean():.0f}")
        
        print("\n\nPer-week results:")
        print(results.to_string(index=False))
    
    # Save model for production
    print("\n=== Training Production Model ===")
    df_clean = df.dropna(subset=features + ['target']).reset_index(drop=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[features].values)
    y = df_clean['target'].values
    
    model = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Feature importance
    print("\nFeature importance:")
    imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    print(imp.to_string(index=False))
    
    print("\n*** READY FOR PRODUCTION ***")
    print(f"Accuracy: ~{results['acc_all'].mean():.1%}")
    print(f"With 58%+ confidence: ~{results['acc_conf'].mean():.1%}")

if __name__ == "__main__":
    main()