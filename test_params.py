#!/usr/bin/env python3
"""Test model parameters and thresholds"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
df = pd.read_csv(f"{DATA_DIR}/btc_5m_2023_2025.csv", parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Add features
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
for lag in [1, 2, 3, 5]:
    df[f'ret_{lag}'] = df['close'].pct_change(lag).shift(1)
df['body'] = (df['close'] - df['open']) / df['open']
df['range'] = (df['high'] - df['low']) / df['close']
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(7).mean()
loss = delta.where(delta < 0, 0).rolling(7).mean()
df['rsi_7'] = (100 - (100 / (1 + gain / (loss + 1e-10)))).shift(1)
for period in [5, 10, 20]:
    sma = df['close'].rolling(period).mean().shift(1)
    df[f'vsma_{period}'] = (df['close'] - sma) / sma
df['volatility'] = df['ret_1'].rolling(10).std()
df['vol_ratio'] = (df['volume'] / df['volume'].rolling(10).mean()).shift(1)
df['green_streak'] = (df['close'] > df['open']).astype(int)
df['green_streak'] = df['green_streak'].groupby((df['green_streak'] != df['green_streak'].shift()).cumsum()).cumcount()
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['minute'] = df['timestamp'].dt.minute
df['is_hour_start'] = (df['minute'] == 0).astype(int)

features = ['ret_1', 'ret_2', 'ret_3', 'body', 'rsi_7', 'vsma_5', 'vsma_10', 'vsma_20', 'volatility', 'vol_ratio', 'green_streak', 'hour', 'dayofweek', 'is_hour_start']

print(f"Data: {len(df)} bars, Features: {len(features)}")

# Test different model configs
configs = [
    {'n_estimators': 30, 'max_depth': 3, 'lr': 0.1, 'subsample': 0.7},
    {'n_estimators': 50, 'max_depth': 3, 'lr': 0.1, 'subsample': 0.7},
    {'n_estimators': 30, 'max_depth': 4, 'lr': 0.1, 'subsample': 0.7},
    {'n_estimators': 30, 'max_depth': 3, 'lr': 0.05, 'subsample': 0.8},
    {'n_estimators': 20, 'max_depth': 2, 'lr': 0.1, 'subsample': 0.7},
]

min_date = df['timestamp'].min()
df = df.dropna(subset=features + ['target']).reset_index(drop=True)

print(f"\nTesting model configs:\n")
print(f"{'Config':>25} | {'Acc':>6} | {'Base':>6} | {'Edge':>6} | {'Conf':>6}")
print("-" * 60)

results = []
for i, cfg in enumerate(configs):
    period_results = []
    test_start = min_date + pd.Timedelta(days=30)
    
    for period in range(10):
        test_end = test_start + pd.Timedelta(days=7)
        if test_end > df['timestamp'].max():
            break
        
        train_df = df[df['timestamp'] < test_start]
        test_df = df[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)]
        
        if len(train_df) < 5000 or len(test_df) < 300:
            test_start = test_end
            continue
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features].values)
        X_test = scaler.transform(test_df[features].values)
        
        model = GradientBoostingClassifier(
            n_estimators=cfg['n_estimators'],
            max_depth=cfg['max_depth'],
            learning_rate=cfg['lr'],
            subsample=cfg['subsample'],
            random_state=42
        )
        model.fit(X_train, train_df['target'].values)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = (y_pred == test_df['target'].values).mean()
        baseline = test_df['target'].mean()
        
        # Confident threshold 55%
        conf_mask = (y_prob > 0.55) | (y_prob < 0.45)
        conf_acc = (y_pred[conf_mask] == test_df['target'].values[conf_mask]).mean() if conf_mask.sum() > 50 else acc
        
        period_results.append({'acc': acc, 'base': baseline, 'conf': conf_acc})
        test_start = test_end
    
    if period_results:
        r = pd.DataFrame(period_results)
        cfg_name = f"e{cfg['n_estimators']}_d{cfg['max_depth']}_lr{cfg['lr']}"
        print(f"{cfg_name:>25} | {r['acc'].mean():.3f} | {r['base'].mean():.3f} | {r['acc'].mean()-r['base'].mean():+.3f} | {r['conf'].mean():.3f}")
        results.append({'cfg': cfg_name, 'acc': r['acc'].mean(), 'edge': r['acc'].mean()-r['base'].mean(), 'conf': r['conf'].mean()})

best = max(results, key=lambda x: x['edge'])
print(f"\n*** Best: {best['cfg']} edge={best['edge']:+.3f} conf={best['conf']:.3f} ***")

# Now test different confidence thresholds
print("\n\nTesting confidence thresholds:")
for thresh in [0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58]:
    period_results = []
    test_start = min_date + pd.Timedelta(days=30)
    
    for period in range(10):
        test_end = test_start + pd.Timedelta(days=7)
        if test_end > df['timestamp'].max():
            break
        
        train_df = df[df['timestamp'] < test_start]
        test_df = df[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)]
        
        if len(train_df) < 5000 or len(test_df) < 300:
            test_start = test_end
            continue
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features].values)
        X_test = scaler.transform(test_df[features].values)
        
        model = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
        model.fit(X_train, train_df['target'].values)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Apply threshold
        conf_mask = (y_prob > thresh) | (y_prob < (1-thresh))
        if conf_mask.sum() > 50:
            conf_acc = (y_pred[conf_mask] == test_df['target'].values[conf_mask]).mean()
            n_trades = conf_mask.sum()
        else:
            conf_acc = 0.5
            n_trades = 0
        
        period_results.append({'acc': conf_acc, 'n': n_trades})
        test_start = test_end
    
    if period_results:
        r = pd.DataFrame(period_results)
        print(f"Threshold {thresh}: acc={r['acc'].mean():.3f}, avg_trades={r['n'].mean():.0f}")