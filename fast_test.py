#!/usr/bin/env python3
"""Fast BTC Strategy Test - Key combinations only"""

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
df['green_streak'] = df['is_green'] = (df['close'] > df['open']).astype(int)
df['green_streak'] = df['green_streak'].groupby((df['green_streak'] != df['green_streak'].shift()).cumsum()).cumcount()
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['minute'] = df['timestamp'].dt.minute
df['is_hour_start'] = (df['minute'] == 0).astype(int)

print(f"Data: {len(df)} bars")

# Feature sets to test
feature_sets = {
    'basic': ['ret_1', 'ret_2', 'ret_3', 'body', 'rsi_7'],
    'w_momentum': ['ret_1', 'ret_2', 'ret_3', 'ret_5', 'body', 'range', 'rsi_7'],
    'w_sma': ['ret_1', 'ret_2', 'body', 'rsi_7', 'vsma_5', 'vsma_10', 'vsma_20'],
    'w_vol': ['ret_1', 'ret_2', 'ret_3', 'body', 'rsi_7', 'volatility', 'vol_ratio'],
    'full': ['ret_1', 'ret_2', 'ret_3', 'ret_5', 'body', 'range', 'rsi_7', 'vsma_5', 'vsma_10', 'vsma_20', 'volatility', 'vol_ratio', 'green_streak', 'hour', 'dayofweek', 'is_hour_start']
}

min_date = df['timestamp'].min()
test_start = min_date + pd.Timedelta(days=30)

results = []

for feat_name, features in feature_sets.items():
    features = [f for f in features if f in df.columns]
    df_test = df.dropna(subset=features + ['target']).reset_index(drop=True)
    
    period_results = []
    test_start = min_date + pd.Timedelta(days=30)
    
    for period in range(10):
        test_end = test_start + pd.Timedelta(days=7)
        if test_end > df_test['timestamp'].max():
            break
        
        train_df = df_test[df_test['timestamp'] < test_start]
        test_df = df_test[(df_test['timestamp'] >= test_start) & (df_test['timestamp'] < test_end)]
        
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
        
        acc = (y_pred == test_df['target'].values).mean()
        baseline = test_df['target'].mean()
        
        # Confident trades
        conf_mask = (y_prob > 0.55) | (y_prob < 0.45)
        conf_acc = (y_pred[conf_mask] == test_df['target'].values[conf_mask]).mean() if conf_mask.sum() > 50 else acc
        
        period_results.append({'acc': acc, 'base': baseline, 'conf': conf_acc})
        test_start = test_end
    
    if period_results:
        r = pd.DataFrame(period_results)
        print(f"{feat_name:>12}: acc={r['acc'].mean():.3f}, base={r['base'].mean():.3f}, edge={r['acc'].mean()-r['base'].mean():+.3f}, conf={r['conf'].mean():.3f}")
        results.append({'name': feat_name, 'acc': r['acc'].mean(), 'base': r['base'].mean(), 'conf': r['conf'].mean()})

# Best
best = max(results, key=lambda x: x['acc'] - x['base'])
print(f"\n*** Best: {best['name']} with {best['acc']-best['base']:+.3f} edge ***")