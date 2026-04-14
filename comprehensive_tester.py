#!/usr/bin/env python3
"""
BTC Polymarket - Comprehensive Strategy Testing
Tests more features, models, and combinations
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"

def load_data():
    df = pd.read_csv(f"{DATA_DIR}/btc_5m_2023_2025.csv", parse_dates=['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)

def add_features(df):
    df = df.copy()
    
    # Target: next bar UP
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Returns
    for lag in [1, 2, 3, 5, 10, 15]:
        df[f'ret_{lag}'] = df['close'].pct_change(lag).shift(1)
    
    # Candle features
    df['body'] = (df['close'] - df['open']) / df['open']
    df['range'] = (df['high'] - df['low']) / df['close']
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_wick'] = ((df[['open', 'close']].min(axis=1)) - df['low']) / df['close']
    df['is_green'] = (df['close'] > df['open']).astype(int)
    
    # RSI multiple periods
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        df[f'rsi_{period}'] = (100 - (100 / (1 + gain / (loss + 1e-10)))).shift(1)
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        sma = df['close'].rolling(period).mean().shift(1)
        df[f'vsma_{period}'] = (df['close'] - sma) / sma
        df[f'sma_slope_{period}'] = sma.pct_change(3)
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean().shift(1)
    ema26 = df['close'].ewm(span=26, adjust=False).mean().shift(1)
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger
    bb_mid = df['close'].rolling(20).mean().shift(1)
    bb_std = df['close'].rolling(20).std().shift(1)
    df['bb_position'] = (df['close'] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    
    # Volatility
    df['volatility'] = df['ret_1'].rolling(10).std()
    df['volatility_20'] = df['ret_1'].rolling(20).std()
    df['vol_change'] = df['volatility'].pct_change(3)
    
    # Volume
    df['vol_ratio'] = (df['volume'] / df['volume'].rolling(10).mean()).shift(1)
    df['vol_ratio_20'] = (df['volume'] / df['volume'].rolling(20).mean()).shift(1)
    df['vol_change'] = df['volume'].pct_change()
    
    # Streaks
    df['green_streak'] = df['is_green']
    df['green_streak'] = df['green_streak'].groupby(
        (df['green_streak'] != df['green_streak'].shift()).cumsum()
    ).cumcount()
    df['red_streak'] = (1 - df['is_green'])
    df['red_streak'] = df['red_streak'].groupby(
        (df['red_streak'] != df['red_streak'].shift()).cumsum()
    ).cumcount()
    
    # High/Low position
    for period in [10, 20]:
        roll_high = df['high'].rolling(period).max().shift(1)
        roll_low = df['low'].rolling(period).min().shift(1)
        df[f'hl_pos_{period}'] = (df['close'] - roll_low) / (roll_high - roll_low + 1e-10)
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_hour_start'] = (df['minute'] == 0).astype(int)
    df['is_hour_end'] = (df['minute'] == 55).astype(int)
    
    return df

def test_model(df, features, model_type, n_periods=15):
    df = df.dropna(subset=features + ['target']).reset_index(drop=True)
    
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
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features].values)
        X_test = scaler.transform(test_df[features].values)
        
        if model_type == 'gb':
            model = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.7, random_state=42)
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000)
        
        model.fit(X_train, train_df['target'].values)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = (y_pred == test_df['target'].values).mean()
        baseline = test_df['target'].mean()
        
        # Confident only (>55%)
        conf_mask = (y_prob > 0.55) | (y_prob < 0.45)
        if conf_mask.sum() > 100:
            conf_acc = (y_pred[conf_mask] == test_df['target'].values[conf_mask]).mean()
        else:
            conf_acc = acc
        
        results.append({
            'period': test_start,
            'acc': acc,
            'baseline': baseline,
            'confident_acc': conf_acc
        })
        
        test_start = test_end
    
    return pd.DataFrame(results)

def main():
    print("=" * 60)
    print("BTC Polymarket - Comprehensive Strategy Test")
    print("=" * 60 + "\n")
    
    df = load_data()
    print(f"Data: {len(df)} bars\n")
    
    print("Adding features...")
    df = add_features(df)
    
    # Test different feature sets
    feature_sets = {
        'minimal': ['ret_1', 'ret_2', 'ret_3', 'body', 'rsi_7'],
        'basic': ['ret_1', 'ret_2', 'ret_3', 'body', 'rsi_7', 'rsi_14', 'vsma_10', 'volatility'],
        'extended': ['ret_1', 'ret_2', 'ret_3', 'ret_5', 'body', 'range', 'rsi_7', 'rsi_14', 'vsma_5', 'vsma_10', 'vsma_20', 'volatility', 'vol_ratio', 'green_streak', 'hour', 'dayofweek'],
        'full': ['ret_1', 'ret_2', 'ret_3', 'ret_5', 'ret_10', 'body', 'range', 'upper_wick', 'lower_wick', 'rsi_7', 'rsi_14', 'rsi_21', 'vsma_5', 'vsma_10', 'vsma_20', 'vsma_50', 'sma_slope_10', 'macd', 'macd_hist', 'bb_position', 'volatility', 'volatility_20', 'vol_ratio', 'green_streak', 'red_streak', 'hl_pos_10', 'hl_pos_20', 'hour', 'dayofweek', 'is_hour_start', 'is_hour_end']
    }
    
    models = ['gb', 'rf', 'lr']
    
    print("\nTesting feature sets + models:\n")
    
    all_results = []
    
    for feat_name, features in feature_sets.items():
        features = [f for f in features if f in df.columns]
        
        for model_type in models:
            results = test_model(df, features, model_type, n_periods=12)
            
            if not results.empty:
                acc = results['acc'].mean()
                base = results['baseline'].mean()
                conf_acc = results['confident_acc'].mean()
                edge = acc - base
                
                all_results.append({
                    'features': feat_name,
                    'model': model_type,
                    'accuracy': acc,
                    'baseline': base,
                    'confident_acc': conf_acc,
                    'edge': edge
                })
                
                print(f"{feat_name:>10} + {model_type:>3}: acc={acc:.3f}, base={base:.3f}, edge={edge:+.3f}, conf={conf_acc:.3f}")
    
    # Best combination
    best = max(all_results, key=lambda x: x['edge'])
    print(f"\n*** Best: {best['features']} + {best['model']} with {best['edge']:+.3f} edge ***")

if __name__ == "__main__":
    main()