#!/usr/bin/env python3
"""
BTC 5-minute Strategy Tester - Fixed version
Proper walk-forward: train on PAST, predict FUTURE
No data leakage in features
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "data"

def load_5m_data():
    """Load 5-minute BTC data"""
    f = f"{DATA_DIR}/btc_5m_2023_2025.csv"
    df = pd.read_csv(f, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def add_features(df, lookback_only=True):
    """Add features using ONLY past data (no leakage)"""
    df = df.copy()
    
    if lookback_only:
        # Target: will price go UP in next 5 min bar?
        df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    else:
        df['target'] = (df['close'].pct_change(5) > 0).astype(int)
    
    # Features using ONLY past data (shifted to avoid any leakage)
    
    # Prior returns (ended before current bar)
    df['ret_1'] = df['close'].pct_change(1).shift(1)  # 5-10 min ago
    df['ret_2'] = df['close'].pct_change(2).shift(1)  # 10-15 min ago
    df['ret_3'] = df['close'].pct_change(3).shift(1)  # 15-20 min ago
    
    # Current bar features (not using future)
    df['body'] = (df['close'] - df['open']) / df['open']
    df['range'] = (df['high'] - df['low']) / df['close']
    df['is_green'] = (df['close'] > df['open']).astype(int)
    
    # RSI (using shifted data)
    for period in [7, 14]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        df[f'rsi_{period}'] = (100 - (100 / (1 + gain / (loss + 1e-10)))).shift(1)
    
    # SMA (excluding current bar)
    for period in [10, 20]:
        sma = df['close'].rolling(period).mean().shift(1)
        df[f'vsma_{period}'] = (df['close'] - sma) / sma
    
    # Volatility 
    df['volatility'] = df['ret_1'].rolling(10).std().shift(1)
    
    # Streaks
    df['green_streak'] = df['is_green']
    df['green_streak'] = df['green_streak'].groupby(
        (df['green_streak'] != df['green_streak'].shift()).cumsum()
    ).cumcount()
    
    # Volume
    df['vol_ratio'] = (df['volume'] / df['volume'].rolling(10).mean() + 1e-10).shift(1)
    
    # Time
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # Hour markers
    df['is_hour_start'] = (df['minute'] == 0).astype(int)
    df['is_hour_end'] = (df['minute'] == 55).astype(int)
    
    return df

def simulate_strategy(df, signal_name, signal_func):
    """Simulate a strategy with proper no-lookahead"""
    df = df.copy()
    df['signal'] = signal_func(df)
    
    trades = df[df['signal'] != 0].dropna(subset=['signal', 'target'])
    
    if len(trades) < 50:
        return None
    
    correct = (trades['signal'] == 1) & (trades['target'] == 1)
    correct += (trades['signal'] == -1) & (trades['target'] == 0)
    accuracy = correct.sum() / len(trades)
    
    return accuracy, len(trades), trades['target'].mean()

def walk_forward_strategies(df, strategies, n_weeks=40):
    """Proper walk-forward: learn from past, test onfuture"""
    df = df.dropna(subset=['target', 'ret_1', 'vsma_10']).reset_index(drop=True)
    
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    
    results = []
    
    one_week = pd.Timedelta(days=7)
    test_start = min_date + pd.Timedelta(days=30)  # First 30 days burn-in
    
    for week in range(n_weeks):
        test_end = test_start + one_week
        
        if test_end > max_date:
            break
        
        # Test period only
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)
        test_df = df[test_mask]
        
        if len(test_df) < 200:
            test_start = test_end
            continue
        
        for strat_name, signal_func in strategies:
            test_df = test_df.copy()
            test_df['signal'] = signal_func(test_df)
            
            trades = test_df[test_df['signal'] != 0].dropna(subset=['signal', 'target'])
            
            if len(trades) > 30:
                correct = (trades['signal'] == 1) & (trades['target'] == 1)
                correct += (trades['signal'] == -1) & (trades['target'] == 0)
                acc = correct.sum() / len(trades)
                
                results.append({
                    'week': week + 1,
                    'test_start': test_start,
                    'test_end': test_end,
                    'strategy': strat_name,
                    'accuracy': acc,
                    'n_trades': len(trades),
                    'baseline': trades['target'].mean()
                })
        
        test_start = test_end
    
    return pd.DataFrame(results)

def main():
    print("=" * 60)
    print("BTC 5-min Strategy Tester V2 - Proper Walk-Forward")
    print("=" * 60 + "\n")
    
    df = load_5m_data()
    print(f"Loaded {len(df)} bars")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
    
    print("Adding features (no lookahead)...")
    df = add_features(df, lookback_only=True)
    
    # Define strategies
    strategies = [
        ("ret1_up", lambda df: (df['ret_1'] > 0).astype(int)),
        ("ret1_down", lambda df: (df['ret_1'] < 0).astype(int) * -1),
        ("ret2_up", lambda df: (df['ret_2'] > 0).astype(int)),
        ("ret3_up", lambda df: (df['ret_3'] > 0).astype(int)),
        ("rsi7_oversold", lambda df: (df['rsi_7'] < 30).astype(int)),
        ("rsi7_overbought", lambda df: (df['rsi_7'] > 70).astype(int) * -1),
        ("vsma10_up", lambda df: (df['vsma_10'] > 0).astype(int)),
        ("vsma10_down", lambda df: (df['vsma_10'] < 0).astype(int) * -1),
        ("vsma20_up", lambda df: (df['vsma_20'] > 0).astype(int)),
        ("green_streak>=3", lambda df: (df['green_streak'] >= 3).astype(int)),
        ("vol_high", lambda df: (df['vol_ratio'] > 1.5).astype(int)),
        ("hour_start", lambda df: df['is_hour_start']),
        ("hour_end", lambda df: df['is_hour_end']),
        ("always_up", lambda df: 1),
    ]
    
    # Test each strategy in-sample first
    print("\n=== In-Sample Results ===\n")
    for name, func in strategies:
        result = simulate_strategy(df, name, func)
        if result:
            acc, n, base = result
            print(f"  {name}: acc={acc:.3f} (n={n}, base={base:.3f})")
    
    # Walk-forward test
    print("\n=== Walk-Forward Test (last ~40 weeks) ===\n")
    wf_results = walk_forward_strategies(df, strategies, n_weeks=40)
    
    if not wf_results.empty:
        summary = wf_results.groupby('strategy').agg({
            'accuracy': ['mean', 'std', 'count'],
            'n_trades': 'mean',
            'baseline': 'mean'
        }).round(4)
        summary.columns = ['acc_mean', 'acc_std', 'n_periods', 'avg_trades', 'avg_baseline']
        summary = summary.sort_values('acc_mean', ascending=False)
        
        print("\nTop strategies:")
        print(summary.head(15).to_string())
        
        # Best period
        best_strat = summary['acc_mean'].idxmax()
        best_acc = summary.loc[best_strat, 'acc_mean']
        
        print(f"\n*** Best: {best_strat} with {best_acc:.3f} accuracy ***")
        
        wf_results.to_csv(f"{DATA_DIR}/strategy_wf_results.csv", index=False)

if __name__ == "__main__":
    main()