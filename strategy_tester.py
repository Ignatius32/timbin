#!/usr/bin/env python3
"""
BTC 5-minute Strategy Tester
Tests various strategies to find profitable prediction for 5-min intervals (11:00 to 11:05 etc)
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

def add_basic_features(df):
    """Add basic features for strategy testing"""
    df = df.copy()
    
    # Return over this 5-min bar (the target)
    df['return_5'] = df['close'].pct_change(5)
    df['direction'] = (df['return_5'] > 0).astype(int)
    
    # Current bar info
    df['body'] = (df['close'] - df['open']) / df['open']
    df['range'] = (df['high'] - df['low']) / df['close']
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_wick'] = ((df[['open', 'close']].min(axis=1)) - df['low']) / df['close']
    df['is_green'] = (df['close'] > df['open']).astype(int)
    
    # Prior returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'ret_lag_{lag}'] = df['close'].pct_change(lag)
    
    # Streaks (consecutive green/red)
    df['green_streak'] = df['is_green']
    df['green_streak'] = df['green_streak'].groupby(
        (df['green_streak'] != df['green_streak'].shift()).cumsum()
    ).cumcount()
    df['red_streak'] = (1 - df['is_green'])
    df['red_streak'] = df['red_streak'].groupby(
        (df['red_streak'] != df['red_streak'].shift()).cumsum()
    ).cumcount()
    
    # RSI
    for period in [7, 14]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        df[f'rsi_{period}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        sma = df['close'].rolling(period).mean()
        df[f'vsma_{period}'] = (df['close'] - sma) / sma
    
    # Volatility
    for window in [5, 10, 20]:
        df[f'vol_{window}'] = df['return_5'].shift(1).rolling(window).std()
    
    # Volume
    df['vol_ma_10'] = df['volume'].rolling(10).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma_10'] + 1e-10)
    
    # Position in daily range
    df['daily_high'] = df['close'].rolling(288).max()  # 24h * 12 bars/h = 288 bars/day
    df['daily_low'] = df['close'].rolling(288).min()
    df['daily_position'] = (df['close'] - df['daily_low']) / (df['daily_high'] - df['daily_low'] + 1e-10)
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # Hour start marker (first bar of hour)
    df['is_hour_start'] = (df['minute'] == 0).astype(int)
    df['is_hour_end'] = (df['minute'] == 55).astype(int)
    
    return df

def test_strategy(df, name, signal_func):
    """Test a single strategy"""
    df = df.copy()
    df['signal'] = signal_func(df)
    
    valid = df.dropna(subset=['direction', 'signal']).copy()
    
    if valid.empty:
        return None
    
    trades = valid[valid['signal'] != 0]
    
    if len(trades) == 0:
        return None
    
    correct = (trades['signal'] == 1) & (trades['direction'] == 1)
    correct += (trades['signal'] == -1) & (trades['direction'] == 0)
    accuracy = correct.sum() / len(trades)
    
    baseline = trades['direction'].mean()
    
    return {
        'strategy': name,
        'accuracy': accuracy,
        'baseline': baseline,
        'n_trades': len(trades),
        'beats_baseline': accuracy > baseline
    }

def walk_forward_test_strategies(df, strategy_funcs, n_train_days=30, n_test_days=7):
    """Proper walk-forward validation"""
    df = df.dropna(subset=['direction']).copy()
    df = df.reset_index(drop=True)
    
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    
    results = []
    
    current_train_end = min_date + pd.Timedelta(days=n_train_days)
    
    while current_train_end + pd.Timedelta(days=n_test_days) <= max_date:
        test_start = current_train_end
        test_end = test_start + pd.Timedelta(days=n_test_days)
        
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] < test_end)
        test_df = df[test_mask]
        
        if len(test_df) < 500:
            current_train_end += pd.Timedelta(days=n_test_days)
            continue
        
        for name, signal_func in strategy_funcs:
            test_df = test_df.copy()
            test_df['signal'] = signal_func(test_df)
            
            valid = test_df.dropna(subset=['direction', 'signal'])
            if len(valid) < 50:
                continue
            
            trades = valid[valid['signal'] != 0]
            if len(trades) < 20:
                continue
            
            correct = (trades['signal'] == 1) & (trades['direction'] == 1)
            correct += (trades['signal'] == -1) & (trades['direction'] == 0)
            accuracy = correct.sum() / len(trades)
            
            results.append({
                'period_start': test_start,
                'period_end': test_end,
                'strategy': name,
                'accuracy': accuracy,
                'n_trades': len(trades)
            })
        
        current_train_end += pd.Timedelta(days=n_test_days)
    
    return pd.DataFrame(results)

def define_strategies():
    """Define all strategies to test"""
    strategies = []
    
    # 1. Simple momentum - if last 5min was up, bet up again
    strategies.append(("momentum_5", lambda df: (df['ret_lag_5'] > 0).astype(int) - (df['ret_lag_5'] < 0).astype(int)))
    
    # 2. Momentum 10min
    strategies.append(("momentum_10", lambda df: (df['ret_lag_10'] > 0).astype(int) - (df['ret_lag_10'] < 0).astype(int)))
    
    # 3. RSI oversold - bet up
    strategies.append(("rsi_oversold_7", lambda df: (df['rsi_7'] < 30).astype(int) - (df['rsi_7'] > 70).astype(int)))
    strategies.append(("rsi_oversold_14", lambda df: (df['rsi_14'] < 30).astype(int) - (df['rsi_14'] > 70).astype(int)))
    
    # 4. Price vs SMA
    strategies.append(("vsma_10_bullish", lambda df: (df['vsma_10'] > 0).astype(int) - (df['vsma_10'] < 0).astype(int)))
    strategies.append(("vsma_20_bullish", lambda df: (df['vsma_20'] > 0).astype(int) - (df['vsma_20'] < 0).astype(int)))
    strategies.append(("vsma_50_bullish", lambda df: (df['vsma_20'] > 0).astype(int) - (df['vsma_20'] < 0).astype(int)))
    
    # 5. Green streak continued
    strategies.append(("green_streak_up", lambda df: (df['green_streak'] >= 3).astype(int)))
    strategies.append(("red_streak_down", lambda df: (df['red_streak'] >= 3).astype(int) * -1))
    
    # 6. Low volatility
    strategies.append(("low_vol_up", lambda df: (df['vol_10'] < df['vol_10'].quantile(0.3)).astype(int)))
    
    # 7. Mean reversion - price below SMA
    strategies.append(("mean_reversion", lambda df: (df['vsma_10'] < -0.01).astype(int) - (df['vsma_10'] > 0.01).astype(int)))
    
    # 8. Hour start effect (11:00, 12:00 etc)
    strategies.append(("hour_start_up", lambda df: df['is_hour_start'].shift(1)))  # bet up at hour start
    
    # 9. Daily position extremes
    strategies.append(("daily_low_reversal", lambda df: (df['daily_position'] < 0.1).astype(int)))
    strategies.append(("daily_high_reversal", lambda df: (df['daily_position'] > 0.9).astype(int) * -1))
    
    # 10. Volume spike
    strategies.append(("vol_spike_up", lambda df: (df['vol_ratio'] > 2).astype(int)))
    
    return strategies

def main():
    print("=" * 60)
    print("BTC 5-min Strategy Tester")
    print("=" * 60 + "\n")
    
    # Load data
    df = load_5m_data()
    print(f"Loaded {len(df)} bars")
    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
    
    # Add features
    print("Adding features...")
    df = add_basic_features(df)
    
    # Get strategies
    strategies = define_strategies()
    
    print("\n=== Testing Each Strategy ===\n")
    
    all_results = []
    for name, signal_func in strategies:
        result = test_strategy(df, name, signal_func)
        if result:
            all_results.append(result)
            beats = "✓" if result['beats_baseline'] else " "
            print(f"{beats} {name}: Acc={result['accuracy']:.3f} (baseline={result['baseline']:.3f}, n={result['n_trades']})")
    
    # Sort by accuracy
    all_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
    
    print("\n=== Top Results ===\n")
    for r in all_results[:10]:
        beats = "✓" if r['beats_baseline'] else " "
        print(f"{beats} {r['strategy']}: {r['accuracy']:.3f}")
    
    # Walk-forward test best strategies
    print("\n=== Walk-Forward Validation ===\n")
    
    # Take top 5 strategies for walk-forward
    top_strategies = [(r['strategy'], 
                      [s[1] for s in strategies if s[0] == r['strategy']][0]) 
                     for r in all_results[:5]]
    
    wf_results = walk_forward_test_strategies(df, top_strategies, n_train_days=30, n_test_days=7)
    
    if not wf_results.empty:
        by_strategy = wf_results.groupby('strategy')['accuracy'].agg(['mean', 'std', 'count'])
        print(by_strategy)
        
        # Save results
        wf_results.to_csv(f"{DATA_DIR}/strategy_results.csv", index=False)

if __name__ == "__main__":
    main()