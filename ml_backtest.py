import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

DATA_DIR = "data"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"

def download_binance_klines():
    csv_path = f"{DATA_DIR}/btc_1m.csv"
    if os.path.exists(csv_path):
        print("Loading data...")
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        return df
    
    print("Downloading 1m data...")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    
    start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
    end_ts = int(datetime(2024, 12, 31).timestamp() * 1000)
    
    while start_ts < end_ts:
        params = {"symbol": SYMBOL, "interval": INTERVAL, "startTime": start_ts, "endTime": end_ts, "limit": 1000}
        data = requests.get(url, params=params).json()
        if not data:
            break
        all_data.extend(data)
        start_ts = data[-1][0] + 1
        if len(all_data) % 50000 == 0:
            print(f"Downloaded {len(all_data)}...")
    
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
        'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.to_csv(csv_path, index=False)
    return df

def add_features(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df['returns_1'] = df['close'].pct_change(1)
    df['returns_3'] = df['close'].pct_change(3)
    df['returns_5'] = df['close'].pct_change(5)
    df['returns_10'] = df['close'].pct_change(10)
    df['returns_30'] = df['close'].pct_change(30)
    
    df['candle_body'] = (df['close'] - df['open']) / df['open']
    df['candle_range'] = (df['high'] - df['low']) / df['low']
    
    for window in [7, 14, 30]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        df[f'rsi_{window}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    for window in [5, 10, 20, 50, 100]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'price_vs_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']
    
    for window in [5, 15, 30]:
        df[f'volatility_{window}'] = df['returns_1'].rolling(window=window).std()
        df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].shift(1)
    
    df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
    df['volume_ma_60'] = df['volume'].rolling(window=60).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_60'] + 1e-10)
    
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    for window in [5, 10, 20]:
        roll_high = df['high'].rolling(window=window).max()
        roll_low = df['low'].rolling(window=window).min()
        df[f'high_low_pos_{window}'] = (df['close'] - roll_low) / (roll_high - roll_low + 1e-10)
    
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    for lag in [1, 2, 3, 5, 10]:
        df[f'returns_lag_{lag}'] = df['returns_1'].shift(lag)
    
    return df

def ml_backtest(df):
    feature_cols = [
        'returns_1', 'returns_3', 'returns_5', 'returns_10', 'returns_30',
        'candle_body', 'candle_range',
        'rsi_7', 'rsi_14', 'rsi_30',
        'price_vs_sma_5', 'price_vs_sma_20', 'price_vs_sma_50',
        'volatility_5', 'volatility_15', 'volatility_30',
        'volume_ratio',
        'macd', 'macd_hist',
        'bb_position',
        'high_low_pos_5', 'high_low_pos_20',
        'hour', 'dayofweek',
        'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5'
    ]
    
    df = df.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
    
    X = df[feature_cols].values
    y = df['target'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    train_size = int(len(X) * 0.7)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    }
    
    print("\n=== ML Backtest (70/30 split) ===\n")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        up_pred = (y_pred == 1).sum() / len(y_pred)
        
        print(f"{name}:")
        print(f"  Train: {train_acc:.2%}, Test: {test_acc:.2%}")
        print(f"  Predicted UP: {up_pred:.2%}\n")
    
    print("=== Walk-Forward ML ===\n")
    
    tscv = TimeSeriesSplit(n_splits=10)
    
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
        print(f"{name}: CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"  Per fold: {[f'{s:.4f}' for s in cv_scores]}\n")
    
    print("\n=== Real Walk-Forward (train on past, test on future) ===\n")
    
    results = []
    days_in_data = (df['timestamp'].max() - df['timestamp'].min()).days
    
    for test_start_day in range(60, days_in_data - 30, 30):
        train_end = df['timestamp'].min() + pd.Timedelta(days=test_start_day)
        test_end = train_end + pd.Timedelta(days=14)
        
        train_mask = df['timestamp'] < train_end
        test_mask = (df['timestamp'] >= train_end) & (df['timestamp'] < test_end)
        
        if train_mask.sum() < 10000 or test_mask.sum() < 1000:
            continue
        
        X_train = scaler.fit_transform(df.loc[train_mask, feature_cols].values)
        y_train = df.loc[train_mask, 'target'].values
        X_test = scaler.transform(df.loc[test_mask, feature_cols].values)
        y_test = df.loc[test_mask, 'target'].values
        
        model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        acc = model.score(X_test, y_test)
        results.append({'period': f"{train_end.date()} to {test_end.date()}", 'accuracy': acc})
        
        print(f"Period {train_end.date()} - {test_end.date()}: {acc:.2%}")
    
    if results:
        results_df = pd.DataFrame(results)
        print(f"\n=== Summary ===")
        print(f"Mean accuracy: {results_df['accuracy'].mean():.4f}")
        print(f"Std: {results_df['accuracy'].std():.4f}")
        print(f"Min: {results_df['accuracy'].min():.4f}, Max: {results_df['accuracy'].max():.4f}")

if __name__ == "__main__":
    df = download_binance_klines()
    print(f"Loaded {len(df)} candles\n")
    
    print("Adding features...")
    df = add_features(df)
    print(f"Shape: {df.shape}\n")
    
    ml_backtest(df)