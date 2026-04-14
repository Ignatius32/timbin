import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"

def download_data():
    csv_path = f"{DATA_DIR}/btc_1m.csv"
    if os.path.exists(csv_path):
        print("Loading data...")
        return pd.read_csv(csv_path, parse_dates=['timestamp'])
    
    print("Downloading...")
    url = "https://api.binance.com/api/v3/klines"
    data = []
    start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
    end_ts = int(datetime(2024, 12, 31).timestamp() * 1000)
    
    while start_ts < end_ts:
        d = requests.get(url, params={"symbol": "BTCUSDT", "interval": "1m", "startTime": start_ts, "endTime": end_ts, "limit": 1000}).json()
        if not d: break
        data.extend(d)
        start_ts = d[-1][0] + 1
        if len(data) % 100000 == 0: print(f"  {len(data)}...")
    
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume','ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
    df.to_csv(csv_path, index=False)
    return df

def features(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df['ret1'] = df['close'].pct_change(1)
    df['ret3'] = df['close'].pct_change(3)
    df['ret5'] = df['close'].pct_change(5)
    df['body'] = (df['close'] - df['open']) / df['open']
    df['range'] = (df['high'] - df['low']) / df['low']
    
    # RSI
    for w in [7, 14]:
        d = df['close'].diff()
        g = d.where(d>0,0).rolling(w).mean()
        l = (-d.where(d<0,0)).rolling(w).mean()
        df[f'rsi_{w}'] = 100 - (100/(1 + g/(l+1e-10)))
    
    # Price vs SMA
    for w in [10, 30]:
        sma = df['close'].rolling(w).mean()
        df[f'vsma_{w}'] = (df['close'] - sma) / sma
    
    # Volatility
    df['vol5'] = df['ret1'].rolling(5).std()
    df['vol15'] = df['ret1'].rolling(15).std()
    
    # Volume
    df['vol_ma'] = df['volume'].rolling(30).mean()
    df['vol_rat'] = df['volume'] / (df['vol_ma']+1e-10)
    
    # MACD
    e12, e26 = df['close'].ewm(span=12, adjust=False).mean(), df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = e12 - e26
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_h'] = df['macd'] - df['macd_sig']
    
    # Hour
    df['hour'] = df['timestamp'].dt.hour
    
    return df

def main():
    df = download_data()
    print(f"Loaded {len(df)} rows\n")
    
    print("Computing features...")
    df = features(df)
    
    feats = ['ret1','ret3','ret5','body','range','rsi_7','rsi_14','vsma_10','vsma_30','vol5','vol15','vol_rat','macd','macd_h','hour']
    df = df.dropna(subset=feats+['target']).reset_index(drop=True)
    print(f"After clean: {len(df)} rows\n")
    
    X = df[feats].values
    y = df['target'].values
    times = df['timestamp'].values
    
    # Walk-forward: use last 30 days as test, rest as train
    split_idx = int(len(df) * 0.9)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[split_idx]}")
    print(f"Test period: {df['timestamp'].iloc[split_idx]} to {df['timestamp'].iloc[-1]}\n")
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print("Training Gradient Boosting...")
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42, subsample=0.5)
    model.fit(X_train_s, y_train)
    
    train_acc = (model.predict(X_train_s) == y_train).mean()
    test_acc = (model.predict(X_test_s) == y_test).mean()
    
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    
    # Feature importance
    imp = pd.DataFrame({'feat': feats, 'imp': model.feature_importances_}).sort_values('imp', ascending=False)
    print("\nTop features:")
    print(imp.head(10).to_string(index=False))

if __name__ == "__main__":
    main()