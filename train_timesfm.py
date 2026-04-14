#!/usr/bin/env python3
"""
Fine-tune TimesFM on BTC 1m data
Run on AWS GPU, uses LOCAL CSV data from repo
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TimesFm2_5ModelForPrediction
import warnings
warnings.filterwarnings('ignore')
import os
import time

torch.cuda.empty_cache()

MODEL_NAME = "google/timesfm-2.5-200m-transformers"
OUTPUT_DIR = "timesfm_btc_ft"

# Hyperparams (AWS g4dn.xlarge = 16GB VRAM)
EPOCHS = int(os.environ.get('EPOCHS', '1'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '1'))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '1e-5'))
CONTEXT_LEN = 128
MAX_BARS = int(os.environ.get('MAX_BARS', '50000'))

def load_btc_csv():
    """Load all BTC 1m data from local CSV files in repo"""
    # Files from Binance Vision (download_data.sh)
    csv_files = [
        'data/btcusdt-1m-2023.csv',
        'data/btcusdt-1m-2024.csv',
        'data/btcusdt-1m-2025.csv',
        'data/btcusdt-1m-2026.csv',
    ]
    
    all_prices = []
    for f in csv_files:
        if os.path.exists(f):
            print(f"Loading {f}...")
            try:
                df = pd.read_csv(f, parse_dates=['timestamp'])
                all_prices.append(df['close'].values)
            except:
                # Try loading without timestamp parsing
                df = pd.read_csv(f)
                all_prices.append(df['close'].values)
    
    if not all_prices:
        # Fallback to API if no CSV
        print("No CSV found, using Binance API...")
        import requests
        url = "https://api.binance.com/api/v3/klines"
        df = pd.DataFrame(
            requests.get(url, params={"symbol":"BTCUSDT","interval":"1m","limit":1000}).json(),
            columns=['t','o','h','l','c','v','ct','qv','tr','tbb','tbq','i']
        )
        return pd.to_numeric(df['c'], errors='coerce').values
    
    prices = np.concatenate(all_prices)
    # Remove duplicates if any
    prices = np.unique(prices)
    print(f"Total bars: {len(prices)}")
    return prices

class BTCDataset(Dataset):
    def __init__(self, prices, ctx):
        self.data = [(prices[i:i+ctx], prices[i+ctx:i+ctx*2]) 
                  for i in range(len(prices)-ctx*2+1)]
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        c, t = self.data[i]
        return torch.tensor(c, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)

print(f"=== TimesFM Fine-tune ===")
print(f"Model: {MODEL_NAME}")
print(f"Bars: {MAX_BARS}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")
print(f"Using data from Binance Vision (2023-2026)")

# Get data - load from CSV files (from Binance Vision)
print("Loading BTC data from CSV files...")
prices = load_btc_csv()

# Use only last N bars (memory constraint)
prices = prices[-MAX_BARS:]
print(f"Using last {MAX_BARS} bars for training")
print(f"Training samples: {len(prices) - CONTEXT_LEN*2 + 1}")

ds = BTCDataset(prices, CONTEXT_LEN)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

print("Loading TimesFM (AWS GPU)...")
model = TimesFm2_5ModelForPrediction.from_pretrained(MODEL_NAME, device_map="cuda")
model.train()
print("Model on: cuda")

opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"Training: {len(loader)} steps/epoch")
start = time.time()

for ep in range(EPOCHS):
    total = 0
    for step, (ctx, tgt) in enumerate(loader):
        ctx = ctx.cuda()
        tgt = tgt.cuda()
        opt.zero_grad()
        out = model(past_values=ctx, return_dict=True)
        loss = torch.nn.MSELoss()(out.mean_predictions, tgt)
        loss.backward()
        opt.step()
        total += loss.item()
        if step % 100 == 0:
            print(f"Ep{ep+1} Step{step} Loss:{loss.item():.2f}")
        torch.cuda.empty_cache()
    
    print(f"Epoch {ep+1}: avg loss = {total/max(step,1):.2f}")

print(f"\nTime: {time.time()-start:.1f}s")
print(f"Saving to {OUTPUT_DIR}/...")

model.save_pretrained(OUTPUT_DIR)
print("Done!")
print(f"\nTo download locally:")
print(f"  aws s3 sync s3://your-bucket/{OUTPUT_DIR} ./cache/")