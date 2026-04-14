#!/usr/bin/env python3
"""
Fine-tune TimesFM on BTC 1m data
Run on AWS GPU, save model locally
"""
import numpy as np
import pandas as pd
import requests
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

# Hyperparams - tune for your GPU
EPOCHS = int(os.environ.get('EPOCHS', '1'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '2'))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '1e-5'))
CONTEXT_LEN = 128
MAX_BARS = int(os.environ.get('MAX_BARS', '50000'))

def get_btc_1m(limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    df = pd.DataFrame(
        requests.get(url, params={"symbol": "BTCUSDT","interval":"1m","limit":limit}).json(),
        columns=['t','o','h','l','c','v','ct','qv','tr','tbb','tbq','i']
    )
    for c in ['o','h','l','c','v']: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df['c'].values

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

# Get data - use Binance API (limited) or load from file
prices = get_btc_1m(1000)
if len(prices) < CONTEXT_LEN*2:
    print(f"Using synthetic data augmentation...")
    prices = np.concatenate([prices] * 20)

prices = prices[-MAX_BARS:]
print(f"Training samples: {len(prices) - CONTEXT_LEN*2 + 1}")

ds = BTCDataset(prices, CONTEXT_LEN)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

print("Loading TimesFM...")
model = TimesFm2_5ModelForPrediction.from_pretrained(MODEL_NAME)
model.train()

opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"Training: {len(loader)} steps/epoch")
start = time.time()

for ep in range(EPOCHS):
    total = 0
    for step, (ctx, tgt) in enumerate(loader):
        opt.zero_grad()
        out = model(past_values=ctx, return_dict=True)
        loss = torch.nn.MSELoss()(out.mean_predictions, tgt)
        loss.backward()
        opt.step()
        total += loss.item()
        if step % 100 == 0:
            print(f"Ep{ep+1} Step{step} Loss:{loss.item():.2f}")
    
    print(f"Epoch {ep+1}: avg loss = {total/max(step,1):.2f}")

print(f"\nTime: {time.time()-start:.1f}s")
print(f"Saving to {OUTPUT_DIR}/...")

model.save_pretrained(OUTPUT_DIR)
print("Done!")
print(f"\nTo download locally:")
print(f"  aws s3 sync s3://your-bucket/{OUTPUT_DIR} ./cache/")