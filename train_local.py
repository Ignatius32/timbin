#!/usr/bin/env python3
"""
Fine-tune TimesFM on BTC 1m data - Local GPU version
RTX 2070: 8GB VRAM - optimized
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TimesFm2_5ModelForPrediction
import warnings
import gc
warnings.filterwarnings('ignore')

torch.cuda.empty_cache()
gc.collect()

MODEL_NAME = "google/timesfm-2.5-200m-transformers"
EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
CONTEXT_LEN = 128
MAX_BARS = 20000  # Smaller for memory

def load_btc_data():
    """Load BTC data from local files"""
    import os
    csv_files = [
        'data/btc_1m_2024.csv',
        'data/btc_1m_2023_2025.csv',
    ]
    
    for f in csv_files:
        if os.path.exists(f):
            print(f"Loading {f}...")
            df = pd.read_csv(f, parse_dates=['timestamp'])
            return df['close'].values
    
    print("No local data, using API...")
    import requests
    url = "https://api.binance.com/api/v3/klines"
    df = pd.DataFrame(
        requests.get(url, params={"symbol":"BTCUSDT","interval":"1m","limit":1000}).json(),
        columns=['t','o','h','l','c','v','ct','qv','tr','tbb','tbq','i']
    )
    return pd.to_numeric(df['c'], errors='coerce').values

class BTCDataset(Dataset):
    def __init__(self, prices, ctx):
        self.data = [(prices[i:i+ctx], prices[i+ctx:i+ctx*2]) 
                  for i in range(len(prices)-ctx*2+1)]
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        c, t = self.data[i]
        return torch.tensor(c, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)

print("=== TimesFM Local GPU Training ===")
print(f"Bars: {MAX_BARS}, Batch: {BATCH_SIZE}")

# Load data
print("Loading BTC data...")
prices = load_btc_data()
prices = prices[-MAX_BARS:]
print(f"Using last {len(prices)} bars")

# Try to free memory
del prices
torch.cuda.empty_cache()
gc.collect()

# Create dataset
dataset = BTCDataset(prices, CONTEXT_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Loading TimesFM model (GPU)...")
try:
    model = TimesFm2_5ModelForPrediction.from_pretrained(MODEL_NAME, device_map="cuda")
except Exception as e:
    print(f"GPU failed: {e}, trying CPU...")
    model = TimesFm2_5ModelForPrediction.from_pretrained(MODEL_NAME, device_map="cpu")

model.train()
print(f"Model on: {model.device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"Training: {len(loader)} steps")
start_time = __import__('time').time()

step_count = 0
for epoch in range(EPOCHS):
    total_loss = 0
    
    for ctx, tgt in loader:
        if torch.cuda.is_available():
            ctx = ctx.cuda()
            tgt = tgt.cuda()
        
        optimizer.zero_grad()
        
        outputs = model(past_values=ctx, return_dict=True)
        pred = outputs.mean_predictions
        
        loss = torch.nn.MSELoss()(pred, tgt)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"NaN loss at step {step_count}, skipping...")
            continue
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        step_count += 1
        
        if step_count % 50 == 0:
            print(f"Step {step_count}, Loss: {loss.item():.2f}")
        
        # Memory cleanup
        del loss, pred, outputs
        torch.cuda.empty_cache()
    
    print(f"Epoch {epoch+1}: avg loss = {total_loss/max(step_count,1):.2f}")

print(f"\nTime: {__import__('time').time() - start_time:.1f}s")

# Save
output_dir = "cache/timesfm_btc_ft"
print(f"Saving to {output_dir}/...")
model.save_pretrained(output_dir)
print("Done!")