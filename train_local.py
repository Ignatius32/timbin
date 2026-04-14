#!/usr/bin/env python3
"""
Fine-tune TimesFM on BTC 1m historical data
Uses existing local CSV data
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TimesFm2_5ModelForPrediction
import warnings
import gc
import os
warnings.filterwarnings('ignore')

torch.cuda.empty_cache()
gc.collect()

MODEL_NAME = "google/timesfm-2.5-200m-transformers"
EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
CONTEXT_LEN = 128
MAX_BARS = 5000

def load_btc_data():
    """Load BTC data from existing CSV files"""
    csv_files = [
        'data/btc_1m_2024.csv',
        'data/btc_1m_2023_2025.csv',
    ]
    
    all_prices = []
    for f in csv_files:
        if os.path.exists(f):
            print(f"Loading {f}...")
            df = pd.read_csv(f, parse_dates=['timestamp'])
            all_prices.append(df['close'].values)
            print(f"  Loaded {len(df)} bars")
    
    if not all_prices:
        raise Exception("No data files found!")
    
    prices = np.concatenate(all_prices)
    prices = np.unique(prices)
    print(f"Total unique bars: {len(prices)}")
    return prices

class BTCDataset(Dataset):
    def __init__(self, prices, ctx):
        self.data = [(prices[i:i+ctx], prices[i+ctx:i+ctx*2]) 
                  for i in range(len(prices)-ctx*2+1)]
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        c, t = self.data[i]
        return torch.tensor(c, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)

print("=== TimesFM Fine-tune with Historical Data ===")
print(f"Bars: {MAX_BARS}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")

# Load historical data
print("\n=== Loading Historical Data ===")
prices = load_btc_data()
print(f"Total bars: {len(prices)}")

# Use last N bars
prices = prices[-MAX_BARS:]
print(f"Using last {len(prices)} bars for training")

# Create dataset
dataset = BTCDataset(prices, CONTEXT_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load model
print("\n=== Loading TimesFM ===")
model = TimesFm2_5ModelForPrediction.from_pretrained(MODEL_NAME, device_map="cuda")
model.train()
print(f"Model on: cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"\n=== Training: {len(loader)} steps ===")
import time
start = time.time()

step_count = 0
for epoch in range(EPOCHS):
    total_loss = 0
    
    for ctx, tgt in loader:
        ctx = ctx.cuda()
        tgt = tgt.cuda()
        
        optimizer.zero_grad()
        outputs = model(past_values=ctx, return_dict=True)
        pred = outputs.mean_predictions
        loss = torch.nn.MSELoss()(pred, tgt)
        
        if torch.isnan(loss):
            continue
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        step_count += 1
        
        if step_count % 25 == 0:
            print(f"Step {step_count}, Loss: {loss.item():.2f}")
        
        del loss, pred, outputs
        torch.cuda.empty_cache()
    
    print(f"Epoch {epoch+1}: avg loss = {total_loss/max(step_count,1):.2f}")

print(f"\nTime: {time.time()-start:.1f}s")

# Save
output_dir = "cache/timesfm_btc_ft"
print(f"Saving to {output_dir}/...")
model.save_pretrained(output_dir)
print("Done!")