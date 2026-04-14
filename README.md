# BTC Polymarket Predictor

Predicts Bitcoin UP/DOWN for 5-minute windows on Polymarket.

## How It Works

- **Predict at**: Every :10 bar (e.g., 11:10, 12:10, etc.)
- **Predict**: Will next 5-min bar close HIGHER than current :10 bar close?
- **Accuracy**: ~63% with 60%+ confidence threshold
- **Edge**: +12.6% over baseline

## Files

- `predict_v3.py` - Main production predictor
- `data/btc_5m_2023_2025.csv` - Historical data

## Usage

```bash
python predict_v3.py
```

Output:
```
=== SIGNAL ===
At :10 bar, predict: DOWN
Probability UP: 42.3%

>>> NO BET (confidence below 60%)
```

## Strategy

1. After :10 bar closes, run prediction
2. If model probability >60% → bet UP
3. If model probability <40% → bet DOWN  
4. Otherwise → no bet (wait for better confidence)

## Timing Table

| Polymarket Window | Run Prediction After |
|------------------|---------------------|
| 11:00-11:05 | 11:10 bar closed |
| 12:00-12:05 | 12:10 bar closed |
| 13:00-13:05 | 13:10 bar closed |
| ... | ... |

## Historical Results

- Best minute: :10 → 63.2% accuracy
- Threshold 60%: ~63% with +12.6% edge
- All trades: ~57%