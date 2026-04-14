#!/bin/bash
# Download BTCUSDT 1m data from Binance Vision
# Run this locally, then commit the data files to repo

mkdir -p data

echo "Downloading BTCUSDT 1m historical data..."
echo ""

# 2023
echo "Downloading 2023..."
wget -q https://data.binance.vision/data/spot/monthly/klines/btcusdt/1m/btcusdt-1m-2023.zip
unzip -o btcusdt-1m-2023.zip -d data/
rm btcusdt-1m-2023.zip

# 2024
echo "Downloading 2024..."
wget -q https://data.binance.vision/data/spot/monthly/klines/btcusdt/1m/btcusdt-1m-2024.zip
unzip -o btcusdt-1m-2024.zip -d data/
rm btcusdt-1m-2024.zip

# 2025
echo "Downloading 2025..."
wget -q https://data.binance.vision/data/spot/monthly/klines/btcusdt/1m/btcusdt-1m-2025.zip
unzip -o btcusdt-1m-2025.zip -d data/
rm btcusdt-1m-2025.zip

# 2026 (if available)
echo "Downloading 2026..."
wget -q https://data.binance.vision/data/spot/monthly/klines/btcusdt/1m/btcusdt-1m-2026.zip
if [ -f btcusdt-1m-2026.zip ]; then
    unzip -o btcusdt-1m-2026.zip -d data/
    rm btcusdt-1m-2026.zip
else
    echo "2026 not available yet"
fi

echo ""
echo "=== Download complete ==="
ls -la data/*.csv 2>/dev/null | head -10