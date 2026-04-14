#!/bin/bash
# TimesFM Fine-tune - Single Command Deployment
# Run this ONE command and everything happens automatically
#
# Prereqs:
# 1. AWS CLI configured: aws configure
# 2. Key pair in AWS Console
# 3. S3 bucket for model output
#
# Usage: ./deploy.sh <key-name> <s3-bucket>
#
# Example: ./deploy.sh my-key my-bucket

KEY_NAME=${1:-your-key-name}
S3_BUCKET=${2:-your-bucket}
REGION="us-east-1"
INSTANCE_TYPE="g4dn.xlarge"

if [ "$KEY_NAME" = "your-key-name" ]; then
    echo "Usage: ./deploy.sh <key-name> <s3-bucket>"
    echo "Example: ./deploy.sh timbin-key timbin-models"
    exit 1
fi

echo "=== TimesFM One-Command Deployment ==="
echo "Key: $KEY_NAME | Bucket: $S3_BUCKET"

# Check AWS
aws sts get-caller-identity >/dev/null 2>&1 || { echo "Run 'aws configure' first"; exit 1; }

# Security group
aws ec2 create-security-group --group-name timbin-sg --description "TimesFM" --region $REGION 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-name timbin-sg --protocol tcp --port 22 --cidr 0.0.0.0/0 --region $REGION 2>/dev/null || true

# Full automatic user script
USER_DATA=$(cat << 'SCRIPT' | base64 -w0
#!/bin/bash
exec > /home/ubuntu/train.log 2>&1

cd /home/ubuntu

# Install miniconda
wget -q https://repo.anaconda.com/miniconda3/Miniconda3-latest-Linux-x86_64.sh -O mc.sh
bash mc.sh -b -p /opt/conda; rm mc.sh
export PATH=/opt/conda/bin:$PATH

# Install Python & deps
conda install -y python=3.11 pip
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate pandas numpy requests

# Clone repo
cd /home/ubuntu
git clone https://github.com/Ignatius32/timbin.git timbin 2>/dev/null || cd timbin && git pull
cd timbin

# Download BTC data from Binance Vision (if not in repo)
echo "=== Downloading BTC historical data ==="
wget -q https://data.binance.vision/data/spot/monthly/klines/btcusdt/1m/btcusdt-1m-2023.zip
wget -q https://data.binance.vision/data/spot/monthly/klines/btcusdt/1m/btcusdt-1m-2024.zip
wget -q https://data.binance.vision/data/spot/monthly/klines/btcusdt/1m/btcusdt-1m-2025.zip
mkdir -p data
unzip -o btcusdt-1m-2023.zip -d data/ 2>/dev/null || true
unzip -o btcusdt-1m-2024.zip -d data/ 2>/dev/null || true
unzip -o btcusdt-1m-2025.zip -d data/ 2>/dev/null || true
rm -f btcusdt-1m-*.zip

echo "=== Data ready ==="
ls -la data/*.csv | head -5

# Fine-tune TimesFM
export EPOCHS=1 MAX_BARS=100000 BATCH_SIZE=2

echo "=== Starting TimesFM Fine-tune ==="
python train_timesfm.py

# Upload to S3
aws s3 cp timesfm_btc_ft/ s3://S3_BUCKET/timesfm_btc_ft/ --recursive

echo "DONE" > /home/ubuntu/status
SCRIPT
)

# Launch instance
ID=$(aws ec2 run-instances \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-groups timbin-sg \
  --region $REGION \
  --image-id ami-0c55b159c9a5e4412 \
  --user-data "$USER_DATA" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance: $ID"
aws ec2 wait instance-running --instance-ids $ID --region $REGION

DNS=$(aws ec2 describe-instances --instance-ids $ID --region $REGION \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)

echo ""
echo "=== TRAINING IN PROGRESS ==="
echo "Instance: $ID"
echo "SSH: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$DNS"
echo ""
echo "Monitor:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$DNS 'tail -f train.log'"
echo ""
echo "Download model when done (~30-60 min):"
echo "  aws s3 sync s3://${S3_BUCKET}/timesfm_btc_ft/ ./cache/"
echo ""
echo "Terminate:"
echo "  aws ec2 terminate-instances --instance-ids $ID --region $REGION"