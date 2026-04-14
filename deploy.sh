#!/bin/bash
# Deploy TimesFM to AWS - Full version
# 
# Prereqs:
# 1. AWS CLI configured: aws configure
# 2. Key pair created in AWS console
# 3. S3 bucket created for model output
#
# Usage: ./deploy.sh <key-name> <s3-bucket>

KEY_NAME=${1:-your-key-name}
S3_BUCKET=${2:-your-bucket}
REGION="us-east-1"
INSTANCE_TYPE="g4dn.xlarge"
AMI="ami-0c55b159c9a5e4412"  # Ubuntu DL

if [ "$KEY_NAME" = "your-key-name" ]; then
    echo "Usage: ./deploy.sh <key-name> <s3-bucket>"
    echo "Example: ./deploy.sh my-key my-bucket-name"
    exit 1
fi

echo "=== TimesFM Fine-tune Deployment ==="
echo "Key: $KEY_NAME"
echo "Bucket: $S3_BUCKET"

# Check AWS
aws sts get-caller-identity >/dev/null 2>&1 || { echo "Run 'aws configure' first"; exit 1; }

# Create SG
aws ec2 create-security-group --group-name timbin-sg --description "TimesFM" --region $REGION 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-name timbin-sg --protocol tcp --port 22 --cidr 0.0.0.0/0 --region $REGION 2>/dev/null || true

# User data script (base64 encoded)
USER_DATA=$(cat << 'SCRIPT' | base64 -w0
#!/bin/bash
exec > /home/ubuntu/train.log 2>&1

cd /home/ubuntu

# Install conda
wget -q https://repo.anaconda.com/miniconda3/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /opt/conda
rm miniconda.sh
export PATH=/opt/conda/bin:$PATH

# Install Python & deps
conda install -y python=3.11 pip
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate pandas numpy requests

# Clone repo (with GitHub token if needed)
cd /home/ubuntu
git clone https://github.com/Ignatius32/timbin.git timbin 2>/dev/null || cd timbin
cd timbin

# Download BTC data (if not in repo)
if [ ! -f data/btcusdt-1m-2024.csv ]; then
    echo "Downloading BTC data from Binance Vision..."
    ./download_data.sh
fi

# Configure and train
export EPOCHS=1 MAX_BARS=100000 BATCH_SIZE=2 OUTPUT_DIR=timesfm_btc_ft

echo "=== Starting TimesFM Fine-tune ==="
python train_timesfm.py

# Upload model to S3
echo "Uploading model to S3..."
aws s3 cp timesfm_btc_ft/ s3://S3_BUCKET/timesfm_btc_ft/ --recursive || echo "S3 upload failed"

# Write completion marker
echo "TRAINING_COMPLETE" > /home/ubuntu/status
echo "Done!" >> /home/ubuntu/status
SCRIPT
)

echo "Launching instance..."
ID=$(aws ec2 run-instances \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-groups timbin-sg \
  --region $REGION \
  --image-id $AMI \
  --user-data "$USER_DATA" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance: $ID"
aws ec2 wait instance-running --instance-ids $ID --region $REGION

DNS=$(aws ec2 describe-instances --instance-ids $ID --region $REGION \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)

echo ""
echo "=== TRAINING STARTED ==="
echo "DNS: $DNS"
echo ""
echo "Monitor with:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$DNS 'tail -f train.log'"
echo ""
echo "Or check instance status:"
echo "  aws ec2 describe-instances --instance-ids $ID --region $REGION --output text"
echo ""
echo "Download model when done (~30-60 min):"
echo "  aws s3 sync s3://${S3_BUCKET}/timesfm_btc_ft/ ./cache/"
echo ""
echo "To terminate:"
echo "  aws ec2 terminate-instances --instance-ids $ID --region $REGION"