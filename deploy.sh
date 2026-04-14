#!/bin/bash
# Deploy TimesFM to AWS - Simple version
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

echo "=== Deploy TimesFM Fine-tune ==="
echo "Key: $KEY_NAME"
echo "Bucket: $S3_BUCKET"

# Check AWS
aws sts get-caller-identity >/dev/null 2>&1 || { echo "Run 'aws configure' first"; exit 1; }

# Create SG
aws ec2 create-security-group --group-name timbin-sg --description "TimesFM" --region $REGION 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-name timbin-sg --protocol tcp --port 22 --cidr 0.0.0.0/0 --region $REGION 2>/dev/null || true

# Launch
echo "Launching instance..."
ID=$(aws ec2 run-instances \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-groups timbin-sg \
  --region $REGION \
  --image-id $AMI \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance: $ID"
aws ec2 wait instance-running --instance-ids $ID --region $REGION

DNS=$(aws ec2 describe-instances --instance-ids $ID --region $REGION \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)

echo ""
echo "=== READY ==="
echo "SSH: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$DNS"
echo ""
echo "Then run:"
echo "  cd /home/ubuntu"
echo "  git clone https://github.com/Ignatius32/timbin.git"
echo "  cd timbin"
echo "  python3 -m venv venv && source venv/bin/activate"
echo "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
echo "  pip install transformers accelerate pandas requests"
echo "  export EPOCHS=1 MAX_BARS=50000 BATCH_SIZE=2"
echo "  python train_timesfm.py"
echo ""
echo "After training (~$30 min):"
echo "  aws s3 cp timesfm_btc_ft/ s3://${S3_BUCKET}/ --recursive"
echo ""
echo "To terminate: aws ec2 terminate-instances --instance-ids $ID --region $REGION"