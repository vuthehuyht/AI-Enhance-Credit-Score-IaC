#!/bin/bash
# Deploy Flask app to EC2 instance

set -e

# Configuration
EC2_USER="ec2-user"
EC2_IP="$1"
KEY_FILE="$2"

if [ -z "$EC2_IP" ] || [ -z "$KEY_FILE" ]; then
    echo "Usage: ./deploy_flask.sh <ec2-ip> <path-to-key.pem>"
    echo "Example: ./deploy_flask.sh 54.123.45.67 ~/.ssh/my-key.pem"
    exit 1
fi

echo "=== Deploying Flask App to EC2 ==="
echo "EC2 IP: $EC2_IP"
echo "Key file: $KEY_FILE"
echo ""

# Create deployment package
echo "Creating deployment package..."
cd flask-app
tar -czf ../flask-app.tar.gz .
cd ..

# Upload to EC2
echo "Uploading files to EC2..."
scp -i "$KEY_FILE" flask-app.tar.gz ${EC2_USER}@${EC2_IP}:/home/${EC2_USER}/

# Extract and setup on EC2
echo "Setting up Flask app on EC2..."
ssh -i "$KEY_FILE" ${EC2_USER}@${EC2_IP} << 'ENDSSH'
    # Extract files
    sudo mkdir -p /opt/flask-app
    sudo tar -xzf /home/ec2-user/flask-app.tar.gz -C /opt/flask-app
    sudo chown -R ec2-user:ec2-user /opt/flask-app
    cd /opt/flask-app

    # Make setup script executable
    chmod +x setup.sh

    # Run setup script
    ./setup.sh

    echo ""
    echo "=== Deployment Complete ==="
ENDSSH

# Cleanup
rm flask-app.tar.gz

echo ""
echo "Flask app deployed successfully!"
echo "Access the API at: http://$EC2_IP:5000"
echo ""
echo "Test the API:"
echo "  curl http://$EC2_IP:5000/health"
