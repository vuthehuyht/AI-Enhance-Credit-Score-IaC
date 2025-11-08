#!/bin/bash
# Setup script for Flask application on EC2

set -e

echo "=== Flask Credit Score API Setup ==="

# Update system
echo "Updating system packages..."
sudo yum update -y

# Install Python 3.9
echo "Installing Python 3.9..."
sudo yum install python3.9 python3.9-pip -y

# Install git
echo "Installing git..."
sudo yum install git -y

# Create application directory
echo "Creating application directory..."
sudo mkdir -p /opt/flask-app
sudo chown ec2-user:ec2-user /opt/flask-app
cd /opt/flask-app

# Copy application files (assume files are already uploaded)
# Or clone from git repository
# git clone <your-repo-url> .

# Install Python dependencies
echo "Installing Python dependencies..."
python3.9 -m pip install --user -r requirements.txt

# Configure AWS credentials (if not using IAM role)
# aws configure

# Set environment variables
echo "Setting environment variables..."
cat > /opt/flask-app/.env << EOF
ML_DATA_BUCKET=vpbank-hackathon-dev-models
AWS_DEFAULT_REGION=ap-southeast-1
EOF

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/flask-app.service > /dev/null <<EOF
[Unit]
Description=Flask Credit Score API
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/flask-app
EnvironmentFile=/opt/flask-app/.env
ExecStart=/home/ec2-user/.local/bin/gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and start service
echo "Starting Flask application..."
sudo systemctl daemon-reload
sudo systemctl enable flask-app
sudo systemctl start flask-app

# Check status
echo ""
echo "=== Service Status ==="
sudo systemctl status flask-app --no-pager

echo ""
echo "=== Setup Complete ==="
echo "Flask app is running on http://localhost:5000"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status flask-app   # Check status"
echo "  sudo systemctl restart flask-app  # Restart service"
echo "  sudo journalctl -u flask-app -f   # View logs"
