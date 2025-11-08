# Flask Credit Score API

RESTful API for credit score prediction using PyTorch models stored in S3.

## Features

- Load and cache PyTorch models from S3
- Predict credit scores using Traditional and Social models
- Aggregate predictions from multiple models
- Health check endpoint
- Hot-reload models without restarting

## API Endpoints

### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "models_cached": ["traditional", "social"]
}
```

### 2. Predict Credit Score
```bash
POST /predict
```

Request body:
```json
{
  "input": {
    "features": [0.5, 0.3, 0.8, ...]
  }
}
```

Response:
```json
{
  "models": [
    {
      "model": "traditional",
      "result": {
        "score": 750.5,
        "score_scaled": 0.82,
        "model": "traditional"
      }
    },
    {
      "model": "social",
      "result": {
        "score": 720.3,
        "score_scaled": 0.76,
        "model": "social"
      }
    }
  ],
  "aggregated_score": 735.4,
  "num_models": 2
}
```

### 3. Reload Models
```bash
POST /models/reload
```

Clears model cache and forces reload from S3 on next request.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export ML_DATA_BUCKET=vpbank-hackathon-dev-models
export AWS_DEFAULT_REGION=ap-southeast-1
```

3. Run the app:
```bash
python app.py
```

Or with Gunicorn:
```bash
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 app:app
```

## EC2 Deployment

1. Upload files to EC2:
```bash
scp -i your-key.pem -r flask-app ec2-user@<ec2-ip>:/home/ec2-user/
```

2. SSH to EC2 and run setup:
```bash
ssh -i your-key.pem ec2-user@<ec2-ip>
cd /home/ec2-user/flask-app
chmod +x setup.sh
./setup.sh
```

3. Check service status:
```bash
sudo systemctl status flask-app
```

4. View logs:
```bash
sudo journalctl -u flask-app -f
```

## Testing

Test the API:
```bash
curl -X POST http://<ec2-ip>:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "features": [0.5, 0.3, 0.8, 0.6, 0.7, 0.4, 0.9, 0.2]
    }
  }'
```

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────┐
│   Flask     │
│     App     │
└──────┬──────┘
       │
       ├──► Load models from S3
       │
       └──► PyTorch inference
            (CPU)
```

## Environment Variables

- `ML_DATA_BUCKET`: S3 bucket containing model files (default: `vpbank-hackathon-dev-models`)
- `AWS_DEFAULT_REGION`: AWS region (default: `ap-southeast-1`)

## Notes

- Models are cached in memory after first load
- Uses CPU-only PyTorch for inference
- Gunicorn with 2 workers recommended for production
- EC2 instance should have IAM role with S3 read access
