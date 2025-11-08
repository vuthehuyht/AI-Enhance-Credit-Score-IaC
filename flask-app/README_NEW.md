# Credit Score Prediction Flask API

Flask API for credit score prediction with intelligent routing:
- **Existing customers**: Uses traditional + social models with score aggregation
- **New customers**: Uses XGBoost model for initial assessment

## Architecture

```
Client Request
     ↓
  customer_id
     ↓
 Check DynamoDB
     ↓
 ┌───────────┴──────────┐
 ↓                      ↓
EXISTING           NEW CUSTOMER
CUSTOMER              ↓
 ↓                 XGBoost Model
Traditional +         ↓
Social Models      Single Score
 ↓
Aggregate Score
```

## API Endpoints

### 1. POST /api/v1/predict
Predict credit score with automatic routing based on customer type.

**Request (Existing Customer):**
```json
{
  "customer_id": "CUS123456",
  "features": {
    "traditional": [0.5, 0.8, 0.3, ...],
    "social": [0.6, 0.7, 0.5, ...]
  }
}
```

**Response (Existing Customer):**
```json
{
  "customer_id": "CUS123456",
  "customer_type": "existing",
  "aggregated_score": 735.4,
  "models": [
    {"model": "traditional", "score": 750.5},
    {"model": "social", "score": 720.3}
  ],
  "num_models_used": 2,
  "prediction_type": "multi_model_aggregate",
  "timestamp": "2025-11-08T10:30:00Z"
}
```

**Request (New Customer):**
```json
{
  "customer_id": "CUS999999",
  "features": {
    "xgboost": [0.5, 0.8, 0.3, ...]
  }
}
```

**Response (New Customer):**
```json
{
  "customer_id": "CUS999999",
  "customer_type": "new",
  "score": 680.2,
  "model": "xgboost",
  "prediction_type": "new_customer_xgboost",
  "timestamp": "2025-11-08T10:30:00Z"
}
```

### 2. GET/POST /api/v1/search
Search customer credit score history.

**Request (GET):**
```
GET /api/v1/search?customer_id=CUS123456
```

**Request (POST):**
```json
{
  "customer_id": "CUS123456"
}
```

**Response:**
```json
{
  "customer_id": "CUS123456",
  "found": true,
  "latest_prediction": {
    "timestamp": "2025-11-08T10:30:00Z",
    "credit_score": 735.4,
    "prediction_type": "multi_model_aggregate",
    "models_used": ["traditional", "social"],
    "raw_predictions": [...]
  }
}
```

### 3. GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_cached": ["traditional", "social", "xgboost"]
}
```

## Environment Variables

```bash
ML_DATA_BUCKET=vpbank-hackathon-dev-models        # S3 bucket for models
CUSTOMER_TABLE_NAME=vpbank-customers              # DynamoDB table name
AWS_DEFAULT_REGION=ap-southeast-1                 # AWS region
```

## Model Storage Structure

```
s3://bucket/
├── traditional/
│   ├── model.pt
│   └── scaler.joblib
├── social/
│   ├── model.pkl
│   └── scaler.joblib
└── xgboost/
    ├── model.json
    ├── scaler.joblib
    └── config.json
```

## Setup on EC2

1. **Install dependencies:**
```bash
chmod +x setup.sh
./setup.sh
```

2. **Set environment variables:**
```bash
export ML_DATA_BUCKET=your-bucket-name
export CUSTOMER_TABLE_NAME=your-table-name
export AWS_DEFAULT_REGION=ap-southeast-1
```

3. **Run the application:**
```bash
# Development
python app.py

# Production with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Testing

Use the provided `test_api.http` file with REST Client extension:

```http
### Test existing customer
POST http://localhost:5000/api/v1/predict
Content-Type: application/json

{
  "customer_id": "CUS123456",
  "features": {
    "traditional": [...],
    "social": [...]
  }
}
```

## DynamoDB Schema

**Table Name:** `vpbank-customers`

**Primary Key:** `customer_id` (String)

**Attributes:**
- `customer_id`: String (Partition Key)
- `prediction_timestamp`: String (ISO 8601)
- `credit_score`: Number
- `models_used`: List
- `prediction_type`: String
- `raw_predictions`: List

## Features

✅ **Intelligent Routing**: Automatically selects models based on customer type  
✅ **Model Caching**: Models loaded once and cached in memory  
✅ **Database Integration**: Stores and retrieves predictions from DynamoDB  
✅ **Multi-Model Support**: PyTorch, Sklearn, XGBoost  
✅ **Error Handling**: Graceful degradation if models fail  
✅ **Health Checks**: Monitor application status  

## Production Deployment

```bash
# Start with systemd
sudo systemctl start flask-credit-api

# Check status
sudo systemctl status flask-credit-api

# View logs
sudo journalctl -u flask-credit-api -f
```

## Request Flow Examples

### Example 1: Existing Customer
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUS123456",
    "features": {
      "traditional": [0.5, 0.8, 0.3, 0.7, 0.9],
      "social": [0.6, 0.7, 0.5]
    }
  }'
```

### Example 2: New Customer
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUS999999",
    "features": {
      "xgboost": [0.5, 0.8, 0.3, 0.7, 0.9, 0.2]
    }
  }'
```

### Example 3: Search Customer
```bash
curl "http://localhost:5000/api/v1/search?customer_id=CUS123456"
```
