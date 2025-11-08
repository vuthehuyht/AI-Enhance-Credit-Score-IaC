# Infrastructure Changes - Cost Optimization

## Summary

Đã loại bỏ API Gateway và Lambda aggregate inference, thay thế bằng Flask API trên EC2 để tối ưu chi phí.

## Changes Made

### ❌ Removed (Deprecated)
1. **API Gateway** (`apigw.tf`) - REST API và tất cả resources
2. **Lambda Aggregate Inference** - Function gọi SageMaker endpoints
3. **AWS Amplify** (`amplify.tf`) - Frontend hosting
4. **SageMaker Endpoints** - Real-time inference endpoints (không cần thiết)

### ✅ Added (New Infrastructure)  
1. **Flask API on EC2** (`compute.tf`, `flask-app/`)
   - Smart routing dựa trên customer type
   - Load models từ S3, cache trong memory
   - DynamoDB integration cho customer lookup
   
2. **DynamoDB Table** (`dynamodb.tf`)
   - Lưu trữ customer data
   - Track prediction history
   
3. **Enhanced Security**
   - EC2 IAM role với S3 + DynamoDB access
   - VPC, Security Groups đầy đủ

## Cost Comparison

### Old Architecture (API Gateway + Lambda + Endpoints)
```
Assuming 100K requests/month:

- API Gateway: 100K * $3.50/million = $0.35/month
- Lambda: 100K * $0.20/million = $0.02/month  
- SageMaker Endpoints (2x):
  * traditional: $0.134/hour * 730 = $97.82/month
  * social: $0.134/hour * 730 = $97.82/month
- Data Transfer: ~$5/month

TOTAL: ~$201/month
```

### New Architecture (Flask on EC2)
```
- EC2 t3.medium: $0.0416/hour * 730 = $30.37/month
- S3 Storage (models): ~$1/month
- DynamoDB (on-demand): ~$2/month
- Data Transfer: ~$3/month

TOTAL: ~$36/month

SAVINGS: $165/month (82% reduction)
```

## Architecture Flow

### Old Flow
```
Client → API Gateway → Lambda → SageMaker Endpoints → Response
          ($)           ($)            ($$$$)
```

### New Flow  
```
Client → EC2 Flask API → S3 (models) + DynamoDB → Response
              ($)              (free-tier)
```

## API Endpoints

### Flask API (Port 5000)

**1. Predict Credit Score**
```bash
POST http://<EC2_IP>:5000/api/v1/predict
Content-Type: application/json

# Existing Customer (uses traditional + social)
{
  "customer_id": "CUS123456",
  "features": {
    "traditional": [...],
    "social": [...]
  }
}

# New Customer (uses xgboost)
{
  "customer_id": "CUS999999",
  "features": {
    "xgboost": [...]
  }
}
```

**2. Search Customer**
```bash
GET http://<EC2_IP>:5000/api/v1/search?customer_id=CUS123456
```

**3. Health Check**
```bash
GET http://<EC2_IP>:5000/health
```

## Deployment

### 1. Enable Flask EC2
```bash
# terraform.tfvars
enable_flask_ec2 = true
flask_instance_type = "t3.medium"  # or t3.small for dev
```

### 2. Deploy Infrastructure
```bash
terraform init
terraform plan
terraform apply
```

### 3. Get EC2 Public IP
```bash
terraform output flask_api_public_ip
# Output: 13.250.123.45
```

### 4. Test API
```bash
curl http://13.250.123.45:5000/health

curl -X POST http://13.250.123.45:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUS123456",
    "features": {
      "traditional": [0.5, 0.8, ...],
      "social": [0.6, 0.7, ...]
    }
  }'
```

## Model Loading Strategy

### S3 Structure
```
s3://vpbank-dev-models/
├── traditional/
│   ├── model.pt          # PyTorch model
│   └── scaler.joblib     # StandardScaler
├── social/
│   ├── model.pkl         # ExtraTrees model  
│   └── scaler.joblib
└── xgboost/
    ├── model.json        # XGBoost model
    ├── scaler.joblib     # Identity scaler
    └── config.json       # Model metadata
```

### Caching Strategy
1. **First request**: Download from S3, cache in `/tmp/models/`
2. **Subsequent requests**: Use cached models (fast inference)
3. **Manual reload**: `POST /models/reload` to clear cache

## Smart Routing Logic

```python
def predict(customer_id, features):
    if customer_exists_in_db(customer_id):
        # Existing customer → Traditional + Social
        scores = [
            predict_traditional(features.traditional),
            predict_social(features.social)
        ]
        return aggregate(scores)
    else:
        # New customer → XGBoost only
        return predict_xgboost(features.xgboost)
```

## Frontend Integration

Replace old Amplify/API Gateway URLs with Flask API:

```javascript
// Old (deprecated)
const API_URL = 'https://abcd1234.execute-api.ap-southeast-1.amazonaws.com/prod/predict';

// New
const API_URL = 'http://13.250.123.45:5000/api/v1/predict';

// Predict
fetch(`${API_URL}`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    customer_id: 'CUS123456',
    features: { traditional: [...], social: [...] }
  })
});

// Search
fetch(`http://13.250.123.45:5000/api/v1/search?customer_id=CUS123456`);
```

## Production Considerations

### Security
- [ ] Add HTTPS with SSL certificate (ALB + ACM or Let's Encrypt)
- [ ] Enable CORS for frontend domain
- [ ] Add API authentication (JWT, API keys)
- [ ] Rate limiting (Flask-Limiter)
- [ ] Input validation

### Scaling
- [ ] Auto Scaling Group for EC2
- [ ] Application Load Balancer
- [ ] CloudWatch metrics & alarms
- [ ] Multiple AZs for high availability

### Monitoring
- [ ] CloudWatch Logs for Flask
- [ ] Application metrics (request count, latency)
- [ ] Model performance tracking
- [ ] DynamoDB capacity monitoring

## Migration Checklist

- [x] Remove API Gateway resources
- [x] Remove Lambda aggregate function
- [x] Remove Amplify app
- [x] Create Flask API application
- [x] Add DynamoDB table
- [x] Configure EC2 IAM roles
- [x] Test smart routing logic
- [ ] Update frontend to use new API
- [ ] Update documentation
- [ ] Train team on new architecture
- [ ] Decommission old resources

## Rollback Plan

If issues arise, the old infrastructure can be re-enabled by:
1. Uncomment resources in `apigw.tf`
2. Uncomment resources in `amplify.tf`
3. Run `terraform apply`
4. Update frontend URLs back to API Gateway

## Support

For questions or issues:
- See `flask-app/README_NEW.md` for Flask API details
- See `flask-app/ARCHITECTURE.md` for flow diagrams
- Check `flask-app/test_api.http` for example requests
