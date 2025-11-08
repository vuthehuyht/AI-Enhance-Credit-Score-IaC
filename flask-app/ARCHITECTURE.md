# Flask API Flow Architecture

## Request Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Request                         │
│                  POST /api/v1/predict                       │
│                { customer_id, features }                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask API Server                         │
│                     (EC2 Instance)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Check Customer in DynamoDB                       │
│         customer_table.get_item(customer_id)                │
└────────────┬────────────────────────┬───────────────────────┘
             │                        │
    ┌────────▼─────────┐    ┌────────▼─────────┐
    │   EXISTS = True  │    │  EXISTS = False  │
    │  (Old Customer)  │    │  (New Customer)  │
    └────────┬─────────┘    └────────┬─────────┘
             │                        │
             ▼                        ▼
┌──────────────────────┐    ┌──────────────────────┐
│ Load Models from S3  │    │ Load Model from S3   │
│ • traditional/       │    │ • xgboost/           │
│ • social/            │    │                      │
└──────────┬───────────┘    └──────────┬───────────┘
           │                           │
           ▼                           ▼
┌──────────────────────┐    ┌──────────────────────┐
│ Run Inference        │    │ Run Inference        │
│ • PyTorch (trad)     │    │ • XGBoost            │
│ • ExtraTrees (soc)   │    │                      │
└──────────┬───────────┘    └──────────┬───────────┘
           │                           │
           ▼                           │
┌──────────────────────┐               │
│ Aggregate Scores     │               │
│ avg([trad, social])  │               │
└──────────┬───────────┘               │
           │                           │
           └───────────┬───────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Save Prediction to DynamoDB                    │
│  {customer_id, score, timestamp, models_used, type}         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Return Response                            │
│                                                             │
│  Existing: {customer_type, aggregated_score, models}        │
│  New: {customer_type, score, model: "xgboost"}              │
└─────────────────────────────────────────────────────────────┘
```

## Search Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Request                         │
│         GET /api/v1/search?customer_id=CUS123               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask API Server                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Query DynamoDB Table                           │
│         customer_table.get_item(customer_id)                │
└────────────┬────────────────────────┬───────────────────────┘
             │                        │
    ┌────────▼─────────┐    ┌────────▼─────────┐
    │    Item Found    │    │  Item Not Found  │
    └────────┬─────────┘    └────────┬─────────┘
             │                        │
             ▼                        ▼
┌──────────────────────┐    ┌──────────────────────┐
│ Return Prediction    │    │ Return Not Found     │
│ • timestamp          │    │ {found: false}       │
│ • credit_score       │    │                      │
│ • models_used        │    │                      │
│ • raw_predictions    │    │                      │
└──────────────────────┘    └──────────────────────┘
```

## Data Flow

### 1. Model Loading (First Request)
```
S3 Bucket (vpbank-*-models)
    ├── traditional/
    │   ├── model.pt          ──┐
    │   └── scaler.joblib       │
    │                           ├──> Load & Cache in Memory
    ├── social/                 │
    │   ├── model.pkl           │
    │   └── scaler.joblib     ──┘
    │
    └── xgboost/
        ├── model.json        ──> Load & Cache in Memory
        ├── scaler.joblib
        └── config.json
```

### 2. Database Schema
```
DynamoDB Table: vpbank-*-customers
─────────────────────────────────────────────────
| customer_id (PK) | String | "CUS123456"      |
| prediction_timestamp | String | ISO 8601      |
| credit_score     | Number | 735.4            |
| models_used      | List   | ["trad", "soc"]  |
| prediction_type  | String | "multi_model"    |
| raw_predictions  | List   | [{model, score}] |
─────────────────────────────────────────────────
```

## Model Decision Logic

```python
def predict(customer_id, features):
    if check_customer_exists(customer_id):
        # Existing customer
        traditional_score = predict_traditional(features.traditional)
        social_score = predict_social(features.social)
        final_score = (traditional_score + social_score) / 2
        return {
            'customer_type': 'existing',
            'aggregated_score': final_score,
            'models': ['traditional', 'social']
        }
    else:
        # New customer
        xgboost_score = predict_xgboost(features.xgboost)
        return {
            'customer_type': 'new',
            'score': xgboost_score,
            'model': 'xgboost'
        }
```

## AWS Services Integration

```
┌──────────────┐
│   EC2        │
│ Flask App    │◄────┐
└──────┬───────┘     │
       │             │ IAM Role
       │             │ Permissions
       ├─────────────┼──────────────────┐
       │             │                  │
       ▼             ▼                  ▼
┌──────────┐  ┌──────────┐      ┌──────────┐
│ S3       │  │ DynamoDB │      │ CloudWatch│
│ Models   │  │ Customers│      │ Logs     │
└──────────┘  └──────────┘      └──────────┘
```

## Performance Considerations

1. **Model Caching**: Models loaded once on first request, cached in memory
2. **Connection Pooling**: DynamoDB connections reused across requests
3. **Async I/O**: Non-blocking S3/DynamoDB operations (future enhancement)
4. **Horizontal Scaling**: Multiple EC2 instances behind Load Balancer

## Error Handling

```
┌─────────────┐
│   Request   │
└──────┬──────┘
       │
       ▼
┌────────────────┐     Error      ┌────────────────┐
│ Validate Input │────────────────>│ Return 400     │
└──────┬─────────┘                 └────────────────┘
       │
       ▼
┌────────────────┐     Error      ┌────────────────┐
│ Check DB       │────────────────>│ Treat as New   │
└──────┬─────────┘                 │ Customer       │
       │                           └────────────────┘
       ▼
┌────────────────┐     Error      ┌────────────────┐
│ Load Models    │────────────────>│ Return 500     │
└──────┬─────────┘                 │ with Details   │
       │                           └────────────────┘
       ▼
┌────────────────┐     Error      ┌────────────────┐
│ Predict        │────────────────>│ Try Fallback   │
└──────┬─────────┘                 │ Model          │
       │                           └────────────────┘
       ▼
┌────────────────┐
│ Return Result  │
└────────────────┘
```
