# Flask API Guide - Credit Score Prediction

## Tổng quan

Flask API cung cấp 3 endpoints chính để đánh giá tín dụng:

1. **New Customer**: Khách hàng mới → XGBoost model
2. **Existing Customer**: Khách hàng hiện hữu → Traditional + Social models
3. **Search**: Tìm kiếm thông tin khách hàng

---

## Data Flow

### 1. New Customer Prediction

```
User Input (Form) → XGBoost Features → XGBoost Model → Score → Save to DB
```

**Workflow:**
1. User gửi form với thông tin cơ bản (CMND, thu nhập, tiết kiệm, nợ, etc.)
2. API kiểm tra khách hàng đã tồn tại chưa (DynamoDB lookup)
3. Nếu là khách hàng mới:
   - Chuẩn bị 10 features cho XGBoost từ form data
   - Chạy XGBoost model prediction
   - Tính FICO group (Poor/Fair/Good/Very Good/Exceptional)
   - Lưu vào DynamoDB với thông tin: `national_id`, `income`, `savings`, `debt`, `credit_score`, `credit_group`

**Example Request:**
```json
POST /api/v1/predict-new-customer
{
  "full_name": "Nguyen Van A",
  "national_id": "001234567890",
  "email": "nguyenvana@example.com",
  "phone_number": "0912345678",
  "income": 50000000,
  "savings": 10000000,
  "debt": 5000000,
  "cat_dependents": "Yes",
  "cat_mortgage": "No",
  "cat_savings_account": "Yes",
  "cat_credit_card": "Yes"
}
```

**Response:**
```json
{
  "predicted_score": 720.5,
  "predicted_group": "Good",
  "customer_info": {
    "full_name": "Nguyen Van A",
    "national_id": "001234567890",
    "email": "nguyenvana@example.com",
    "phone_number": "0912345678"
  },
  "timestamp": "2025-11-08T10:30:00Z"
}
```

---

### 2. Existing Customer Prediction

```
Customer ID → Retrieve from DB → Traditional Features + Social Features → 
Traditional Model + Social Model → Aggregate Scores → Update DB
```

**Workflow:**
1. User gửi identifier (national_id/email/phone)
2. API tìm kiếm khách hàng trong DynamoDB
3. Nếu tìm thấy:
   - Lấy toàn bộ thông tin customer từ DB
   - Chuẩn bị features cho Traditional model từ `traditional_features` field
   - Chuẩn bị features cho Social model từ `social_features` field
   - Chạy cả 2 models song song
   - Tính trung bình cộng (aggregated score)
   - Cập nhật `credit_score` và `credit_group` mới vào DB

**Example Request:**
```json
POST /api/v1/predict-existing-customer
{
  "national_id": "001234567890"
}
```

**Response:**
```json
{
  "predicted_score": 735.4,
  "predicted_group": "Very Good",
  "customer_info": {
    "full_name": "Nguyen Van A",
    "national_id": "001234567890",
    "email": "nguyenvana@example.com",
    "phone_number": "0912345678"
  },
  "models_used": ["traditional", "social"],
  "model_scores": {
    "traditional": 750.5,
    "social": 720.3
  },
  "timestamp": "2025-11-08T10:30:00Z"
}
```

---

### 3. Search Customer

```
Identifier (national_id/email/phone) → DynamoDB Query → Return Customer Data
```

**Example Request:**
```http
GET /api/v1/search?national_id=001234567890
```

**Response:**
```json
{
  "found": true,
  "customer": {
    "full_name": "Nguyen Van A",
    "national_id": "001234567890",
    "email": "nguyenvana@example.com",
    "phone_number": "0912345678",
    "credit_score": 735.4,
    "credit_group": "Very Good",
    "last_prediction": "2025-11-08T10:30:00Z",
    "created_at": "2025-11-01T08:00:00Z"
  }
}
```

---

## DynamoDB Schema

```json
{
  "national_id": "001234567890",              // Primary Key
  "full_name": "Nguyen Van A",
  "email": "nguyenvana@example.com",          // GSI
  "phone_number": "0912345678",               // GSI
  "created_at": "2025-11-01T08:00:00Z",
  "last_prediction": "2025-11-08T10:30:00Z",
  "credit_score": 735.4,
  "credit_group": "Very Good",
  
  // Financial data (from new customer form)
  "income": 50000000,
  "savings": 10000000,
  "debt": 5000000,
  
  // Model features (for existing customer predictions)
  "traditional_features": [/* array of floats */],
  "social_features": [/* array of floats */],
  
  // History
  "prediction_history": [
    {
      "predicted_score": 735.4,
      "predicted_group": "Very Good",
      "timestamp": "2025-11-08T10:30:00Z"
    }
  ]
}
```

---

## Model Types

| Model Name | Type | Input Features | Usage |
|------------|------|----------------|-------|
| **xgboost** | XGBoost | 10 features (income, savings, debt, etc.) | New customers |
| **traditional** | PyTorch DNN | Variable (stored in DB) | Existing customers |
| **social** | Sklearn | Variable (stored in DB) | Existing customers |

---

## Feature Preparation

### XGBoost Features (10 features)
```python
[
  income,                          # Thu nhập
  savings,                         # Tiết kiệm
  debt,                           # Tổng nợ
  dependents,                     # Có người phụ thuộc (0/1)
  mortgage,                       # Có thế chấp (0/1)
  savings_account,                # Có tài khoản tiết kiệm (0/1)
  credit_card,                    # Có thẻ tín dụng (0/1)
  savings_ratio,                  # savings / income
  debt_to_income_ratio,           # debt / income
  net_worth                       # income - debt
]
```

### Traditional Features
- Lưu trong DB field: `traditional_features`
- TODO: Cần mapping với training data

### Social Features
- Lưu trong DB field: `social_features`
- TODO: Cần mapping với social media data crawling

---

## FICO Score Groups

| Score Range | Group | Description |
|-------------|-------|-------------|
| 800 - 850 | Exceptional | Tín dụng xuất sắc |
| 740 - 799 | Very Good | Tín dụng rất tốt |
| 670 - 739 | Good | Tín dụng tốt |
| 580 - 669 | Fair | Tín dụng trung bình |
| 300 - 579 | Poor | Tín dụng kém |

---

## Error Handling

### Customer Already Exists (400)
```json
{
  "error": "Customer already exists. Please use the existing customer prediction endpoint.",
  "existing_customer": {
    "national_id": "001234567890",
    "full_name": "Nguyen Van A"
  }
}
```

### Customer Not Found (404)
```json
{
  "error": "Customer not found. Please use the new customer prediction endpoint.",
  "suggestion": "POST /api/v1/predict-new-customer"
}
```

### Missing Required Fields (400)
```json
{
  "error": "Missing required fields: income, savings, debt"
}
```

---

## Environment Variables

```bash
ML_DATA_BUCKET=vpbank-hackathon-dev-models
CUSTOMER_TABLE_NAME=vpbank-hackathon-dev-customers
AWS_REGION=ap-southeast-1
```

---

## Next Steps

### For Production:
1. **Feature Engineering**: Map actual training features to DB schema
2. **Social Data Collection**: Implement social media crawler
3. **Feature Storage**: Update `save_customer_to_db()` to store `traditional_features` and `social_features`
4. **Batch Processing**: Add endpoint to update all existing customers periodically
5. **Monitoring**: Add CloudWatch metrics and alerts
6. **Authentication**: Add API key or JWT authentication
7. **Rate Limiting**: Implement request throttling
8. **Caching**: Add Redis for frequent queries
