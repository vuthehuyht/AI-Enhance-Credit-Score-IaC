# VPBank AI Credit Score - Infrastructure as Code

## Y Tuong Du An

### Van de
He thong cham diem tin dung truyen thong chi dua vao du lieu tai chinh → Thieu chinh xac cho khach hang moi.

### Giai phap  
He thong AI cham diem tin dung da chieu ket hop 3 nguon du lieu:

1. Traditional Data: Thong tin tai chinh (thu nhap, tiet kiem, no)
2. Transaction Data: Lich su giao dich ngan hang
3. Social Data: Hanh vi mang xa hoi

### Ket qua
- Du doan chinh xac hon voi 3 models ML chay song song
- Tu dong phan loai FICO score groups  
- API thoi gian thuc cho ung dung ngan hang
- Dashboard truc quan cho quan ly

---

## Kien Truc Tong Quan

```
DATA PIPELINE
=============

1. DATA COLLECTION
   CSV Upload → S3 Raw Bucket

2. DATA PROCESSING (AWS Glue ETL)
   S3 Raw → Glue Jobs → S3 Cleaned

3. MODEL TRAINING (SageMaker)
   S3 Cleaned → SageMaker Training Jobs
   ├── XGBoost Model (khach hang moi)
   ├── Traditional Model (PyTorch DNN)  
   └── Social Model (Sklearn ExtraTrees)
   
   → S3 Models Bucket

4. INFERENCE API (Flask on EC2)
   Flask API Server (EC2 t3.medium)
   • New Customer → XGBoost Model
   • Existing Customer → Traditional + Social
   
   Load Models from S3 → Cache in Memory
   Query/Update Customer Data → DynamoDB
   
   Response: Credit Score + FICO Group
```

---

## Workflow Chi Tiet

### Scenario 1: Khach Hang Moi
```
POST /api/v1/predict-new-customer
{
  "national_id": "001234567890",
  "income": 50000000,
  "savings": 10000000,
  "debt": 5000000,
  ...
}

Flask API:
1. Check DynamoDB → Customer chua ton tai
2. Extract 10 features tu form
3. Load XGBoost model tu S3
4. Predict → Score: 720
5. Map to FICO Group → "Good"
6. Save vao DynamoDB
7. Return response
```

### Scenario 2: Khach Hang Hien Huu
```
POST /api/v1/predict-existing-customer
{
  "national_id": "001234567890"
}

Flask API:
1. Query DynamoDB → Lay toan bo data
2. Extract traditional_features
3. Extract social_features
4. Load Traditional model (PyTorch) → Score A
5. Load Social model (Sklearn) → Score B
6. Aggregate: (A + B) / 2 = Final Score
7. Update DynamoDB
8. Return response
```

---

## Tech Stack

### Infrastructure
- AWS VPC: Network isolation
- EC2: Flask API server (t3.medium)
- S3: Data lake (raw, cleaned, models)
- DynamoDB: Customer database
- IAM: Role-based access

### ML/AI
- AWS Glue: ETL jobs
- SageMaker: Model training
- XGBoost: Gradient boosting
- PyTorch: Deep learning
- Sklearn: Random Forest

### API Layer
- Flask: RESTful API
- Gunicorn: WSGI server
- boto3: AWS SDK

---

## Quick Start

### 1. Prerequisites
```bash
- AWS Account
- Terraform >= 1.0
- AWS CLI configured
- Python 3.9+
```

### 2. Configure terraform.tfvars
```hcl
aws_profile          = "vpbank"
aws_region           = "ap-southeast-1"
environment          = "dev"
project_name         = "vpbank-ai-credit-score"
enable_flask_ec2     = true
flask_instance_type  = "t3.medium"
```

### 3. Deploy
```bash
terraform init
terraform plan
terraform apply
```

### 4. Test API
```bash
FLASK_IP=$(terraform output -raw flask_api_public_ip)

curl http://$FLASK_IP:5000/health

curl -X POST http://$FLASK_IP:5000/api/v1/predict-new-customer \
  -H "Content-Type: application/json" \
  -d '{
    "national_id": "001234567890",
    "income": 50000000,
    "savings": 10000000,
    "debt": 5000000,
    ...
  }'
```

---

## Project Structure

```
.
├── main.tf                       # VPC, networking
├── compute.tf                    # EC2, security groups
├── s3.tf                         # S3 buckets
├── dynamodb.tf                   # DynamoDB table
├── glue.tf                       # Glue ETL jobs
├── sagemaker.tf                  # SageMaker training
│
├── flask-app/                    # Flask API
│   ├── app.py                    # Main app
│   ├── requirements.txt
│   ├── API_GUIDE.md              # Full API docs
│   └── test_api.http
│
├── glue-scripts/                 # ETL scripts
├── train_xgboost.py              # XGBoost training
├── train_traditional_pytorch.py  # PyTorch training
└── train_social.py               # Sklearn training
```

---

## Cost Optimization

### Development (~$30/month)
```hcl
enable_nat_gateway = false
flask_instance_type = "t3.small"
```

### Production (~$200/month)
```hcl
enable_nat_gateway = true
flask_instance_type = "t3.medium"
# + Auto Scaling
# + Load Balancer
```

### Savings vs Old Architecture
```
Old (API Gateway + SageMaker Endpoints): $201/month
New (Flask on EC2): $36/month
SAVINGS: 82%
```

---

## Documentation

- **API Guide**: `flask-app/API_GUIDE.md`
- **Architecture**: `flask-app/ARCHITECTURE.md`
- **Migration**: `INFRASTRUCTURE_CHANGES.md`
- **Tests**: `flask-app/test_api.http`

---

## Monitoring

```bash
# Flask logs
sudo journalctl -u flask-app -f

# Glue logs
aws logs tail /aws-glue/jobs/output --follow

# Lambda logs
aws logs tail /aws/lambda/glue-starter --follow
```

---

## Cleanup

```bash
terraform destroy

# Empty S3 buckets first
aws s3 rm s3://vpbank-ai-credit-score-dev-raw --recursive
aws s3 rm s3://vpbank-ai-credit-score-dev-models --recursive
```

---

## Roadmap

### Phase 1: MVP (Completed)
- [x] Infrastructure setup
- [x] 3 ML models pipeline
- [x] Flask API
- [x] DynamoDB storage

### Phase 2: Production (In Progress)
- [ ] HTTPS/SSL
- [ ] Auto Scaling
- [ ] Load Balancer
- [ ] CI/CD pipeline

### Phase 3: Advanced (Planned)
- [ ] Real-time streaming
- [ ] A/B testing
- [ ] Auto retraining
- [ ] Fraud detection

---

**Made with ❤️ for VPBank Hackathon 2025**

VPBank AI Team
