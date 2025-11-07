# VPBank AI Credit Score Infrastructure# VPBank AI Enhanced Credit Score Infrastructure



Hạ tầng AWS AI/ML cho hệ thống chấm điểm tín dụng tự động sử dụng Terraform.This Terraform project creates a complete AWS infrastructure for an AI-enhanced credit scoring system for the VPBank Hackathon.



## 📋 Mục lục## Architecture Overview



- [Tổng quan](#tổng-quan)The infrastructure includes:

- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)

- [Yêu cầu](#yêu-cầu)### Core Components

- [Cài đặt](#cài-đặt)- **VPC with Public/Private Subnets**: Secure network architecture across multiple AZs

- [Cấu trúc thư mục](#cấu-trúc-thư-mục)- **Application Load Balancer**: High availability and traffic distribution

- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)- **Auto Scaling Group**: Automatic scaling based on demand

- [Các dịch vụ AWS](#các-dịch-vụ-aws)- **RDS MySQL Database**: Secure, managed database for customer and transaction data

- [Pipeline ML](#pipeline-ml)- **ElastiCache Redis**: In-memory caching for improved performance

- [Giao diện người dùng](#giao-diện-người-dùng)

- [Troubleshooting](#troubleshooting)### AI/ML Components

- **SageMaker Notebook**: For model development and training

## 🎯 Tổng quan- **Lambda Function**: Real-time credit score inference API

- **S3 Bucket**: Storage for ML models and training data

Hệ thống AI/ML hoàn chỉnh để chấm điểm tín dụng khách hàng dựa trên 3 loại dữ liệu:- **API Gateway**: RESTful API endpoint for credit scoring



- **Traditional Data**: Thông tin tài chính truyền thống (thu nhập, tài sản, nợ)### Security & Monitoring

- **Transaction Data**: Lịch sử giao dịch ngân hàng- **Security Groups**: Network-level security controls

- **Social Data**: Dữ liệu mạng xã hội và hành vi người dùng- **IAM Roles & Policies**: Fine-grained access control

- **Secrets Manager**: Secure storage of database credentials

Hệ thống tự động thu thập dữ liệu, xử lý ETL, huấn luyện model, và cung cấp API để tra cứu điểm tín dụng.- **CloudWatch**: Logging and monitoring



## 🏗️ Kiến trúc hệ thống## Prerequisites



```1. **AWS Account**: With appropriate permissions to create resources

┌─────────────┐2. **Terraform**: Version >= 1.0 installed

│   S3 Raw    │  ← Upload dữ liệu thô (CSV)3. **AWS CLI**: Configured with your credentials

└──────┬──────┘4. **Git**: For version control

       │ trigger

       ▼## Quick Start

┌─────────────┐

│ AWS Glue    │  ← ETL: Transform & Clean dữ liệu### 1. Clone and Setup

└──────┬──────┘

       │```bash

       ▼git clone <repository-url>

┌─────────────┐cd AI_Enhance_Credit_Score_Infra

│ S3 Cleaned  │  ← Dữ liệu đã xử lý```

└──────┬──────┘

       │ schedule (daily)### 2. Configure Variables

       ▼

┌─────────────┐
│ SageMaker   │  ← Huấn luyện 2 models (PyTorch & scikit-learn)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  S3 Models  │  ← Lưu trữ trained models
└──────┬──────┘
       │ auto-deploy
       ▼
┌─────────────┐
│ SageMaker   │  ← 2 Inference Endpoints
│ Endpoints   │
└──────┬──────┘
       │

       ▼terraform plan

┌─────────────┐

│ API Gateway │  ← REST API: /predict# Apply the infrastructure

└──────┬──────┘terraform apply

       │```

       ▼

┌─────────────┐## Configuration

│   Amplify   │  ← Web UI để tra cứu điểm

└─────────────┘### Core Variables (terraform.tfvars)

```

```hcl

## 📦 Yêu cầu# AWS Configuration

aws_region = "us-east-1"

### Phần mềm cần thiết:environment = "dev"

- **Terraform**: >= 1.0project_name = "vpbank-ai-credit-score"

- **AWS CLI**: >= 2.0

- **Python**: 3.9+# Network Configuration

- **Git**vpc_cidr = "10.0.0.0/16"

enable_nat_gateway = true

### AWS Credentials:

Cấu hình AWS profile trong `~/.aws/credentials`:# Compute Configuration

instance_type = "t3.medium"

```iniml_instance_type = "ml.t3.medium"

[vpbank]

aws_access_key_id = YOUR_ACCESS_KEY# Database Configuration

aws_secret_access_key = YOUR_SECRET_KEYdatabase_instance_class = "db.t3.micro"

region = ap-southeast-1multi_az = false

```deletion_protection = false

```

## 🚀 Cài đặt

### Environment-Specific Configurations

### 1. Clone repository

#### Development Environment

```bash- Smaller instance types (t3.micro, t3.small)

git clone https://github.com/yourusername/AI_Enahance_Credit_Score_Infra.git- Single AZ RDS deployment

cd AI_Enahance_Credit_Score_Infra- Minimal backup retention

```- NAT Gateway optional (set `enable_nat_gateway = false` to save costs)



### 2. Cấu hình biến môi trường#### Production Environment

- Larger instance types (t3.medium, t3.large or higher)

Tạo file `terraform.tfvars`:- Multi-AZ RDS deployment

- Extended backup retention

```hcl- Deletion protection enabled

aws_profile     = "vpbank"- Enhanced monitoring

aws_region      = "ap-southeast-1"

environment     = "dev"## API Usage

project_name    = "vpbank-ai-credit-score"

vpc_cidr        = "10.0.0.0/16"### Credit Score API Endpoint

ml_instance_type = "ml.t3.medium"

```The deployed infrastructure creates an API Gateway endpoint for credit scoring:



### 3. Initialize Terraform```bash

# Get the API Gateway URL from Terraform outputs

```bashterraform output api_gateway_url

terraform init

```# Example API call

curl -X POST https://your-api-gateway-url/predict \

### 4. Review và Apply  -H "Content-Type: application/json" \

  -d '{

```bash    "customer_data": {

# Xem trước những gì sẽ được tạo      "income": 75000,

terraform plan      "age": 32,

      "employment_length": 3,

# Tạo infrastructure      "loan_amount": 25000,

terraform apply      "credit_history_length": 5,

```      "existing_debt": 10000

    }

Quá trình deploy mất khoảng **5-10 phút**.  }'

```

## 📁 Cấu trúc thư mục

### Response Format

```

.```json

├── main.tf                       # VPC, subnets, NAT gateway, security groups{

├── variables.tf                  # Khai báo biến  "credit_score": 725,

├── s3.tf                         # S3 buckets (raw, cleaned, models)  "risk_category": "Low Risk",

├── glue.tf                       # Glue jobs + Lambda trigger  "recommendation": "Approve - Excellent creditworthiness",

├── sagemaker.tf                  # SageMaker training + deployment  "factors": {

├── apigw.tf                      # API Gateway + Lambda inference    "income": "Positive - High income",

├── amplify.tf                    # AWS Amplify frontend hosting    "employment": "Positive - Stable employment",

├── quicksight.tf                 # QuickSight visualization setup
├── config.py                     # Config cho training scripts
├── train_traditional.py          # Script huấn luyện model traditional (bao gồm cả transaction data)
├── train_traditional_pytorch.py  # Script huấn luyện model PyTorch DNN
├── inference_pytorch.py          # SageMaker inference handler cho PyTorch
├── train_social.py               # Script huấn luyện model social
├── lambda_glue_starter.py        # Lambda trigger Glue jobs
├── lambda_start_training.py      # Lambda start SageMaker training
├── lambda_deploy_model.py        # Lambda deploy SageMaker endpoints
├── lambda_aggregate_inference.py # Lambda aggregate 2 model predictions
└── glue-scripts/
    ├── transform_traditional.py  # Glue ETL script cho traditional + transaction data
    └── transform_social.py       # Glue ETL script cho social data
```

## 🎮 Hướng dẫn sử dụng

### Security Groups

### Bước 1: Upload dữ liệu thô- **ALB Security Group**: HTTP/HTTPS inbound from internet

- **EC2 Security Group**: HTTP/HTTPS from ALB, SSH from VPC

Upload file CSV vào S3 bucket raw:- **RDS Security Group**: MySQL/PostgreSQL from EC2 instances

- **Lambda Security Group**: Outbound internet access

```bash- **SageMaker Security Group**: HTTPS within VPC

aws s3 cp data.csv s3://vpbank-ai-credit-score-dev-raw/traditional/data.csv --profile vpbank

```### Database Layer

- **RDS MySQL 8.0**: Primary application database

Cấu trúc folder trong raw bucket:- **ElastiCache Redis**: Session storage and caching

```- **Secrets Manager**: Database credential storage

raw/
├── traditional/
│   └── data.csv
└── social/
    └── data.csv
```



### Bước 2: Tự động ETL## Monitoring and Logging



Khi upload file vào S3 raw bucket:### CloudWatch Integration

1. Lambda `glue_starter` được trigger tự động- **Application Logs**: Centralized logging for all services

2. Glue job tương ứng chạy ETL- **Metrics**: Custom metrics for credit scoring operations

3. Dữ liệu sạch được lưu vào `cleaned/` bucket- **Alarms**: Automated alerting for system health



### Bước 3: Huấn luyện Model### Performance Monitoring

- **RDS Performance Insights**: Database performance tracking

Model được huấn luyện tự động theo lịch (mỗi ngày 1 lần) hoặc manual trigger:- **Lambda Metrics**: Function execution metrics

- **ALB Metrics**: Load balancer health and performance

```bash

# Trigger training thủ công qua AWS CLI## Cost Optimization

aws lambda invoke \

  --function-name start-sagemaker-training \### Development Environment

  --payload '{"model_type": "traditional"}' \```hcl

  --profile vpbank \# Cost-optimized settings for development

  output.jsoninstance_type = "t3.micro"

```database_instance_class = "db.t3.micro"

ml_instance_type = "ml.t3.medium"

### Bước 4: Tra cứu điểm tín dụngenable_nat_gateway = false

multi_az = false

Sử dụng API Gateway endpoint:```



```bash### Estimated Monthly Costs (Development)

curl -X POST \- **EC2 Instances (2x t3.micro)**: ~$15

  https://YOUR_API_ID.execute-api.ap-southeast-1.amazonaws.com/prod/predict \- **RDS (db.t3.micro)**: ~$15

  -H "Content-Type: application/json" \- **Load Balancer**: ~$20

  -d '{"customer_id": "12345"}'- **ElastiCache (cache.t3.micro)**: ~$15

```- **S3 Storage**: ~$5

- **Lambda**: ~$5 (for moderate usage)

Response:- **Other Services**: ~$10

```json

{
  "customer_id": "12345",
  "traditional_score": 720,
  "social_score": 750,
  "final_score": 735,
  "risk_level": "low"
}
```

### Data Security

## ☁️ Các dịch vụ AWS- Encryption at rest for RDS and S3

- Encryption in transit for all communications

### S3 Buckets- Secrets Manager for credential management

- **raw**: Dữ liệu thô từ nguồn

- **cleaned**: Dữ liệu đã được xử lý ETL### Access Control

- **models**: Trained models (joblib format)- IAM roles with principle of least privilege

- Service-specific permissions

### AWS Glue- No hardcoded credentials

- **3 Glue Jobs**: Transform data cho 3 loại model

- **Python Shell**: Xử lý CSV, làm sạch dữ liệu## Backup and Recovery

- **Output**: Timestamped CSV files

### Database Backups

### SageMaker- Automated daily backups

- **Training Jobs**: Huấn luyện model sklearn LogisticRegression- Point-in-time recovery enabled

### SageMaker
- **Endpoints**: 2 inference endpoints (traditional, social)
- **Instance**: ml.t3.medium

### Lambda Functions
- `glue-starter`: Trigger Glue jobs khi có file mới
- `start-training`: Khởi động SageMaker training
- `deploy-model`: Deploy model lên endpoint
- `aggregate-inference`: Gọi 2 endpoints và tổng hợp kết quả

### API Gateway
- **REST API**: `/predict` endpoint
- **Method**: POST
- **Integration**: Lambda proxy

### AWS Amplify
- **Frontend Hosting**: Web UI để tra cứu điểm
```

- **Auto Deploy**: CI/CD từ GitHub- Easy instance type upgrades

- **Environment Variables**: API endpoint được inject tự động

## Deployment Environments

### QuickSight

- **Dashboards**: Visualize kết quả model### Multi-Environment Setup

- **Data Sources**: S3 manifest filesCreate separate Terraform workspaces or directories:

- **Manual Setup**: Cần tạo dashboard qua console

```bash

## 🔄 Pipeline ML# Development

terraform workspace new dev

### 1. Data Ingestionterraform apply -var-file="dev.tfvars"

```

Upload CSV → S3 Raw → Trigger Lambda → Start Glue Job# Staging

```terraform workspace new staging

terraform apply -var-file="staging.tfvars"

### 2. ETL Process

```# Production

Glue Job → Read from Raw → Transform → Write to Cleanedterraform workspace new prod

├── Remove nullsterraform apply -var-file="prod.tfvars"

├── Feature engineering```

├── Normalize data

└── Save with timestamp## Troubleshooting

```

### Common Issues

### 3. Model Training

```1. **Terraform Init Fails**

EventBridge (daily) → Lambda → SageMaker Training Job   - Ensure AWS credentials are configured

├── Read from S3 Cleaned   - Check Terraform version compatibility

├── Train sklearn model

├── Save to S3 Models2. **Resource Creation Fails**

└── Trigger deployment   - Verify AWS permissions

```   - Check resource limits in target region

   - Review Terraform error messages

### 4. Model Deployment

```3. **Application Not Accessible**

Training Complete Event → Lambda → Create/Update Endpoint   - Verify security group rules

└── Deploy model to inference endpoint   - Check NAT Gateway configuration

```   - Review route table associations



### 5. Inference### Useful Commands

```

API Request → API Gateway → Lambda```bash

├── Call Traditional Endpoint# View current infrastructure state

├── Call Transaction Endpointterraform state list

├── Call Social Endpoint

└── Aggregate & Return Score# Get specific resource information

```terraform state show aws_instance.example



## 🖥️ Giao diện người dùng# Refresh state with actual AWS resources

terraform refresh

### Setup Amplify Frontend

# Plan changes before applying

Sau khi Terraform apply, làm theo output `frontend_setup_instructions`:terraform plan -out=tfplan



1. **Tạo React App**:# Apply specific plan

```bashterraform apply tfplan

npx create-react-app credit-score-frontend```

cd credit-score-frontend

```## Contributing



2. **Tạo Component** `src/CreditScoreChecker.js`:1. Fork the repository

```javascript2. Create a feature branch

import React, { useState } from 'react';3. Make your changes

4. Test thoroughly

function CreditScoreChecker() {5. Submit a pull request

  const [customerId, setCustomerId] = useState('');

  const [result, setResult] = useState(null);## License

  const [loading, setLoading] = useState(false);

  const API_ENDPOINT = process.env.REACT_APP_API_ENDPOINT;This project is licensed under the MIT License - see the LICENSE file for details.



  const checkScore = async () => {## Support

    setLoading(true);

    try {For questions or issues:

      const response = await fetch(`${API_ENDPOINT}/predict`, {1. Check the troubleshooting section

        method: 'POST',2. Review AWS and Terraform documentation

        headers: { 'Content-Type': 'application/json' },3. Create an issue in the repository

        body: JSON.stringify({ customer_id: customerId })4. Contact the development team

      });

      const data = await response.json();---

      setResult(data);

    } catch (error) {**Note**: This infrastructure is designed for the VPBank Hackathon and should be reviewed and modified for production use according to your organization's security and compliance requirements.
      console.error('Error:', error);
      setResult({ error: error.message });
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
      <h1>🏦 VPBank Credit Score Checker</h1>
      <div style={{ marginBottom: '20px' }}>
        <input 
          type="text"
          value={customerId} 
          onChange={(e) => setCustomerId(e.target.value)}
          placeholder="Nhập Customer ID"
          style={{ padding: '10px', width: '70%', fontSize: '16px' }}
        />
        <button 
          onClick={checkScore}
          disabled={loading || !customerId}
          style={{ padding: '10px 20px', marginLeft: '10px', fontSize: '16px' }}
        >
          {loading ? 'Đang kiểm tra...' : 'Kiểm tra'}
        </button>
      </div>
      {result && (
        <div style={{ 
          padding: '20px', 
          backgroundColor: '#f5f5f5', 
          borderRadius: '8px',
          marginTop: '20px'
        }}>
          <h2>Kết quả:</h2>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default CreditScoreChecker;
```

3. **Push lên GitHub**:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/credit-score-frontend.git
git push -u origin main
```

4. **Connect Amplify**:
- Vào [Amplify Console](https://console.aws.amazon.com/amplify/)
- Click "Connect repository"
- Chọn GitHub → Authorize → Select repo
- Amplify sẽ tự động deploy!

5. **Access App**:
```
https://main.YOUR_APP_ID.amplifyapp.com
```

## 🔍 Troubleshooting

### Lỗi: "NoSuchBucket"
**Nguyên nhân**: Terraform tạo bucket nhưng chưa đợi ready.

**Giải pháp**:
```bash
terraform destroy -target=aws_s3_object.glue_script_traditional
terraform apply
```

### Lỗi: "Access Denied" khi Glue Job chạy
**Nguyên nhân**: IAM role thiếu quyền S3.

**Giải pháp**: Kiểm tra `glue.tf` - role cần có quyền:
- `s3:GetObject` trên raw bucket
- `s3:PutObject` trên cleaned bucket

### Lỗi: Training Job Failed
**Nguyên nhân**: Không có dữ liệu trong cleaned bucket.

**Giải pháp**:
1. Kiểm tra Glue job đã chạy thành công chưa
2. Xem log CloudWatch của Glue job
3. Upload dữ liệu sample để test

### Lỗi: Endpoint not found
**Nguyên nhân**: Model chưa được deploy.

**Giải pháp**:
```bash
# Trigger deployment manually
aws lambda invoke \
  --function-name deploy-sagemaker-model \
  --payload '{"model_type": "traditional"}' \
  --profile vpbank \
  output.json
```

### Lỗi: Amplify "You should provide valid token"
**Nguyên nhân**: Amplify cần GitHub token để connect repo.

**Giải pháp**: 
- Tạo app không có repo (đã fix trong code)
- Connect repo manually qua Amplify Console

## 📊 Monitoring & Logs

### CloudWatch Logs
```bash
# Xem log Glue job
aws logs tail /aws-glue/jobs/output --follow --profile vpbank

# Xem log Lambda
aws logs tail /aws/lambda/aggregate-inference --follow --profile vpbank

# Xem log SageMaker training
aws logs tail /aws/sagemaker/TrainingJobs --follow --profile vpbank
```

### Metrics
- **S3**: Object count, bucket size
- **Lambda**: Invocations, errors, duration
- **SageMaker**: Training job status, endpoint latency
- **API Gateway**: Request count, 4xx/5xx errors

## 🧹 Dọn dẹp tài nguyên

**⚠️ Cảnh báo**: Lệnh này sẽ xóa TẤT CẢ tài nguyên!

```bash
terraform destroy
```

Nếu muốn xóa từng phần:
```bash
# Xóa Amplify
terraform destroy -target=aws_amplify_app.credit_score_frontend

# Xóa SageMaker endpoints (tốn phí nhất)
terraform destroy -target=aws_lambda_function.deploy_model

# Xóa S3 buckets (cần empty trước)
aws s3 rm s3://vpbank-ai-credit-score-dev-raw --recursive --profile vpbank
terraform destroy -target=aws_s3_bucket.raw
```

## 💰 Ước tính chi phí

Chi phí hàng tháng (region ap-southeast-1):

| Dịch vụ | Chi phí/tháng | Ghi chú |
|---------|---------------|---------|
| S3 | $5-20 | Tùy lượng data |
| Glue | $10-50 | $0.44/DPU-hour |
| SageMaker Training | $20-100 | ml.t3.medium, chạy daily |
| SageMaker Endpoints | $100-200 | 2 endpoints 24/7 |
| Lambda | $1-5 | Free tier 1M requests |
| API Gateway | $3-10 | Free tier 1M calls |
| Amplify | $0-15 | Tùy traffic |
| **Tổng** | **$139-400** | |

**💡 Tips tiết kiệm**:
- Dùng SageMaker Serverless Inference thay vì real-time endpoints
- Tắt endpoints khi không dùng
- Sử dụng S3 Intelligent-Tiering
- Set up Lambda reserved concurrency

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📝 License

MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 👥 Team

- **VPBank AI Team**
- **Hackathon 2025**

---

**Made with ❤️ for VPBank Hackathon 2025**
