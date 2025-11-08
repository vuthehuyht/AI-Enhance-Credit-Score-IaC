// Minimal compute helpers: security group for Lambdas running in the VPC
resource "aws_security_group" "lambda" {
  name_prefix = "${var.project_name}-lambda-"
  vpc_id      = aws_vpc.main.id

  description = "Security group for Lambdas in VPC"

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [aws_vpc.main.cidr_block]
    description = "Allow internal VPC traffic"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-lambda-sg"
  }
}
# Compute resources for AI Enhanced Credit Score Infrastructure

// EC2/ALB resources removed per user's request (only serverless infra retained)

# Security Group for Lambda functions (allow outbound access and limited inbound from VPC)
// duplicate lambda SG removed (single declaration kept above)

# Application Load Balancer
// EC2 AutoScaling / ALB removed

# Data source for Amazon Linux 2 AMI
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

// EC2 instance for Flask API
variable "enable_flask_ec2" {
  description = "Enable EC2 instance for Flask API"
  type        = bool
  default     = true
}

variable "flask_instance_type" {
  description = "Instance type for Flask API server"
  type        = string
  default     = "t3.medium"
}

// Security Group for Flask EC2
resource "aws_security_group" "flask_api" {
  count       = var.enable_flask_ec2 ? 1 : 0
  name_prefix = "${var.project_name}-flask-api-"
  vpc_id      = aws_vpc.main.id
  description = "Security group for Flask API EC2 instance"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH access"
  }

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Flask API"
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = {
    Name = "${var.project_name}-flask-api-sg"
  }
}

// IAM role for EC2 to access S3
resource "aws_iam_role" "flask_ec2_role" {
  count = var.enable_flask_ec2 ? 1 : 0
  name  = "${var.project_name}-flask-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "flask_ec2_s3_access" {
  count = var.enable_flask_ec2 ? 1 : 0
  name  = "${var.project_name}-flask-ec2-s3-policy"
  role  = aws_iam_role.flask_ec2_role[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.project_name}-${var.environment}-models",
          "arn:aws:s3:::${var.project_name}-${var.environment}-models/*"
        ]
      }
    ]
  })
}

# Attach DynamoDB access policy to Flask EC2 role
resource "aws_iam_role_policy_attachment" "flask_ec2_dynamodb" {
  count      = var.enable_flask_ec2 ? 1 : 0
  role       = aws_iam_role.flask_ec2_role[0].name
  policy_arn = aws_iam_policy.dynamodb_access.arn
}

resource "aws_iam_instance_profile" "flask_ec2_profile" {
  count = var.enable_flask_ec2 ? 1 : 0
  name  = "${var.project_name}-flask-ec2-profile"
  role  = aws_iam_role.flask_ec2_role[0].name
}

// User data script to setup Flask app
data "template_file" "flask_userdata" {
  count    = var.enable_flask_ec2 ? 1 : 0
  template = file("${path.module}/flask-app/setup.sh")
}

// EC2 instance for Flask API
resource "aws_instance" "flask_api" {
  count                  = var.enable_flask_ec2 ? 1 : 0
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.flask_instance_type
  subnet_id              = aws_subnet.public[0].id
  vpc_security_group_ids = [aws_security_group.flask_api[0].id]
  iam_instance_profile   = aws_iam_instance_profile.flask_ec2_profile[0].name

  user_data = <<-EOF
              #!/bin/bash
              yum update -y
              yum install -y python3.9 python3.9-pip git

              # Create app directory
              mkdir -p /opt/flask-app
              cd /opt/flask-app

              # Note: You need to copy your Flask app files here
              # This can be done via S3, git, or CodeDeploy

              # Set environment variables
              echo "ML_DATA_BUCKET=${var.project_name}-${var.environment}-models" > /opt/flask-app/.env
              echo "AWS_DEFAULT_REGION=${data.aws_region.current.name}" >> /opt/flask-app/.env
              EOF

  tags = {
    Name = "${var.project_name}-flask-api"
  }
}

output "flask_api_public_ip" {
  value       = var.enable_flask_ec2 ? aws_instance.flask_api[0].public_ip : null
  description = "Public IP of Flask API EC2 instance"
}

output "flask_api_url" {
  value       = var.enable_flask_ec2 ? "http://${aws_instance.flask_api[0].public_ip}:5000" : null
  description = "URL of Flask API"
}

