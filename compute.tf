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
