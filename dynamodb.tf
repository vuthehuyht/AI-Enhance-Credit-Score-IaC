// DynamoDB table for customer data and predictions

resource "aws_dynamodb_table" "customers" {
  name           = "${var.project_name}-${var.environment}-customers"
  billing_mode   = "PAY_PER_REQUEST"  # On-demand pricing
  hash_key       = "customer_id"

  attribute {
    name = "customer_id"
    type = "S"
  }

  # Enable point-in-time recovery
  point_in_time_recovery {
    enabled = true
  }

  # Server-side encryption
  server_side_encryption {
    enabled = true
  }

  # Tags
  tags = {
    Name        = "${var.project_name}-${var.environment}-customers"
    Environment = var.environment
    ManagedBy   = "Terraform"
    Purpose     = "Customer credit score predictions"
  }
}

# IAM policy for Lambda/EC2 to access DynamoDB
resource "aws_iam_policy" "dynamodb_access" {
  name        = "${var.project_name}-${var.environment}-dynamodb-access"
  description = "Allow access to customers DynamoDB table"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:Query",
          "dynamodb:Scan",
          "dynamodb:BatchGetItem",
          "dynamodb:BatchWriteItem"
        ]
        Resource = [
          aws_dynamodb_table.customers.arn,
          "${aws_dynamodb_table.customers.arn}/index/*"
        ]
      }
    ]
  })
}

# Output table name for Flask app configuration
output "dynamodb_table_name" {
  description = "DynamoDB table name for customer data"
  value       = aws_dynamodb_table.customers.name
}

output "dynamodb_table_arn" {
  description = "DynamoDB table ARN"
  value       = aws_dynamodb_table.customers.arn
}
