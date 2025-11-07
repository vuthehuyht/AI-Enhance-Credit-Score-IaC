// QuickSight for visualizing ML model results and training metrics

variable "quicksight_user_arn" {
  description = "ARN of QuickSight user (e.g., arn:aws:quicksight:region:account-id:user/namespace/username)"
  type        = string
  default     = ""
}

// Get current AWS account ID for QuickSight ARNs
data "aws_caller_identity" "current" {}

locals {
  quicksight_principal = var.quicksight_user_arn != "" ? var.quicksight_user_arn : "arn:aws:quicksight:${var.aws_region}:${data.aws_caller_identity.current.account_id}:user/default/Admin"
}

// IAM role for QuickSight to access S3
resource "aws_iam_role" "quicksight_s3_role" {
  name = "${var.project_name}-quicksight-s3-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "quicksight.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "quicksight_s3_policy" {
  name = "${var.project_name}-quicksight-s3-policy-${var.environment}"
  role = aws_iam_role.quicksight_s3_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:ListBucket",
          "s3:ListBucketVersions"
        ]
        Resource = [
          "${aws_s3_bucket.cleaned.arn}",
          "${aws_s3_bucket.cleaned.arn}/*",
          "${aws_s3_bucket.models.arn}",
          "${aws_s3_bucket.models.arn}/*"
        ]
      }
    ]
  })
}

// Create manifest files for QuickSight to read S3 data
resource "aws_s3_object" "quicksight_cleaned_manifest" {
  bucket  = aws_s3_bucket.cleaned.bucket
  key     = "quicksight/manifest.json"
  content = jsonencode({
    fileLocations = [
      {
        URIPrefixes = [
          "s3://${aws_s3_bucket.cleaned.bucket}/traditional/",
          "s3://${aws_s3_bucket.cleaned.bucket}/social/"
        ]
      }
    ]
    globalUploadSettings = {
      format = "CSV"
      delimiter = ","
      textqualifier = "\""
      containsHeader = "true"
    }
  })
}

resource "aws_s3_object" "quicksight_models_manifest" {
  bucket  = aws_s3_bucket.models.bucket
  key     = "quicksight/manifest.json"
  content = jsonencode({
    fileLocations = [
      {
        URIPrefixes = [
          "s3://${aws_s3_bucket.models.bucket}/traditional/",
          "s3://${aws_s3_bucket.models.bucket}/social/"
        ]
      }
    ]
    globalUploadSettings = {
      format = "CSV"
      delimiter = ","
      textqualifier = "\""
      containsHeader = "true"
    }
  })
}

// Output QuickSight setup instructions
output "quicksight_setup_instructions" {
  value = <<-EOT
    QuickSight Setup Instructions:

    1. Go to AWS QuickSight console: https://${var.aws_region}.quicksight.aws.amazon.com/
    2. Sign up for QuickSight if not already done (Enterprise Edition recommended)
    3. Grant QuickSight access to S3 buckets:
       - Cleaned bucket: ${aws_s3_bucket.cleaned.bucket}
       - Models bucket: ${aws_s3_bucket.models.bucket}

    4. Create Data Sources manually:
       a. Cleaned Data Source:
          - Name: "Cleaned Credit Score Data"
          - Type: S3
          - Manifest: s3://${aws_s3_bucket.cleaned.bucket}/quicksight/manifest.json

       b. Model Results Data Source:
          - Name: "Model Training Results"
          - Type: S3
          - Manifest: s3://${aws_s3_bucket.models.bucket}/quicksight/manifest.json

    5. Create Datasets for each model:
       - Traditional Model Data
       - Social Model Data

    6. Create Dashboard with visualizations:
       - Credit score distribution
       - Model prediction accuracy
       - Feature importance charts
       - Time series of predictions

    Note: Terraform AWS provider has limited QuickSight support.
    Most resources need to be created manually in QuickSight console.
  EOT
  description = "Instructions for setting up QuickSight dashboard"
}

output "quicksight_s3_role_arn" {
  value       = aws_iam_role.quicksight_s3_role.arn
  description = "IAM role ARN for QuickSight to access S3"
}

output "quicksight_manifest_files" {
  value = {
    cleaned = "s3://${aws_s3_bucket.cleaned.bucket}/quicksight/manifest.json"
    models  = "s3://${aws_s3_bucket.models.bucket}/quicksight/manifest.json"
  }
  description = "S3 manifest file locations for QuickSight"
}

