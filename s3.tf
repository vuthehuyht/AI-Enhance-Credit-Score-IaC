# S3 Data Lake for Credit Scoring POC

# Buckets:
# - raw bucket
# - cleaned bucket
# Each bucket contains three sub-prefixes: traditional/, transaction/, social/

variable "allow_force_destroy" {
  description = "Allow force destroy of S3 buckets (dangerous - use only for dev)"
  type        = bool
  default     = false
}

variable "create_prefix_objects" {
  description = "Create zero-byte prefix objects for console visibility"
  type        = bool
  default     = false
}

locals {
  raw_bucket     = "${var.project_name}-${var.environment}-raw"
  cleaned_bucket = "${var.project_name}-${var.environment}-cleaned"
}

resource "aws_s3_bucket" "raw" {
  bucket        = local.raw_bucket
  force_destroy = var.allow_force_destroy

  tags = {
    Name        = local.raw_bucket
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_s3_bucket" "cleaned" {

  bucket        = local.cleaned_bucket
  force_destroy = var.allow_force_destroy
  lifecycle {
    create_before_destroy = true
  }
  tags = {
    Name        = local.cleaned_bucket
    Project     = var.project_name
    Environment = var.environment
  }
}

# Optional: create empty folder prefixes so they show up visually in S3 UI
# (S3 has no real folders — these create 0-byte objects with trailing slash)
resource "aws_s3_object" "cleaned_prefixes" {
  count = var.create_prefix_objects ? length(["traditional/", "transaction/", "social/"]) : 0

  bucket  = aws_s3_bucket.cleaned.bucket
  key     = element(["traditional/", "transaction/", "social/"], count.index)
  content = ""
}

# Public access block to explicitly block public access
resource "aws_s3_bucket_public_access_block" "raw_block" {
  bucket = aws_s3_bucket.raw.id

  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "cleaned_block" {
  bucket = aws_s3_bucket.cleaned.id

  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

# Enable versioning for buckets
resource "aws_s3_bucket_versioning" "raw_versioning" {
  bucket = aws_s3_bucket.raw.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "cleaned_versioning" {
  bucket = aws_s3_bucket.cleaned.id

  versioning_configuration {
    status = "Enabled"
  }
}

output "raw_bucket_name" {
  value = aws_s3_bucket.raw.bucket
}

output "cleaned_bucket_name" {
  value = aws_s3_bucket.cleaned.bucket
}
