// Glue and S3-trigger infra for transforming raw data

// IAM role for Glue jobs
resource "aws_iam_role" "glue_service_role" {
  name = "${var.project_name}-glue-service-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect    = "Allow",
        Principal = { Service = "glue.amazonaws.com" },
        Action    = "sts:AssumeRole"
      }
    ]
  })
}

variable "glue_script_bucket" {
  description = "S3 bucket name that contains Glue scripts (e.g. <project>-ml-data-xxxx). Set this to the ML data bucket name."
  type        = string
  default     = ""
}

resource "aws_iam_role_policy_attachment" "glue_service_attach" {
  role       = aws_iam_role.glue_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

// Minimal Glue job definitions (placeholder scripts should be provided separately)
resource "aws_glue_job" "traditional_job" {
  name     = "${var.project_name}-${var.environment}-transform-traditional"
  role_arn = aws_iam_role.glue_service_role.arn
  command {
    name            = "glueetl"
    python_version  = "3"
    script_location = "s3://${var.glue_script_bucket}/glue-scripts/transform_traditional.py"
  }
  max_retries = 1
}

resource "aws_glue_job" "transaction_job" {
  name     = "${var.project_name}-${var.environment}-transform-transaction"
  role_arn = aws_iam_role.glue_service_role.arn
  command {
    name            = "glueetl"
    python_version  = "3"
    script_location = "s3://${var.glue_script_bucket}/glue-scripts/transform_transaction.py"
  }
  max_retries = 1
}

resource "aws_glue_job" "social_job" {
  name     = "${var.project_name}-${var.environment}-transform-social"
  role_arn = aws_iam_role.glue_service_role.arn
  command {
    name            = "glueetl"
    python_version  = "3"
    script_location = "s3://${var.glue_script_bucket}/glue-scripts/transform_social.py"
  }
  max_retries = 1
}

// Upload Glue scripts to the scripts bucket (assumes var.glue_script_bucket already exists)
resource "aws_s3_object" "glue_script_traditional" {
  bucket = var.glue_script_bucket
  key    = "glue-scripts/transform_traditional.py"
  source = "${path.module}/glue-scripts/transform_traditional.py"
  etag   = filemd5("${path.module}/glue-scripts/transform_traditional.py")
}

resource "aws_s3_object" "glue_script_transaction" {
  bucket = var.glue_script_bucket
  key    = "glue-scripts/transform_transaction.py"
  source = "${path.module}/glue-scripts/transform_transaction.py"
  etag   = filemd5("${path.module}/glue-scripts/transform_transaction.py")
}

resource "aws_s3_object" "glue_script_social" {
  bucket = var.glue_script_bucket
  key    = "glue-scripts/transform_social.py"
  source = "${path.module}/glue-scripts/transform_social.py"
  etag   = filemd5("${path.module}/glue-scripts/transform_social.py")
}

// IAM role for Lambda that starts Glue jobs
resource "aws_iam_role" "lambda_glue_role" {
  name = "${var.project_name}-lambda-glue-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect    = "Allow",
        Principal = { Service = "lambda.amazonaws.com" },
        Action    = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_exec" {
  role       = aws_iam_role.lambda_glue_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_glue_policy" {
  name = "${var.project_name}-lambda-glue-policy-${var.environment}"
  role = aws_iam_role.lambda_glue_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "glue:StartJobRun",
          "glue:GetJobRun",
          "glue:GetJobRuns"
        ],
        Resource = [
          aws_glue_job.traditional_job.arn,
          aws_glue_job.transaction_job.arn,
          aws_glue_job.social_job.arn
        ]
      },
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ],
        Resource = [
          aws_s3_bucket.raw.arn,
          "${aws_s3_bucket.raw.arn}/*",
          aws_s3_bucket.cleaned.arn,
          "${aws_s3_bucket.cleaned.arn}/*",
          "arn:aws:s3:::${var.glue_script_bucket}",
          "arn:aws:s3:::${var.glue_script_bucket}/*"
        ]
      }
    ]
  })
}

// Lambda function to be triggered by S3 events
data "archive_file" "lambda_glue_zip" {
  type        = "zip"
  output_path = "lambda_glue_starter.zip"
  source {
    filename = "lambda_glue_starter.py"
    content  = file("${path.module}/lambda_glue_starter.py")
  }
}

resource "aws_lambda_function" "glue_starter" {
  filename         = data.archive_file.lambda_glue_zip.output_path
  function_name    = "${var.project_name}-${var.environment}-glue-starter"
  role             = aws_iam_role.lambda_glue_role.arn
  handler          = "lambda_glue_starter.lambda_handler"
  runtime          = "python3.9"
  source_code_hash = data.archive_file.lambda_glue_zip.output_base64sha256

  vpc_config {
    subnet_ids         = aws_subnet.private[*].id
    security_group_ids = [aws_security_group.lambda.id]
  }

  environment {
    variables = {
      TRADITIONAL_JOB = aws_glue_job.traditional_job.name
      TRANSACTION_JOB = aws_glue_job.transaction_job.name
      SOCIAL_JOB      = aws_glue_job.social_job.name
      RAW_BUCKET      = aws_s3_bucket.raw.bucket
    }
  }
}

resource "aws_iam_role_policy_attachment" "lambda_glue_vpc_exec" {
  role       = aws_iam_role.lambda_glue_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

// Allow S3 to invoke the Lambda
resource "aws_lambda_permission" "allow_s3_invoke" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.glue_starter.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.raw.arn
}

// S3 notification configuration to trigger the Lambda for prefixes
resource "aws_s3_bucket_notification" "raw_notifications" {
  bucket = aws_s3_bucket.raw.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.glue_starter.arn
    events              = ["s3:ObjectCreated:*"]
    filter_suffix       = ""
    filter_prefix       = "traditional/"
  }

  lambda_function {
    lambda_function_arn = aws_lambda_function.glue_starter.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "transaction/"
  }

  lambda_function {
    lambda_function_arn = aws_lambda_function.glue_starter.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "social/"
  }

  depends_on = [aws_lambda_permission.allow_s3_invoke]
}
