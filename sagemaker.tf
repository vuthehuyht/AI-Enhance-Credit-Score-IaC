// SageMaker scheduled training + serverless deployment

variable "model_names" {
  type    = list(string)
  default = ["traditional", "transaction", "social"]
}

variable "model_schedules" {
  description = "Map of model -> schedule expression (EventBridge). Use rate() or cron() syntax."
  type        = map(string)
  default = {
    traditional = "rate(1 day)"
    transaction = "rate(1 day)"
    social      = "rate(1 day)"
  }
}

variable "training_image" {
  description = "ECR image URI for training container. If empty, will use SageMaker built-in scikit-learn image"
  type        = string
  default     = ""
}

variable "inference_image" {
  description = "ECR image URI for inference container. If empty, will use SageMaker built-in scikit-learn image"
  type        = string
  default     = ""
}

locals {
  # SageMaker built-in scikit-learn image for ap-southeast-1
  # Format: <account>.dkr.ecr.<region>.amazonaws.com/sagemaker-scikit-learn:<version>
  training_image_default   = var.training_image != "" ? var.training_image : "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
  inference_image_default  = var.inference_image != "" ? var.inference_image : "121021644041.dkr.ecr.ap-southeast-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
}

variable "training_instance_type" {
  description = "Instance type for training (small)"
  type        = string
  default     = "ml.t3.medium"
}

variable "cleaned_bucket" {
  description = "S3 bucket name where cleaned data lands"
  type        = string
  default     = ""
}

variable "ml_data_bucket" {
  description = "S3 bucket for model artifacts / outputs"
  type        = string
  default     = ""
}

// IAM role for SageMaker training jobs
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${var.project_name}-sagemaker-execution-role-${var.environment}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "sagemaker.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy" "sagemaker_s3_access" {
  name = "${var.project_name}-sagemaker-s3-policy-${var.environment}"
  role = aws_iam_role.sagemaker_execution_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ],
        Resource = [
          "arn:aws:s3:::${var.project_name}-${var.environment}-cleaned",
          "arn:aws:s3:::${var.project_name}-${var.environment}-cleaned/*",
          "arn:aws:s3:::${var.project_name}-${var.environment}-models",
          "arn:aws:s3:::${var.project_name}-${var.environment}-models/*"
        ]
      }
    ]
  })
}

// IAM role for Lambda that starts training
resource "aws_iam_role" "lambda_sagemaker_start_role" {
  name = "lambda-sagemaker-start-role-${var.project_name}-${var.environment}"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "lambda.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_start_basic" {
  role       = aws_iam_role.lambda_sagemaker_start_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_start_policy" {
  name = "lambda-start-sagemaker-policy"
  role = aws_iam_role.lambda_sagemaker_start_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = ["sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob", "sagemaker:StopTrainingJob"],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = ["s3:GetObject", "s3:ListBucket", "s3:PutObject"],
        Resource = [
          "arn:aws:s3:::${var.project_name}-${var.environment}-cleaned",
          "arn:aws:s3:::${var.project_name}-${var.environment}-cleaned/*",
          "arn:aws:s3:::${var.project_name}-${var.environment}-models",
          "arn:aws:s3:::${var.project_name}-${var.environment}-models/*"
        ]
      }
    ]
  })
}

// Lambda: start training
data "archive_file" "lambda_start_zip" {
  type        = "zip"
  output_path = "lambda_start_training.zip"
  source {
    filename = "lambda_start_training.py"
    content  = file("${path.module}/lambda_start_training.py")
  }
}

resource "aws_lambda_function" "start_training" {
  filename         = data.archive_file.lambda_start_zip.output_path
  function_name    = "start-sagemaker-training"
  role             = aws_iam_role.lambda_sagemaker_start_role.arn
  handler          = "lambda_start_training.lambda_handler"
  runtime          = "python3.9"
  source_code_hash = data.archive_file.lambda_start_zip.output_base64sha256

  environment {
    variables = {
      TRAINING_IMAGE         = local.training_image_default
      TRAINING_INSTANCE_TYPE = var.training_instance_type
      CLEANED_BUCKET         = "${var.project_name}-${var.environment}-cleaned"
      ML_DATA_BUCKET         = "${var.project_name}-${var.environment}-models"
      SAGEMAKER_ROLE_ARN     = aws_iam_role.sagemaker_execution_role.arn
    }
  }
  vpc_config {
    subnet_ids         = aws_subnet.private[*].id
    security_group_ids = [aws_security_group.lambda.id]
  }
}

// Permission for EventBridge (scheduled) to invoke Lambda
resource "aws_lambda_permission" "allow_events_invoke_start" {
  statement_id  = "AllowEventInvokeStart"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.start_training.function_name
  principal     = "events.amazonaws.com"
}

// Create schedule rules for each model
resource "aws_cloudwatch_event_rule" "model_schedules" {
  for_each            = var.model_schedules
  name                = "sagemaker-schedule-${each.key}"
  schedule_expression = each.value
}

resource "aws_cloudwatch_event_target" "schedule_targets" {
  for_each  = var.model_schedules
  rule      = aws_cloudwatch_event_rule.model_schedules[each.key].name
  target_id = "target-${each.key}"
  arn       = aws_lambda_function.start_training.arn
  input     = jsonencode({ model_name = each.key })
}

// Grant EventBridge permission to invoke the lambda (additional, scoped per rule)
resource "aws_lambda_permission" "allow_rule_invoke" {
  for_each      = var.model_schedules
  statement_id  = "AllowEventInvoke-${each.key}"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.start_training.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.model_schedules[each.key].arn
}

// Lambda role for deployment (responds to training job completed events)
resource "aws_iam_role" "lambda_sagemaker_deploy_role" {
  name = "lambda-sagemaker-deploy-role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = { Service = "lambda.amazonaws.com" },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_deploy_basic" {
  role       = aws_iam_role.lambda_sagemaker_deploy_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_deploy_policy" {
  name = "lambda-deploy-sagemaker-policy"
  role = aws_iam_role.lambda_sagemaker_deploy_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "sagemaker:DescribeTrainingJob",
          "sagemaker:CreateModel",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:CreateEndpoint",
          "sagemaker:DescribeEndpoint",
          "sagemaker:CreateEndpoint",
          "sagemaker:DescribeEndpoint"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = ["s3:GetObject", "s3:ListBucket"],
        Resource = [
          "arn:aws:s3:::${var.project_name}-${var.environment}-models",
          "arn:aws:s3:::${var.project_name}-${var.environment}-models/*"
        ]
      }
    ]
  })
}

data "archive_file" "lambda_deploy_zip" {
  type        = "zip"
  output_path = "lambda_deploy_model.zip"
  source {
    filename = "lambda_deploy_model.py"
    content  = file("${path.module}/lambda_deploy_model.py")
  }
}

resource "aws_lambda_function" "deploy_model" {
  filename         = data.archive_file.lambda_deploy_zip.output_path
  function_name    = "deploy-sagemaker-model"
  role             = aws_iam_role.lambda_sagemaker_deploy_role.arn
  handler          = "lambda_deploy_model.lambda_handler"
  runtime          = "python3.9"
  source_code_hash = data.archive_file.lambda_deploy_zip.output_base64sha256

  environment {
    variables = {
      INFERENCE_IMAGE = local.inference_image_default
      ML_DATA_BUCKET  = "${var.project_name}-${var.environment}-models"
    }
  }
  vpc_config {
    subnet_ids         = aws_subnet.private[*].id
    security_group_ids = [aws_security_group.lambda.id]
  }
}

resource "aws_iam_role_policy_attachment" "lambda_start_vpc_exec" {
  role       = aws_iam_role.lambda_sagemaker_start_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_deploy_vpc_exec" {
  role       = aws_iam_role.lambda_sagemaker_deploy_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

// Permission for EventBridge to invoke deploy lambda
resource "aws_lambda_permission" "allow_events_invoke_deploy" {
  statement_id  = "AllowEventInvokeDeploy"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.deploy_model.function_name
  principal     = "events.amazonaws.com"
}

// Event rule to catch SageMaker Training job completion and trigger deployment
resource "aws_cloudwatch_event_rule" "sagemaker_training_complete" {
  name = "sagemaker-training-complete-rule"
  event_pattern = jsonencode({
    "source" : ["aws.sagemaker"],
    "detail-type" : ["SageMaker Training Job State Change"],
    "detail" : { "TrainingJobStatus" : ["Completed"] }
  })
}

resource "aws_cloudwatch_event_target" "deploy_target" {
  rule = aws_cloudwatch_event_rule.sagemaker_training_complete.name
  arn  = aws_lambda_function.deploy_model.arn
}

resource "aws_lambda_permission" "allow_eventbridge_deploy" {
  statement_id  = "AllowEventBridgeInvokeDeploy"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.deploy_model.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.sagemaker_training_complete.arn
}
