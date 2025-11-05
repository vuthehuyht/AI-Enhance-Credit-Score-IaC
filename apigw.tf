// API Gateway + Lambda to call three SageMaker endpoints and aggregate results

variable "endpoint_traditional" {
  description = "SageMaker endpoint name for traditional model"
  type        = string
  default     = ""
}

variable "endpoint_transaction" {
  description = "SageMaker endpoint name for transaction model"
  type        = string
  default     = ""
}

variable "endpoint_social" {
  description = "SageMaker endpoint name for social model"
  type        = string
  default     = ""
}

locals {
  endpoint_traditional_default = var.endpoint_traditional != "" ? var.endpoint_traditional : "${var.project_name}-${var.environment}-traditional-endpoint"
  endpoint_transaction_default = var.endpoint_transaction != "" ? var.endpoint_transaction : "${var.project_name}-${var.environment}-transaction-endpoint"
  endpoint_social_default      = var.endpoint_social != "" ? var.endpoint_social : "${var.project_name}-${var.environment}-social-endpoint"
}

// IAM role for aggregate lambda
resource "aws_iam_role" "lambda_aggregate_role" {
  name = "lambda-aggregate-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
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

resource "aws_iam_role_policy_attachment" "lambda_aggregate_basic" {
  role       = aws_iam_role.lambda_aggregate_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_aggregate_sagemaker" {
  name = "lambda-aggregate-sagemaker"
  role = aws_iam_role.lambda_aggregate_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      { Effect = "Allow", Action = ["sagemaker:InvokeEndpoint"], Resource = "*" }
    ]
  })
}

// Lambda function archive
data "archive_file" "lambda_aggregate_zip" {
  type        = "zip"
  output_path = "lambda_aggregate_inference.zip"
  source {
    filename = "lambda_aggregate_inference.py"
    content  = file("${path.module}/lambda_aggregate_inference.py")
  }
}

resource "aws_lambda_function" "aggregate_inference" {
  filename         = data.archive_file.lambda_aggregate_zip.output_path
  function_name    = "aggregate-inference"
  role             = aws_iam_role.lambda_aggregate_role.arn
  handler          = "lambda_aggregate_inference.lambda_handler"
  runtime          = "python3.9"
  source_code_hash = data.archive_file.lambda_aggregate_zip.output_base64sha256

  environment {
    variables = {
      ENDPOINT_TRADITIONAL = local.endpoint_traditional_default
      ENDPOINT_TRANSACTION = local.endpoint_transaction_default
      ENDPOINT_SOCIAL      = local.endpoint_social_default
    }
  }
  vpc_config {
    subnet_ids         = aws_subnet.private[*].id
    security_group_ids = [aws_security_group.lambda.id]
  }
}

resource "aws_iam_role_policy_attachment" "lambda_aggregate_vpc_exec" {
  role       = aws_iam_role.lambda_aggregate_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

resource "aws_lambda_permission" "apigw_invoke" {
  statement_id  = "AllowAPIGatewayInvokeAggregate"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.aggregate_inference.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.model_ensemble.execution_arn}/*/*"
}

// API Gateway REST API
resource "aws_api_gateway_rest_api" "model_ensemble" {
  name        = "model-ensemble-api"
  description = "API that aggregates predictions from 3 SageMaker models"
  endpoint_configuration {
    types = ["REGIONAL"]
  }
}

resource "aws_api_gateway_resource" "predict" {
  rest_api_id = aws_api_gateway_rest_api.model_ensemble.id
  parent_id   = aws_api_gateway_rest_api.model_ensemble.root_resource_id
  path_part   = "predict"
}

resource "aws_api_gateway_method" "predict_post" {
  rest_api_id   = aws_api_gateway_rest_api.model_ensemble.id
  resource_id   = aws_api_gateway_resource.predict.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "predict_integration" {
  rest_api_id = aws_api_gateway_rest_api.model_ensemble.id
  resource_id = aws_api_gateway_resource.predict.id
  http_method = aws_api_gateway_method.predict_post.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.aggregate_inference.invoke_arn
}

resource "aws_api_gateway_deployment" "model_ensemble_deployment" {
  depends_on  = [aws_api_gateway_integration.predict_integration]
  rest_api_id = aws_api_gateway_rest_api.model_ensemble.id
  triggers = {
    redeployment = sha1(jsonencode([aws_api_gateway_integration.predict_integration.id]))
  }
}

resource "aws_api_gateway_stage" "prod" {
  deployment_id = aws_api_gateway_deployment.model_ensemble_deployment.id
  rest_api_id   = aws_api_gateway_rest_api.model_ensemble.id
  stage_name    = "prod"
}

data "aws_region" "current" {}

output "aggregate_api_url" {
  value = "https://${aws_api_gateway_rest_api.model_ensemble.id}.execute-api.${data.aws_region.current.name}.amazonaws.com/${aws_api_gateway_stage.prod.stage_name}/predict"
}
