import os
import json
import logging
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker = boto3.client('sagemaker')

INFERENCE_IMAGE = os.environ.get('INFERENCE_IMAGE')
ML_DATA_BUCKET = os.environ.get('ML_DATA_BUCKET')

def lambda_handler(event, context):
    logger.info('Deploy handler received event: %s', json.dumps(event))

    # Extract training job info
    detail = event.get('detail', {}) if isinstance(event, dict) else {}
    training_job_name = detail.get('TrainingJobName') or detail.get('trainingJobName')
    model_name = training_job_name.split('-')[0] if training_job_name else 'model'

    # Create model and serverless endpoint
    try:
        model_artifact = detail.get('ModelArtifacts', {}).get('S3ModelArtifacts') if detail else None
        primary_container = {
            'Image': INFERENCE_IMAGE,
            'ModelDataUrl': model_artifact
        }

        create_model_response = sagemaker.create_model(
            ModelName=f"{model_name}-model-{int(context.aws_request_id[:8], 16) if context and hasattr(context, 'aws_request_id') else '1'}",
            ExecutionRoleArn=os.environ.get('SAGEMAKER_ROLE_ARN'),
            PrimaryContainer=primary_container
        )
        logger.info('create_model_response: %s', create_model_response)

        # Create serverless endpoint config
        endpoint_config_name = f"{model_name}-endpoint-config"
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': create_model_response['ModelName'],
                'InstanceType': 'ml.m5.large',
                'InitialInstanceCount': 1,
            }]
        )

        endpoint_name = f"{model_name}-endpoint"
        create_endpoint_response = sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )

        logger.info('create_endpoint_response: %s', create_endpoint_response)
        return {'statusCode': 200, 'body': json.dumps({'endpoint': endpoint_name})}

    except Exception as e:
        logger.exception('Failed to deploy model: %s', e)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
