import os
import json
import time
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker = boto3.client('sagemaker')

TRAINING_IMAGE = os.environ.get('TRAINING_IMAGE')
TRAINING_INSTANCE_TYPE = os.environ.get('TRAINING_INSTANCE_TYPE')
CLEANED_BUCKET = os.environ.get('CLEANED_BUCKET')
ML_DATA_BUCKET = os.environ.get('ML_DATA_BUCKET')
SAGEMAKER_ROLE_ARN = os.environ.get('SAGEMAKER_ROLE_ARN')

def lambda_handler(event, context):
    logger.info('Start training handler invoked with event: %s', json.dumps(event))

    # Determine model name from EventBridge input or default
    model_name = None
    if isinstance(event, dict):
        if 'model_name' in event:
            model_name = event['model_name']
        elif 'detail' in event and 'model_name' in event['detail']:
            model_name = event['detail']['model_name']

    model_name = model_name or 'traditional'

    # Create a unique training job name
    timestamp = int(time.time())
    training_job_name = f"{model_name}-training-{timestamp}"

    input_s3_uri = f"s3://{CLEANED_BUCKET}/{model_name}/"
    output_s3_uri = f"s3://{ML_DATA_BUCKET}/{model_name}/output/"

    logger.info('Starting training job %s using input %s', training_job_name, input_s3_uri)

    try:
        response = sagemaker.create_training_job(
            TrainingJobName=training_job_name,
            AlgorithmSpecification={
                'TrainingImage': TRAINING_IMAGE,
                'TrainingInputMode': 'File'
            },
            RoleArn=SAGEMAKER_ROLE_ARN,
            InputDataConfig=[
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': input_s3_uri,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    }
                }
            ],
            OutputDataConfig={'S3OutputPath': output_s3_uri},
            ResourceConfig={
                'InstanceType': TRAINING_INSTANCE_TYPE,
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            StoppingCondition={'MaxRuntimeInSeconds': 3600}
        )

        logger.info('CreateTrainingJob response: %s', response)
        return {'statusCode': 200, 'body': json.dumps({'training_job_name': training_job_name})}

    except Exception as e:
        logger.exception('Failed to start training job: %s', e)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
