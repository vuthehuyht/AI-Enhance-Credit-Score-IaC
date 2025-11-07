import json
import os
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

glue = boto3.client('glue')

TRAD_JOB = os.environ.get('TRADITIONAL_JOB')
SOC_JOB  = os.environ.get('SOCIAL_JOB')
RAW_BUCKET = os.environ.get('RAW_BUCKET')

PREFIX_TO_JOB = {
    'traditional/': TRAD_JOB,
    'social/': SOC_JOB
}


def lambda_handler(event, context):
    logger.info('Received event: %s', json.dumps(event))

    for record in event.get('Records', []):
        s3 = record.get('s3', {})
        bucket = s3.get('bucket', {}).get('name')
        key = s3.get('object', {}).get('key')

        if not bucket or not key:
            continue

        logger.info('New object in bucket %s key %s', bucket, key)

        # Only act on the configured raw bucket
        if RAW_BUCKET and bucket != RAW_BUCKET:
            logger.info('Skipping object in bucket %s (expected %s)', bucket, RAW_BUCKET)
            continue

        # Determine prefix
        for prefix, jobname in PREFIX_TO_JOB.items():
            if key.startswith(prefix) and jobname:
                try:
                    logger.info('Starting Glue job %s for object %s', jobname, key)
                    resp = glue.start_job_run(JobName=jobname, Arguments={'--s3_input': f's3://{bucket}/{key}'})
                    logger.info('Glue start response: %s', resp)
                except Exception as e:
                    logger.exception('Failed to start Glue job %s: %s', jobname, e)
                break

    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Processed'})
    }
