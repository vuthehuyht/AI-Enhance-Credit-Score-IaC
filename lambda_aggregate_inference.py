import os
import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

runtime = boto3.client('sagemaker-runtime')

ENDPOINT_TRAD = os.environ.get('ENDPOINT_TRADITIONAL')
ENDPOINT_TRANS = os.environ.get('ENDPOINT_TRANSACTION')
ENDPOINT_SOC = os.environ.get('ENDPOINT_SOCIAL')

# Simple demo: call each endpoint with same payload and average numeric results

def invoke_endpoint(endpoint_name, payload):
    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        body = response['Body'].read().decode('utf-8')
        return json.loads(body)
    except Exception as e:
        logger.exception('Endpoint %s invocation failed: %s', endpoint_name, e)
        return {'error': str(e)}


def lambda_handler(event, context):
    logger.info('Received API event: %s', json.dumps(event))

    # Parse body
    if 'body' in event:
        try:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        except Exception:
            body = {}
    else:
        body = event

    payload = body.get('input', {})

    results = []

    if ENDPOINT_TRAD:
        res = invoke_endpoint(ENDPOINT_TRAD, payload)
        results.append({'model': 'traditional', 'result': res})

    if ENDPOINT_TRANS:
        res = invoke_endpoint(ENDPOINT_TRANS, payload)
        results.append({'model': 'transaction', 'result': res})

    if ENDPOINT_SOC:
        res = invoke_endpoint(ENDPOINT_SOC, payload)
        results.append({'model': 'social', 'result': res})

    # Demo aggregation: if model returns numeric 'score' take average
    scores = [r['result'].get('score') for r in results if isinstance(r['result'], dict) and 'score' in r['result']]
    agg_score = sum(scores)/len(scores) if scores else None

    response = {
        'models': results,
        'aggregated_score': agg_score
    }

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(response)
    }
