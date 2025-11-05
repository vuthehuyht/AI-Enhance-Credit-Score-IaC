import boto3
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from io import StringIO

# Config
S3_BUCKET = os.environ.get('S3_BUCKET', 'your-bucket-name')

from config import get_config

S3_PREFIX = os.environ.get('S3_PREFIX', 'cleaned/transaction/')
cfg = get_config("transaction")
S3_BUCKET = cfg["S3_BUCKET"]
S3_PREFIX = cfg["S3_PREFIX"]
MODEL_OUTPUT = cfg["MODEL_OUTPUT"]

def get_latest_csv(bucket, prefix):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]
    if not files:
        raise Exception('No CSV files found in S3 path')
    latest = sorted(files)[-1]
    return latest

def load_data_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj['Body'])

def main():
    csv_key = get_latest_csv(S3_BUCKET, S3_PREFIX)
    print(f'Loading data from: s3://{S3_BUCKET}/{csv_key}')
    df = load_data_from_s3(S3_BUCKET, csv_key)
    # Simple preprocessing: drop NA, assume last column is target
    df = df.dropna()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    # Save model to /tmp then upload to S3
    local_path = '/tmp/model.joblib'
    joblib.dump(model, local_path)
    s3.upload_file(local_path, S3_BUCKET, MODEL_OUTPUT)
    print(f'Model saved to s3://{S3_BUCKET}/{MODEL_OUTPUT}')

if __name__ == '__main__':
    main()
