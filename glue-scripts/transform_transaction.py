from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext
import sys
import datetime
import boto3
from urllib.parse import urlparse

args = getResolvedOptions(sys.argv, ["s3_input"])
s3_input = args['s3_input']

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
s3 = boto3.client('s3')

# Transaction transform: parse amounts, filter invalid, write single CSV part to cleaned/<date>/data.<ts>.csv
try:
    df = spark.read.option("header", "true").csv(s3_input)
    df = df.withColumn('amount', df['amount'].cast('double'))
    df_clean = df.filter(df['amount'].isNotNull())

    out_prefix = s3_input.replace('/raw/', '/cleaned/')
    if out_prefix.endswith('.csv'):
        out_prefix = out_prefix.rsplit('/', 1)[0] + '/'

    now = datetime.datetime.utcnow()
    date_str = now.strftime('%Y-%m-%d')
    ts = now.strftime('%Y%m%dT%H%M%S')

    target_folder = out_prefix + date_str + '/'
    temp_folder = target_folder + 'tmp-' + ts + '/'

    # write one CSV part
    df_clean.coalesce(1).write.mode('overwrite').option('header', 'true').csv(temp_folder)

    # determine S3 bucket and prefix
    parsed = urlparse(temp_folder)
    if parsed.scheme == 's3':
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')
    else:
        parts = temp_folder.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''

    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    part_key = None
    for obj in resp.get('Contents', []):
        k = obj['Key']
        if k.endswith('.csv') or 'part-' in k:
            part_key = k
            break

    if not part_key:
        raise RuntimeError('No part file produced in temp folder')

    target_key = (target_folder.replace(f's3://{bucket}/', '') if target_folder.startswith('s3://') else target_folder)
    target_key = target_key + f"data.{ts}.csv"
    copy_source = {'Bucket': bucket, 'Key': part_key}
    s3.copy_object(Bucket=bucket, CopySource=copy_source, Key=target_key)

    # cleanup temp objects
    for obj in resp.get('Contents', []):
        s3.delete_object(Bucket=bucket, Key=obj['Key'])

    print(f"Wrote cleaned CSV to s3://{bucket}/{target_key}")
except Exception as e:
    print(f"Transform failed: {e}")
    raise
