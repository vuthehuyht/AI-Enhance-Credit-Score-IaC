"""
AWS Glue transformation script for XGBoost model data
Transforms raw data into format suitable for XGBoost training
"""
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'SOURCE_BUCKET', 'TARGET_BUCKET'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

print(f"Starting XGBoost data transformation job: {args['JOB_NAME']}")
print(f"Source bucket: {args['SOURCE_BUCKET']}")
print(f"Target bucket: {args['TARGET_BUCKET']}")

# Read raw data from S3
source_path = f"s3://{args['SOURCE_BUCKET']}/raw/xgboost/"
print(f"Reading data from: {source_path}")

try:
    # Read CSV files
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(source_path)

    print(f"Schema before transformation:")
    df.printSchema()
    print(f"Row count: {df.count()}")

    # === Data Cleaning ===
    # Remove duplicates
    df = df.dropDuplicates()
    print(f"Row count after removing duplicates: {df.count()}")

    # Remove null values
    df = df.dropna()
    print(f"Row count after removing nulls: {df.count()}")

    # === Feature Engineering for XGBoost ===
    # XGBoost works well with numeric features

    # Convert categorical columns to numeric if needed
    categorical_cols = [field.name for field in df.schema.fields if field.dataType.typeName() == 'string']
    print(f"Categorical columns found: {categorical_cols}")

    # One-hot encode categorical variables
    for col_name in categorical_cols:
        if col_name.lower() not in ['creditscore', 'credit_score']:  # Skip target column
            # Get distinct values
            distinct_values = df.select(col_name).distinct().rdd.flatMap(lambda x: x).collect()
            print(f"Encoding column '{col_name}' with {len(distinct_values)} unique values")

            # Create dummy columns
            for val in distinct_values:
                if val is not None:
                    safe_val = str(val).replace(' ', '_').replace('-', '_').replace('.', '_')
                    new_col_name = f"{col_name}_{safe_val}"
                    df = df.withColumn(new_col_name, F.when(F.col(col_name) == val, 1).otherwise(0))

            # Drop original categorical column
            df = df.drop(col_name)

    # === Ensure numeric types ===
    # Convert all columns to double except target
    target_col = None
    for col in df.columns:
        if col.lower() in ['creditscore', 'credit_score']:
            target_col = col
            df = df.withColumn(col, F.col(col).cast(DoubleType()))
        else:
            df = df.withColumn(col, F.col(col).cast(DoubleType()))

    if target_col is None:
        raise Exception("Target column (CreditScore) not found in data")

    # === Feature Engineering: Create ratio features ===
    # These are commonly useful for credit scoring

    # Example: If we have INCOME, DEBT, SAVINGS columns
    if 'INCOME' in df.columns and 'DEBT' in df.columns:
        df = df.withColumn('DEBT_TO_INCOME_RATIO',
                          F.when(F.col('INCOME') > 0, F.col('DEBT') / F.col('INCOME')).otherwise(0))

    if 'INCOME' in df.columns and 'SAVINGS' in df.columns:
        df = df.withColumn('SAVINGS_TO_INCOME_RATIO',
                          F.when(F.col('INCOME') > 0, F.col('SAVINGS') / F.col('INCOME')).otherwise(0))

    # Replace infinity and NaN values with 0
    for col_name in df.columns:
        df = df.withColumn(col_name,
                          F.when(F.col(col_name).isNull() | F.isnan(col_name), 0).otherwise(F.col(col_name)))

    print(f"Schema after transformation:")
    df.printSchema()
    print(f"Final row count: {df.count()}")
    print(f"Number of features: {len(df.columns)}")

    # === Save transformed data ===
    target_path = f"s3://{args['TARGET_BUCKET']}/xgboost/"
    print(f"Writing transformed data to: {target_path}")

    # Write as CSV (XGBoost can read CSV)
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(target_path)

    print("✓ XGBoost data transformation completed successfully")

    job.commit()

except Exception as e:
    print(f"ERROR: Transformation failed - {str(e)}")
    import traceback
    traceback.print_exc()
    raise
