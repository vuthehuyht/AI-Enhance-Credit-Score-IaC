import os

def get_config(model_type):
    # These values should be replaced by Terraform at deployment time
    default_bucket = "${CLEANED_BUCKET}"
    default_output_bucket = "${ML_DATA_BUCKET}"
    s3_prefix = f"cleaned/{model_type}/"
    model_output = f"models/{model_type}/model.joblib"
    return {
        "S3_BUCKET": os.environ.get("S3_BUCKET", default_bucket),
        "S3_PREFIX": os.environ.get("S3_PREFIX", s3_prefix),
        "MODEL_OUTPUT": os.environ.get("MODEL_OUTPUT", model_output),
        "OUTPUT_BUCKET": os.environ.get("OUTPUT_BUCKET", default_output_bucket),
    }
