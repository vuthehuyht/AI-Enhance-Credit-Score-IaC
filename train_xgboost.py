"""
XGBoost Model for Credit Score Prediction
Compatible with SageMaker training infrastructure
Uses same cleaned data as traditional model
"""
import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import json
import os

# Initialize S3 client
s3 = boto3.client('s3')

# Config - Use traditional data source
from config import get_config

cfg = get_config("traditional")  # XGBoost uses traditional cleaned data
S3_BUCKET = cfg["S3_BUCKET"]
S3_PREFIX = cfg["S3_PREFIX"]
OUTPUT_BUCKET = cfg["OUTPUT_BUCKET"]

# XGBoost specific output path
MODEL_OUTPUT = "xgboost"

# Target column
TARGET = 'CreditScore'

# Hyperparameters
VALID_SPLIT = 0.2
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'eval_metric': 'rmse',
    'seed': 42
}
NUM_ROUNDS = 500
EARLY_STOPPING_ROUNDS = 50


def get_all_csv_files(bucket, prefix):
    """Lấy tất cả file CSV trong S3 bucket/prefix"""
    print(f"\nScanning S3 for CSV files: s3://{bucket}/{prefix}")

    csv_files = []
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.csv'):
                    csv_files.append(obj['Key'])

    if not csv_files:
        raise Exception(f'No CSV files found in s3://{bucket}/{prefix}')

    print(f"Found {len(csv_files)} CSV files")
    return csv_files


def load_all_data_from_s3(bucket, csv_files):
    """Load và combine tất cả CSV files từ S3"""
    print(f"\nLoading {len(csv_files)} CSV files from S3...")

    dataframes = []
    for i, csv_key in enumerate(csv_files, 1):
        try:
            print(f"  [{i}/{len(csv_files)}] Loading {csv_key}...")
            obj = s3.get_object(Bucket=bucket, Key=csv_key)
            df = pd.read_csv(obj['Body'])
            dataframes.append(df)
            print(f"    ✓ Loaded {len(df)} rows")
        except Exception as e:
            print(f"    ✗ Error loading {csv_key}: {str(e)}")
            continue

    if not dataframes:
        raise Exception("Failed to load any CSV files")

    df_combined = pd.concat(dataframes, ignore_index=True)
    print(f"✓ Total rows loaded: {len(df_combined)}")

    # Remove duplicates
    original_len = len(df_combined)
    df_combined = df_combined.drop_duplicates()
    if len(df_combined) < original_len:
        print(f"✓ Removed {original_len - len(df_combined)} duplicate rows")

    return df_combined


def prepare_data(df, target_col):
    """Prepare and preprocess data"""
    print("\n=== Data Preprocessing ===")

    # Remove first column if it's index
    if df.columns[0].lower() in ['unnamed: 0', 'index', 'id']:
        df = df.iloc[:, 1:]
        print("✓ Removed index column")

    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    if categorical_cols:
        print(f"Encoding {len(categorical_cols)} categorical columns...")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"✓ Features after encoding: {len(df.columns)}")

    # Separate features and target
    if target_col not in df.columns:
        raise Exception(f"Target column '{target_col}' not found in data")

    y = df[target_col].values
    X = df.drop(columns=[target_col]).values

    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Target range: [{y.min():.1f}, {y.max():.1f}]")

    # Get feature names for later use
    feature_names = df.drop(columns=[target_col]).columns.tolist()

    return X, y, feature_names


def train_xgboost_model(X_train, y_train, X_valid, y_valid, params, num_rounds, early_stopping):
    """Train XGBoost model with early stopping"""
    print("\n=== Starting XGBoost Training ===")
    print(f"Parameters: {params}")
    print(f"Number of boosting rounds: {num_rounds}")
    print(f"Early stopping rounds: {early_stopping}")

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # Evaluation list
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    evals_result = {}

    # Train model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        early_stopping_rounds=early_stopping,
        evals_result=evals_result,
        verbose_eval=50
    )

    print(f"\n✓ Training Complete")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best train RMSE: {evals_result['train']['rmse'][model.best_iteration]:.4f}")
    print(f"  Best valid RMSE: {evals_result['valid']['rmse'][model.best_iteration]:.4f}")

    return model, evals_result


def main():
    try:
        print("="*60)
        print("XGBoost Credit Score Model Training")
        print("="*60)

        # Load data from S3
        csv_files = get_all_csv_files(S3_BUCKET, S3_PREFIX)
        df = load_all_data_from_s3(S3_BUCKET, csv_files)

        # Prepare data
        X, y, feature_names = prepare_data(df, TARGET)

        # Train/validation split
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=VALID_SPLIT, random_state=42
        )

        print(f"\n✓ Train set: {X_train.shape}")
        print(f"✓ Valid set: {X_valid.shape}")

        # Train XGBoost model
        model, evals_result = train_xgboost_model(
            X_train, y_train,
            X_valid, y_valid,
            XGBOOST_PARAMS,
            NUM_ROUNDS,
            EARLY_STOPPING_ROUNDS
        )

        # Save model and artifacts
        print("\n=== Saving Model ===")

        # Save XGBoost model
        model_path = '/tmp/xgboost_model.json'
        model.save_model(model_path)

        # Upload to S3 - xgboost/model.json
        model_key = f'{MODEL_OUTPUT}/model.json'
        s3.upload_file(model_path, OUTPUT_BUCKET, model_key)
        print(f'✓ Model saved to s3://{OUTPUT_BUCKET}/{model_key}')

        # Save identity scaler (XGBoost doesn't need scaling, but Flask expects it)
        # Create a dummy scaler that returns identity transformation
        scaler = StandardScaler()
        scaler.mean_ = np.array([0.0])
        scaler.scale_ = np.array([1.0])
        scaler.var_ = np.array([1.0])
        scaler.n_features_in_ = 1
        scaler.n_samples_seen_ = len(y)

        scaler_path = '/tmp/scaler.joblib'
        joblib.dump(scaler, scaler_path)
        scaler_key = f'{MODEL_OUTPUT}/scaler.joblib'
        s3.upload_file(scaler_path, OUTPUT_BUCKET, scaler_key)
        print(f'✓ Scaler saved to s3://{OUTPUT_BUCKET}/{scaler_key}')

        # Save config with input size
        config = {
            'input_size': len(feature_names),
            'n_features': len(feature_names),
            'feature_names': feature_names
        }
        config_path = '/tmp/config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        config_key = f'{MODEL_OUTPUT}/config.json'
        s3.upload_file(config_path, OUTPUT_BUCKET, config_key)
        print(f'✓ Config saved to s3://{OUTPUT_BUCKET}/{config_key}')

        # Get feature importance
        importance_dict = model.get_score(importance_type='gain')
        feature_importance = {
            f'f{i}': {'feature': feature_names[i], 'importance': importance_dict.get(f'f{i}', 0)}
            for i in range(len(feature_names))
        }

        # Save metrics
        metrics = {
            'best_iteration': int(model.best_iteration),
            'train_rmse': float(evals_result['train']['rmse'][model.best_iteration]),
            'valid_rmse': float(evals_result['valid']['rmse'][model.best_iteration]),
            'final_train_rmse': float(evals_result['train']['rmse'][-1]),
            'final_valid_rmse': float(evals_result['valid']['rmse'][-1]),
            'n_samples_train': len(X_train),
            'n_samples_valid': len(X_valid),
            'n_samples_total': len(X),
            'n_features': len(feature_names),
            'num_boost_rounds': NUM_ROUNDS,
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
            'params': XGBOOST_PARAMS,
            'top_10_features': sorted(
                feature_importance.items(),
                key=lambda x: x[1]['importance'],
                reverse=True
            )[:10]
        }

        metrics_path = '/tmp/metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        metrics_key = f'{MODEL_OUTPUT}/metrics.json'
        s3.upload_file(metrics_path, OUTPUT_BUCKET, metrics_key)
        print(f'✓ Metrics saved to s3://{OUTPUT_BUCKET}/{metrics_key}')

        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nERROR: Training failed - {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
