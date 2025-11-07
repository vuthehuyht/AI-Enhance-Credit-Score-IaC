import boto3
import pandas as pd
import joblib
import os
from io import StringIO
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import numpy as np

# Initialize S3 client
s3 = boto3.client('s3')

# Config
from config import get_config

cfg = get_config("social")
S3_BUCKET = cfg["S3_BUCKET"]
S3_PREFIX = cfg["S3_PREFIX"]
MODEL_OUTPUT = cfg["MODEL_OUTPUT"]
OUTPUT_BUCKET = cfg["OUTPUT_BUCKET"]

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
    try:
        print(f"Starting Social Credit Score Model Training...")
        print(f"S3 Bucket: {S3_BUCKET}")
        print(f"S3 Prefix: {S3_PREFIX}")

        # Load data
        csv_key = get_latest_csv(S3_BUCKET, S3_PREFIX)
        print(f'Loading data from: s3://{S3_BUCKET}/{csv_key}')
        df = load_data_from_s3(S3_BUCKET, csv_key)

        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Preprocessing
        df = df.dropna()

        # Assume last column is target (CreditScore)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target range: {y.min():.2f} - {y.max():.2f}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\nTrain set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # Robust scaling (good for social data with outliers)
        print("\nScaling features with RobustScaler...")
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        print("\nTraining ExtraTreesRegressor...")
        model = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        model.fit(X_train_scaled, y_train)

        # Evaluate
        print("\n--- Model Evaluation ---")
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        # Training metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)

        # Test metrics
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)

        print(f"\nTraining Metrics:")
        print(f"  MAE:  {train_mae:.2f}")
        print(f"  RMSE: {train_rmse:.2f}")
        print(f"  R²:   {train_r2:.4f}")

        print(f"\nTest Metrics:")
        print(f"  MAE:  {test_mae:.2f}")
        print(f"  RMSE: {test_rmse:.2f}")
        print(f"  R²:   {test_r2:.4f}")

        # Feature importance
        if hasattr(X, 'columns'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\nTop 10 Feature Importance:")
            print(feature_importance.head(10))

        # Save model and scaler as pipeline
        print("\nSaving model and scaler...")
        pipeline = {
            'model': model,
            'scaler': scaler,
            'feature_names': X.columns.tolist() if hasattr(X, 'columns') else None
        }

        local_path = '/tmp/model.joblib'
        joblib.dump(pipeline, local_path)

        # Upload to S3
        model_key = MODEL_OUTPUT
        s3.upload_file(local_path, OUTPUT_BUCKET, model_key)
        print(f'\n✓ Model saved to s3://{OUTPUT_BUCKET}/{model_key}')

        # Save metrics
        metrics = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'n_samples': len(df),
            'n_features': X.shape[1]
        }

        metrics_path = '/tmp/metrics.json'
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        metrics_key = MODEL_OUTPUT.replace('model.joblib', 'metrics.json')
        s3.upload_file(metrics_path, OUTPUT_BUCKET, metrics_key)
        print(f'✓ Metrics saved to s3://{OUTPUT_BUCKET}/{metrics_key}')

        print("\n=== Training completed successfully ===")

    except Exception as e:
        print(f"ERROR: Training failed - {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
