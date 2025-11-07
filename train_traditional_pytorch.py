"""
PyTorch Deep Learning Model for Credit Score Prediction
Adapted from Kaggle code for AWS SageMaker deployment
"""
import boto3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os

# Initialize S3 client
s3 = boto3.client('s3')

# Config
from config import get_config

cfg = get_config("traditional")
S3_BUCKET = cfg["S3_BUCKET"]
S3_PREFIX = cfg["S3_PREFIX"]
MODEL_OUTPUT = cfg["MODEL_OUTPUT"]
OUTPUT_BUCKET = cfg["OUTPUT_BUCKET"]

# Target column
TARGET = 'CreditScore'

# Hyperparameters
INPUT_SIZE = None  # Will be set after loading data
DROPOUT = 0.4
INIT_DROPOUT = 0.2
OUTPUT_SIZE = 1
LEARNING_RATE = 0.0001
BATCH_SIZE = 18
NUM_EPOCHS = 4000
VALID_SPLIT = 0.1

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def augment_data(df, target_col='CreditScore', augmentation_ratio=0.3):
    """
    Làm giàu dữ liệu bằng cách tạo thêm samples dựa trên data có sẵn
    Sử dụng các kỹ thuật:
    1. Interpolation: Nội suy giữa các samples có credit score tương tự
    2. Noise injection: Thêm nhiễu nhỏ vào numerical features
    """
    print("\n=== Data Augmentation ===")
    print(f"Original data shape: {df.shape}")

    df_original = df.copy()
    augmented_samples = []

    # Tách numeric và categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    n_augment = int(len(df) * augmentation_ratio)
    print(f"Generating {n_augment} augmented samples...")

    # Technique 1: Interpolation between similar samples (50%)
    for _ in range(n_augment // 2):
        # Random chọn 2 samples có score gần nhau
        idx1 = np.random.randint(0, len(df))

        # Tìm sample có score gần với idx1
        score_diff = np.abs(df[target_col] - df[target_col].iloc[idx1])
        similar_indices = score_diff.nsmallest(20).index.tolist()
        similar_indices.remove(idx1) if idx1 in similar_indices else None

        if len(similar_indices) > 0:
            idx2 = np.random.choice(similar_indices)

            # Tạo sample mới bằng cách nội suy (trọng số random)
            alpha = np.random.uniform(0.3, 0.7)
            new_sample = df.iloc[idx1].copy()

            # Nội suy numeric features
            for col in numeric_cols:
                new_sample[col] = alpha * df.iloc[idx1][col] + (1 - alpha) * df.iloc[idx2][col]

            # Nội suy target
            new_sample[target_col] = alpha * df.iloc[idx1][target_col] + (1 - alpha) * df.iloc[idx2][target_col]

            # Categorical: chọn random từ một trong hai
            for col in categorical_cols:
                new_sample[col] = df.iloc[idx1][col] if np.random.random() > 0.5 else df.iloc[idx2][col]

            augmented_samples.append(new_sample)

    print(f"✓ Created {len(augmented_samples)} interpolated samples")

    # Technique 2: Noise injection (50%)
    for _ in range(n_augment - len(augmented_samples)):
        idx = np.random.randint(0, len(df))
        new_sample = df.iloc[idx].copy()

        # Thêm Gaussian noise vào numeric features (±5% của std)
        for col in numeric_cols:
            if df[col].std() > 0:
                noise_scale = df[col].std() * 0.05
                noise = np.random.normal(0, noise_scale)
                new_sample[col] = max(0, new_sample[col] + noise)

        # Thêm noise nhỏ vào credit score (±2 điểm)
        noise = np.random.normal(0, 2)
        new_sample[target_col] = np.clip(
            new_sample[target_col] + noise,
            df[target_col].min(),
            df[target_col].max()
        )

        # Categorical giữ nguyên hoặc random swap với probability thấp
        for col in categorical_cols:
            if np.random.random() < 0.1:  # 10% chance thay đổi
                new_sample[col] = np.random.choice(df[col].unique())

        augmented_samples.append(new_sample)

    print(f"✓ Created {n_augment - len(augmented_samples) // 2} noise-injected samples")

    # Combine original + augmented
    df_augmented = pd.concat([df_original, pd.DataFrame(augmented_samples)], ignore_index=True)

    # Shuffle
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n✓ Data augmentation completed!")
    print(f"  Original samples: {len(df_original)}")
    print(f"  Augmented samples: {len(augmented_samples)}")
    print(f"  Total samples: {len(df_augmented)}")
    print(f"  Increase: {len(augmented_samples)/len(df_original)*100:.1f}%")

    return df_augmented


class CreditDataset(Dataset):
    """PyTorch Dataset for Credit Score data"""
    def __init__(self, features, labels, device=None):
        self.device = device if device else torch.device('cpu')
        self.features = torch.tensor(features, dtype=torch.float32).to(self.device)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(self.device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CreditScoreRegressor(nn.Module):
    """Deep Neural Network for Credit Score Regression"""
    def __init__(self, input_size, output_size=1, dropout=0.4):
        super(CreditScoreRegressor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)


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

    return X, y


def scale_target(y_train, y_valid):
    """Scale target variable to [0, 1] range"""
    print("\n=== Scaling Target Variable ===")

    # FICO score range: 300-850
    manual_min = np.array([300])
    manual_max = np.array([850])

    combined_data = np.vstack((
        y_train.reshape(-1, 1),
        manual_min,
        manual_max
    ))

    scaler = MinMaxScaler()
    scaler.fit(combined_data)

    y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_valid_scaled = scaler.transform(y_valid.reshape(-1, 1)).flatten()

    print(f"✓ Scaler range: [{scaler.data_min_[0]:.1f}, {scaler.data_max_[0]:.1f}]")
    print(f"✓ Train target scaled: [{y_train_scaled.min():.3f}, {y_train_scaled.max():.3f}]")
    print(f"✓ Valid target scaled: [{y_valid_scaled.min():.3f}, {y_valid_scaled.max():.3f}]")

    return y_train_scaled, y_valid_scaled, scaler


def train_model(model, train_loader, valid_loader, num_epochs, learning_rate, init_dropout_rate, device):
    """Train the PyTorch model"""
    print("\n=== Starting Model Training ===")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Move model to device
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    init_dropout = nn.Dropout(init_dropout_rate)

    best_train_loss = np.inf
    best_valid_loss = np.inf
    train_loss_history = []
    valid_loss_history = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        for features, labels in train_loader:
            # Data is already on device from CreditDataset
            features = init_dropout(features)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Update best train model
        if avg_train_loss <= best_train_loss:
            best_train_loss = avg_train_loss

        # Validation phase
        model.eval()
        total_valid_loss = 0

        with torch.no_grad():
            for features, labels in valid_loader:
                # Data is already on device from CreditDataset
                outputs = model(features)
                vloss = criterion(outputs, labels)
                total_valid_loss += vloss.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_loss_history.append(avg_valid_loss)

        # Update best valid model
        if avg_valid_loss <= best_valid_loss:
            best_valid_loss = avg_valid_loss

        # Print progress
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {avg_train_loss:.6f} (best: {best_train_loss:.6f})')
            print(f'  Valid Loss: {avg_valid_loss:.6f} (best: {best_valid_loss:.6f})')
            if device.type == 'cuda':
                print(f'  GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB')

    print("\n✓ Training Complete")
    print(f"  Best Train Loss: {best_train_loss:.6f}")
    print(f"  Best Valid Loss: {best_valid_loss:.6f}")

    return {
        'train_loss_history': train_loss_history,
        'valid_loss_history': valid_loss_history,
        'best_train_loss': best_train_loss,
        'best_valid_loss': best_valid_loss
    }


def main():
    try:
        print("="*60)
        print("PyTorch Deep Learning Credit Score Model Training")
        print("="*60)
        print(f"\n🔧 Device Configuration:")
        print(f"   Device: {DEVICE}")
        if DEVICE.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Available: {torch.cuda.is_available()}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print(f"   GPU: Not available, using CPU")
        print("="*60)

        # Load data from S3
        csv_files = get_all_csv_files(S3_BUCKET, S3_PREFIX)
        df = load_all_data_from_s3(S3_BUCKET, csv_files)

        # Data augmentation (increase dataset size by 30%)
        print("\n=== Data Augmentation ===")
        n_samples_original = len(df)
        df = augment_data(df, target_col=TARGET, augmentation_ratio=0.3)
        n_samples_augmented = len(df)
        print(f"✓ Original samples: {n_samples_original:,}")
        print(f"✓ Augmented samples: {n_samples_augmented:,}")
        print(f"✓ Increase: {n_samples_augmented - n_samples_original:,} samples ({((n_samples_augmented/n_samples_original - 1)*100):.1f}%)")

        # Prepare data
        X, y = prepare_data(df, TARGET)

        # Train/validation split
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=VALID_SPLIT, random_state=42
        )

        print(f"\n✓ Train set: {X_train.shape}")
        print(f"✓ Valid set: {X_valid.shape}")

        # Scale target variable
        y_train_scaled, y_valid_scaled, scaler = scale_target(y_train, y_valid)

        # Create datasets and dataloaders
        train_dataset = CreditDataset(X_train, y_train_scaled, device=DEVICE)
        valid_dataset = CreditDataset(X_valid, y_valid_scaled, device=DEVICE)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=len(X_valid), shuffle=False)

        print(f"\n✓ Train batches: {len(train_loader)}")
        print(f"✓ Valid batches: {len(valid_loader)}")

        # Initialize model
        input_size = X_train.shape[1]
        model = CreditScoreRegressor(
            input_size=input_size,
            output_size=OUTPUT_SIZE,
            dropout=DROPOUT
        )

        print(f"\n✓ Model initialized with {input_size} input features")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")

        # Train model
        training_stats = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            init_dropout_rate=INIT_DROPOUT,
            device=DEVICE
        )

        # Move model back to CPU for saving
        model = model.cpu()

        # Save model and artifacts
        print("\n=== Saving Model ===")

        # Save PyTorch model state
        model_state_path = '/tmp/model_state.pt'
        torch.save(model.state_dict(), model_state_path)

        # Save complete model package
        model_package = {
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'input_size': input_size,
            'model_config': {
                'dropout': DROPOUT,
                'output_size': OUTPUT_SIZE
            },
            'training_stats': training_stats
        }

        model_path = '/tmp/model.pt'
        torch.save(model_package, model_path)

        # Upload to S3
        model_key = MODEL_OUTPUT.replace('.joblib', '.pt')
        s3.upload_file(model_path, OUTPUT_BUCKET, model_key)
        print(f'✓ Model saved to s3://{OUTPUT_BUCKET}/{model_key}')

        # Save scaler separately for inference
        scaler_path = '/tmp/scaler.joblib'
        joblib.dump(scaler, scaler_path)
        scaler_key = MODEL_OUTPUT.replace('model.joblib', 'scaler.joblib')
        s3.upload_file(scaler_path, OUTPUT_BUCKET, scaler_key)
        print(f'✓ Scaler saved to s3://{OUTPUT_BUCKET}/{scaler_key}')

        # Save metrics
        metrics = {
            'best_train_loss': float(training_stats['best_train_loss']),
            'best_valid_loss': float(training_stats['best_valid_loss']),
            'final_train_loss': float(training_stats['train_loss_history'][-1]),
            'final_valid_loss': float(training_stats['valid_loss_history'][-1]),
            'n_samples_train': len(X_train),
            'n_samples_valid': len(X_valid),
            'n_samples_total': len(X),
            'n_samples_original': n_samples_original,
            'n_samples_augmented': n_samples_augmented,
            'augmentation_ratio': 0.3,
            'n_features': input_size,
            'n_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'total_parameters': total_params,
            'device': str(DEVICE),
            'cuda_available': torch.cuda.is_available()
        }

        metrics_path = '/tmp/metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        metrics_key = MODEL_OUTPUT.replace('model.joblib', 'metrics.json')
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
