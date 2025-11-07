"""
PyTorch Inference Script for SageMaker
Handles loading and inference for PyTorch credit score model
"""
import json
import torch
import torch.nn as nn
import numpy as np
import joblib
import os


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


def model_fn(model_dir):
    """
    Load PyTorch model from model directory
    """
    model_path = os.path.join(model_dir, 'model.pt')

    # Load model package
    model_package = torch.load(model_path, map_location=torch.device('cpu'))

    # Extract components
    input_size = model_package['input_size']
    model_config = model_package['model_config']
    scaler = model_package['scaler']

    # Initialize model
    model = CreditScoreRegressor(
        input_size=input_size,
        output_size=model_config['output_size'],
        dropout=model_config['dropout']
    )

    # Load trained weights
    model.load_state_dict(model_package['model_state_dict'])
    model.eval()  # Set to evaluation mode

    return {
        'model': model,
        'scaler': scaler,
        'input_size': input_size
    }


def input_fn(request_body, request_content_type):
    """
    Parse input data payload
    """
    if request_content_type == 'application/json':
        data = json.loads(request_body)

        # Handle different input formats
        if 'features' in data:
            # Single prediction: {"features": [1, 2, 3, ...]}
            return np.array([data['features']], dtype=np.float32)
        elif 'data' in data:
            # Batch prediction: {"data": [[1, 2, 3], [4, 5, 6]]}
            return np.array(data['data'], dtype=np.float32)
        elif 'input' in data:
            # Alternative format: {"input": [...]}
            return np.array([data['input']], dtype=np.float32)
        else:
            # Assume the entire payload is the feature array
            return np.array([data], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_package):
    """
    Make predictions using loaded PyTorch model
    """
    model = model_package['model']
    scaler = model_package['scaler']

    # Convert numpy array to PyTorch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Make prediction (scaled output)
    with torch.no_grad():
        scaled_predictions = model(input_tensor)

    # Convert back to numpy
    scaled_predictions = scaled_predictions.numpy()

    # Inverse transform to get original credit score range (300-850)
    predictions = scaler.inverse_transform(scaled_predictions)

    return predictions.flatten()


def output_fn(predictions, accept):
    """
    Format prediction output
    """
    if accept == 'application/json':
        # Convert numpy array to list
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()

        # Format response
        if isinstance(predictions, list):
            if len(predictions) == 1:
                # Single prediction
                score = float(predictions[0])
                response = {
                    'score': score,
                    'credit_score': score,
                    'risk_level': get_risk_level(score),
                    'status': 'success',
                    'model_type': 'pytorch_deep_learning'
                }
            else:
                # Batch predictions
                response = {
                    'predictions': [
                        {
                            'score': float(p),
                            'credit_score': float(p),
                            'risk_level': get_risk_level(float(p))
                        }
                        for p in predictions
                    ],
                    'status': 'success',
                    'model_type': 'pytorch_deep_learning'
                }
        else:
            score = float(predictions)
            response = {
                'score': score,
                'credit_score': score,
                'risk_level': get_risk_level(score),
                'status': 'success',
                'model_type': 'pytorch_deep_learning'
            }

        return json.dumps(response)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


def get_risk_level(score):
    """
    Determine risk level based on FICO credit score
    300-579: Poor (High Risk)
    580-669: Fair (Medium Risk)
    670-739: Good (Low-Medium Risk)
    740-799: Very Good (Low Risk)
    800-850: Excellent (Very Low Risk)
    """
    if score >= 800:
        return "excellent"
    elif score >= 740:
        return "very_good"
    elif score >= 670:
        return "good"
    elif score >= 580:
        return "fair"
    else:
        return "poor"
