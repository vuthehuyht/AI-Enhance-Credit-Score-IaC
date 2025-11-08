"""
Flask API for Credit Score Prediction
- Existing customers: Use traditional + social models with aggregation
- New customers: Use XGBoost model
"""
from flask import Flask, request, jsonify
import boto3
import torch
import torch.nn as nn
import joblib
import xgboost as xgb
import numpy as np
import logging
import os
import json
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS clients
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Configuration from environment variables
ML_DATA_BUCKET = os.environ.get('ML_DATA_BUCKET', 'vpbank-hackathon-dev-models')
CUSTOMER_TABLE_NAME = os.environ.get('CUSTOMER_TABLE_NAME', 'vpbank-customers')
MODEL_CACHE_DIR = '/tmp/models'

# Cache models in memory
MODELS_CACHE = {}

# DynamoDB table
try:
    customer_table = dynamodb.Table(CUSTOMER_TABLE_NAME)
    logger.info(f'Connected to DynamoDB table: {CUSTOMER_TABLE_NAME}')
except Exception as e:
    logger.warning(f'Failed to connect to DynamoDB: {e}')
    customer_table = None


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


def load_model_from_s3(model_name):
    """Load model and scaler from S3, with caching"""
    if model_name in MODELS_CACHE:
        logger.info(f'Using cached model for {model_name}')
        return MODELS_CACHE[model_name]

    try:
        logger.info(f'Loading model {model_name} from S3...')

        # Create cache directory
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        # Download scaler file (common for all models)
        scaler_key = f'{model_name}/scaler.joblib'
        scaler_path = f'{MODEL_CACHE_DIR}/{model_name}_scaler.joblib'
        s3.download_file(ML_DATA_BUCKET, scaler_key, scaler_path)
        scaler = joblib.load(scaler_path)

        # Load model based on type
        if model_name == 'xgboost':
            # Download XGBoost model
            model_key = f'{model_name}/model.json'
            model_path = f'{MODEL_CACHE_DIR}/{model_name}_model.json'
            s3.download_file(ML_DATA_BUCKET, model_key, model_path)

            # Load XGBoost model
            model = xgb.Booster()
            model.load_model(model_path)

            # Get input size from config
            config_key = f'{model_name}/config.json'
            config_path = f'{MODEL_CACHE_DIR}/{model_name}_config.json'
            try:
                s3.download_file(ML_DATA_BUCKET, config_key, config_path)
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                input_size = config.get('input_size', 100)  # Default for XGBoost
            except:
                input_size = 100  # Default

            MODELS_CACHE[model_name] = {
                'model': model,
                'scaler': scaler,
                'input_size': input_size,
                'model_type': 'xgboost'
            }

        elif model_name == 'social':
            # Download sklearn model
            model_key = f'{model_name}/model.pkl'
            model_path = f'{MODEL_CACHE_DIR}/{model_name}_model.pkl'
            s3.download_file(ML_DATA_BUCKET, model_key, model_path)

            # Load sklearn model
            model = joblib.load(model_path)

            # Get input size
            if hasattr(model, 'n_features_in_'):
                input_size = model.n_features_in_
            else:
                input_size = 50  # Default for social

            MODELS_CACHE[model_name] = {
                'model': model,
                'scaler': scaler,
                'input_size': input_size,
                'model_type': 'sklearn'
            }

        else:  # traditional PyTorch model
            # Download model file
            model_key = f'{model_name}/model.pt'
            model_path = f'{MODEL_CACHE_DIR}/{model_name}_model.pt'
            s3.download_file(ML_DATA_BUCKET, model_key, model_path)

            # Load model package
            model_package = torch.load(model_path, map_location=torch.device('cpu'))

            # Initialize model architecture
            model = CreditScoreRegressor(
                input_size=model_package['input_size'],
                output_size=model_package['model_config']['output_size'],
                dropout=model_package['model_config']['dropout']
            )

            # Load model weights
            model.load_state_dict(model_package['model_state_dict'])
            model.eval()

            # Cache for future requests
            MODELS_CACHE[model_name] = {
                'model': model,
                'scaler': scaler,
                'input_size': model_package['input_size'],
                'model_type': 'pytorch'
            }

        logger.info(f'✓ Model {model_name} loaded successfully')
        return MODELS_CACHE[model_name]

    except Exception as e:
        logger.exception(f'Failed to load model {model_name}: {str(e)}')
        raise


def check_customer_exists(customer_id):
    """Check if customer exists in database"""
    if not customer_table:
        logger.warning('DynamoDB table not available, treating as new customer')
        return False

    try:
        response = customer_table.get_item(Key={'customer_id': customer_id})
        exists = 'Item' in response
        logger.info(f'Customer {customer_id} exists: {exists}')
        return exists
    except Exception as e:
        logger.exception(f'Error checking customer {customer_id}: {e}')
        return False


def save_prediction_to_db(customer_id, prediction_data):
    """Save prediction result to database"""
    if not customer_table:
        logger.warning('DynamoDB table not available, skipping save')
        return

    try:
        item = {
            'customer_id': customer_id,
            'prediction_timestamp': datetime.utcnow().isoformat(),
            'credit_score': prediction_data.get('aggregated_score') or prediction_data.get('score'),
            'models_used': prediction_data.get('models_used', []),
            'prediction_type': prediction_data.get('prediction_type'),
            'raw_predictions': prediction_data.get('models', [])
        }

        customer_table.put_item(Item=item)
        logger.info(f'Saved prediction for customer {customer_id}')
    except Exception as e:
        logger.exception(f'Error saving prediction for {customer_id}: {e}')


def predict_score(model_name, input_features):
    """Perform inference for a given model"""
    try:
        # Load model and scaler
        model_data = load_model_from_s3(model_name)
        model = model_data['model']
        scaler = model_data['scaler']
        model_type = model_data['model_type']

        # Validate input features
        if not isinstance(input_features, list):
            raise ValueError("input_features must be a list of numeric values")

        if len(input_features) != model_data['input_size']:
            raise ValueError(
                f"Expected {model_data['input_size']} features, got {len(input_features)}"
            )

        # Convert to numpy array and reshape
        features_array = np.array(input_features).reshape(1, -1)

        # Predict based on model type
        if model_type == 'xgboost':
            # XGBoost prediction
            dmatrix = xgb.DMatrix(features_array)
            prediction_scaled = model.predict(dmatrix)[0]

        elif model_type == 'sklearn':
            # Sklearn prediction
            prediction_scaled = model.predict(features_array)[0]

        else:  # pytorch
            # PyTorch prediction
            with torch.no_grad():
                features_tensor = torch.tensor(features_array, dtype=torch.float32)
                prediction_scaled = model(features_tensor).item()

        # Inverse transform to get actual credit score
        prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]

        # Clip to valid range (300-850)
        prediction = np.clip(prediction, 300, 850)

        return {
            'score': float(prediction),
            'score_scaled': float(prediction_scaled),
            'model': model_name
        }

    except Exception as e:
        logger.exception(f'Prediction failed for {model_name}: {str(e)}')
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_cached': list(MODELS_CACHE.keys())
    })


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    Predict credit score with customer type routing
    - Existing customers: traditional + social models (aggregated)
    - New customers: XGBoost model only

    Request body:
    {
        "customer_id": "CUS123456",
        "features": {
            "traditional": [list of features for traditional model],
            "social": [list of features for social model],
            "xgboost": [list of features for xgboost model]
        }
    }

    Response for existing customer:
    {
        "customer_id": "CUS123456",
        "customer_type": "existing",
        "aggregated_score": 735.4,
        "models": [
            {"model": "traditional", "score": 750.5},
            {"model": "social", "score": 720.3}
        ],
        "prediction_type": "multi_model_aggregate"
    }

    Response for new customer:
    {
        "customer_id": "CUS999999",
        "customer_type": "new",
        "score": 680.2,
        "model": "xgboost",
        "prediction_type": "new_customer_xgboost"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Missing request body'}), 400

        # Extract customer_id and features
        customer_id = data.get('customer_id')
        features_data = data.get('features', {})

        if not customer_id:
            return jsonify({'error': 'Missing "customer_id" in request'}), 400

        if not features_data:
            return jsonify({'error': 'Missing "features" in request'}), 400

        # Check if customer exists in database
        is_existing_customer = check_customer_exists(customer_id)

        if is_existing_customer:
            # Existing customer: Use traditional + social models
            logger.info(f'Processing EXISTING customer: {customer_id}')

            results = []

            # Traditional model
            if 'traditional' in features_data:
                try:
                    trad_result = predict_score('traditional', features_data['traditional'])
                    results.append({
                        'model': 'traditional',
                        'score': trad_result['score'],
                        'score_scaled': trad_result['score_scaled']
                    })
                except Exception as e:
                    logger.exception('Traditional model inference failed')
                    results.append({'model': 'traditional', 'error': str(e)})

            # Social model
            if 'social' in features_data:
                try:
                    social_result = predict_score('social', features_data['social'])
                    results.append({
                        'model': 'social',
                        'score': social_result['score'],
                        'score_scaled': social_result['score_scaled']
                    })
                except Exception as e:
                    logger.exception('Social model inference failed')
                    results.append({'model': 'social', 'error': str(e)})

            # Aggregate scores
            scores = [
                r['score'] for r in results
                if 'score' in r and 'error' not in r
            ]

            if not scores:
                return jsonify({
                    'error': 'All model predictions failed',
                    'models': results
                }), 500

            agg_score = sum(scores) / len(scores)

            response = {
                'customer_id': customer_id,
                'customer_type': 'existing',
                'aggregated_score': float(agg_score),
                'models': results,
                'num_models_used': len(scores),
                'prediction_type': 'multi_model_aggregate',
                'timestamp': datetime.utcnow().isoformat()
            }

            # Save to database
            save_prediction_to_db(customer_id, {
                'aggregated_score': agg_score,
                'models': results,
                'models_used': [r['model'] for r in results if 'score' in r],
                'prediction_type': 'multi_model_aggregate'
            })

            return jsonify(response)

        else:
            # New customer: Use XGBoost only
            logger.info(f'Processing NEW customer: {customer_id}')

            if 'xgboost' not in features_data:
                return jsonify({
                    'error': 'Missing "xgboost" features for new customer prediction'
                }), 400

            try:
                xgb_result = predict_score('xgboost', features_data['xgboost'])

                response = {
                    'customer_id': customer_id,
                    'customer_type': 'new',
                    'score': xgb_result['score'],
                    'score_scaled': xgb_result['score_scaled'],
                    'model': 'xgboost',
                    'prediction_type': 'new_customer_xgboost',
                    'timestamp': datetime.utcnow().isoformat()
                }

                # Save to database
                save_prediction_to_db(customer_id, {
                    'score': xgb_result['score'],
                    'models_used': ['xgboost'],
                    'prediction_type': 'new_customer_xgboost'
                })

                return jsonify(response)

            except Exception as e:
                logger.exception('XGBoost model inference failed')
                return jsonify({
                    'error': f'XGBoost prediction failed: {str(e)}'
                }), 500

    except Exception as e:
        logger.exception('Prediction request failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/search', methods=['GET', 'POST'])
def search():
    """
    Search customer credit score history from database

    GET: /api/v1/search?customer_id=CUS123456
    POST: {"customer_id": "CUS123456"}

    Response:
    {
        "customer_id": "CUS123456",
        "found": true,
        "predictions": [
            {
                "timestamp": "2025-11-08T10:30:00Z",
                "credit_score": 735.4,
                "prediction_type": "multi_model_aggregate",
                "models_used": ["traditional", "social"]
            }
        ]
    }
    """
    try:
        if not customer_table:
            return jsonify({
                'error': 'Database not available'
            }), 503

        # Get customer_id from query param or body
        if request.method == 'GET':
            customer_id = request.args.get('customer_id')
        else:
            data = request.get_json()
            customer_id = data.get('customer_id') if data else None

        if not customer_id:
            return jsonify({'error': 'Missing "customer_id" parameter'}), 400

        # Query DynamoDB
        try:
            response = customer_table.get_item(Key={'customer_id': customer_id})

            if 'Item' not in response:
                return jsonify({
                    'customer_id': customer_id,
                    'found': False,
                    'message': 'Customer not found in database'
                })

            item = response['Item']

            return jsonify({
                'customer_id': customer_id,
                'found': True,
                'latest_prediction': {
                    'timestamp': item.get('prediction_timestamp'),
                    'credit_score': float(item.get('credit_score', 0)),
                    'prediction_type': item.get('prediction_type'),
                    'models_used': item.get('models_used', []),
                    'raw_predictions': item.get('raw_predictions', [])
                }
            })

        except Exception as e:
            logger.exception(f'Database query failed for {customer_id}')
            return jsonify({
                'error': f'Database query failed: {str(e)}'
            }), 500

    except Exception as e:
        logger.exception('Search request failed')
        return jsonify({'error': str(e)}), 500


@app.route('/models/reload', methods=['POST'])
def reload_models():
    """Reload models from S3 (clear cache)"""
    try:
        global MODELS_CACHE
        MODELS_CACHE = {}
        logger.info('Models cache cleared')
        return jsonify({'message': 'Models cache cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
