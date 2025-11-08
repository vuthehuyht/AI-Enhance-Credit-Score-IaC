"""
Flask API for Credit Score Prediction
- New customers: Use XGBoost model with form data
- Existing customers: Use traditional + social models with aggregation
- Search: Customer lookup by identifiers
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


# FICO Score Groups mapping
def get_fico_group(score):
    """Map credit score to FICO group"""
    if score >= 800:
        return "Exceptional"
    elif score >= 740:
        return "Very Good"
    elif score >= 670:
        return "Good"
    elif score >= 580:
        return "Fair"
    else:
        return "Poor"


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
            # Download social model (sklearn-based)
            model_key = f'{model_name}/model.pkl'
            model_path = f'{MODEL_CACHE_DIR}/{model_name}_model.pkl'
            s3.download_file(ML_DATA_BUCKET, model_key, model_path)

            # Load social model
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
                'model_type': 'social'
            }

        else:  # traditional model (PyTorch-based)
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
                'model_type': 'traditional'
            }

        logger.info(f'✓ Model {model_name} loaded successfully')
        return MODELS_CACHE[model_name]

    except Exception as e:
        logger.exception(f'Failed to load model {model_name}: {str(e)}')
        raise


def check_customer_exists(national_id=None, email=None, phone_number=None):
    """Check if customer exists in database by national_id, email or phone"""
    if not customer_table:
        logger.warning('DynamoDB table not available, treating as new customer')
        return False, None

    try:
        # Primary lookup by national_id
        if national_id:
            response = customer_table.get_item(Key={'national_id': national_id})
            if 'Item' in response:
                logger.info(f'Customer found by national_id: {national_id}')
                return True, response['Item']

        # Fallback: scan by email or phone (less efficient, consider GSI in production)
        if email or phone_number:
            scan_filter = {}
            if email:
                scan_filter['email'] = {'AttributeValueList': [email], 'ComparisonOperator': 'EQ'}
            if phone_number:
                scan_filter['phone_number'] = {'AttributeValueList': [phone_number], 'ComparisonOperator': 'EQ'}

            response = customer_table.scan(ScanFilter=scan_filter, Limit=1)
            if response.get('Items'):
                logger.info(f'Customer found by email/phone')
                return True, response['Items'][0]

        return False, None
    except Exception as e:
        logger.exception(f'Error checking customer: {e}')
        return False, None


def prepare_xgboost_features(form_data):
    """Prepare features for XGBoost model from form data"""
    # Extract numerical features
    income = float(form_data.get('income', 0))
    savings = float(form_data.get('savings', 0))
    debt = float(form_data.get('debt', 0))

    # Extract categorical features (Yes/No -> 1/0)
    dependents = 1 if form_data.get('cat_dependents', 'No').lower() == 'yes' else 0
    mortgage = 1 if form_data.get('cat_mortgage', 'No').lower() == 'yes' else 0
    savings_account = 1 if form_data.get('cat_savings_account', 'No').lower() == 'yes' else 0
    credit_card = 1 if form_data.get('cat_credit_card', 'No').lower() == 'yes' else 0

    # Create feature array (adjust order based on your training data)
    features = [
        income,
        savings,
        debt,
        dependents,
        mortgage,
        savings_account,
        credit_card,
        # Derived features
        savings / income if income > 0 else 0,  # Savings ratio
        debt / income if income > 0 else 0,     # Debt-to-income ratio
        income - debt                            # Net worth
    ]

    return features


def save_customer_to_db(form_data, prediction_result):
    """Save new customer and prediction to database"""
    if not customer_table:
        logger.warning('DynamoDB table not available, skipping save')
        return

    try:
        item = {
            'national_id': form_data['national_id'],
            'full_name': form_data.get('full_name', ''),
            'email': form_data.get('email', ''),
            'phone_number': form_data.get('phone_number', ''),
            'created_at': datetime.utcnow().isoformat(),
            'last_prediction': datetime.utcnow().isoformat(),
            'credit_score': prediction_result['predicted_score'],
            'credit_group': prediction_result['predicted_group'],
            'prediction_history': [prediction_result]
        }

        customer_table.put_item(Item=item)
        logger.info(f'Saved new customer: {form_data["national_id"]}')
    except Exception as e:
        logger.exception(f'Error saving customer: {e}')


def update_customer_prediction(customer_data, prediction_result):
    """Update existing customer with new prediction"""
    if not customer_table:
        return

    try:
        # Update customer record with new prediction
        customer_table.update_item(
            Key={'national_id': customer_data['national_id']},
            UpdateExpression='SET last_prediction = :ts, credit_score = :score, credit_group = :group',
            ExpressionAttributeValues={
                ':ts': datetime.utcnow().isoformat(),
                ':score': prediction_result['predicted_score'],
                ':group': prediction_result['predicted_group']
            }
        )
        logger.info(f'Updated customer: {customer_data["national_id"]}')
    except Exception as e:
        logger.exception(f'Error updating customer: {e}')


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

        elif model_type == 'social':
            # Social model (sklearn-based) prediction
            prediction_scaled = model.predict(features_array)[0]

        elif model_type == 'traditional':
            # Traditional model (PyTorch-based) prediction
            with torch.no_grad():
                features_tensor = torch.tensor(features_array, dtype=torch.float32)
                prediction_scaled = model(features_tensor).item()

        else:
            raise ValueError(f"Unknown model type: {model_type}")

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


@app.route('/api/v1/predict-new-customer', methods=['POST'])
def predict_new_customer():
    """
    Predict credit score for NEW customer using XGBoost model

    Request body:
    {
        "full_name": "Nguyen Van A",
        "national_id": "001234567890",
        "email": "nguyenvana@example.com",
        "phone_number": "0912345678",
        "income": 50000000,
        "savings": 10000000,
        "debt": 5000000,
        "cat_dependents": "Yes",
        "cat_mortgage": "No",
        "cat_savings_account": "Yes",
        "cat_credit_card": "Yes"
    }

    Response:
    {
        "predicted_score": 720.5,
        "predicted_group": "Good",
        "customer_info": {
            "full_name": "Nguyen Van A",
            "national_id": "001234567890",
            "email": "nguyenvana@example.com",
            "phone_number": "0912345678"
        },
        "timestamp": "2025-11-08T10:30:00Z"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Missing request body'}), 400

        # Validate required fields
        required_fields = ['full_name', 'national_id', 'email', 'phone_number',
                          'income', 'savings', 'debt']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        # Check if customer already exists
        exists, existing_customer = check_customer_exists(
            national_id=data['national_id'],
            email=data['email'],
            phone_number=data['phone_number']
        )

        if exists:
            return jsonify({
                'error': 'Customer already exists. Please use the existing customer prediction endpoint.',
                'existing_customer': {
                    'national_id': existing_customer.get('national_id'),
                    'full_name': existing_customer.get('full_name')
                }
            }), 400

        # Prepare features for XGBoost
        features = prepare_xgboost_features(data)

        # Run XGBoost prediction
        xgb_result = predict_score('xgboost', features)
        predicted_score = xgb_result['score']
        predicted_group = get_fico_group(predicted_score)

        # Prepare response
        result = {
            'predicted_score': predicted_score,
            'predicted_group': predicted_group,
            'customer_info': {
                'full_name': data['full_name'],
                'national_id': data['national_id'],
                'email': data['email'],
                'phone_number': data['phone_number']
            },
            'timestamp': datetime.utcnow().isoformat()
        }

        # Save customer to database
        save_customer_to_db(data, result)

        return jsonify(result)

    except Exception as e:
        logger.exception('New customer prediction failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/predict-existing-customer', methods=['POST'])
def predict_existing_customer():
    """
    Predict credit score for EXISTING customer using traditional + social models

    Request body:
    {
        "full_name": "Nguyen Van A",
        "national_id": "001234567890",
        "email": "nguyenvana@example.com",
        "phone_number": "0912345678"
    }

    Response:
    {
        "predicted_score": 735.4,
        "predicted_group": "Very Good",
        "customer_info": {
            "full_name": "Nguyen Van A",
            "national_id": "001234567890",
            "email": "nguyenvana@example.com",
            "phone_number": "0912345678"
        },
        "models_used": ["traditional", "social"],
        "model_scores": {
            "traditional": 750.5,
            "social": 720.3
        },
        "timestamp": "2025-11-08T10:30:00Z"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Missing request body'}), 400

        # Validate at least one identifier
        if not any([data.get('national_id'), data.get('email'), data.get('phone_number')]):
            return jsonify({'error': 'At least one identifier required: national_id, email, or phone_number'}), 400

        # Check if customer exists
        exists, customer_data = check_customer_exists(
            national_id=data.get('national_id'),
            email=data.get('email'),
            phone_number=data.get('phone_number')
        )

        if not exists:
            return jsonify({
                'error': 'Customer not found. Please use the new customer prediction endpoint.',
                'suggestion': 'POST /api/v1/predict-new-customer'
            }), 404

        logger.info(f'Processing EXISTING customer: {customer_data.get("national_id")}')

        # TODO: Get actual features from customer data or additional input
        # For now, use placeholder - in production, extract from customer history
        # You may need to pass features in request or retrieve from database

        results = []

        # Traditional model prediction
        # Note: You need to determine how to get features for existing customers
        # Option 1: Pass features in request
        # Option 2: Retrieve from customer history in database
        # Option 3: Use stored feature vectors

        if 'features' in data:
            # If features provided in request
            if 'traditional' in data['features']:
                try:
                    trad_result = predict_score('traditional', data['features']['traditional'])
                    results.append({
                        'model': 'traditional',
                        'score': trad_result['score']
                    })
                except Exception as e:
                    logger.exception('Traditional model failed')

            if 'social' in data['features']:
                try:
                    social_result = predict_score('social', data['features']['social'])
                    results.append({
                        'model': 'social',
                        'score': social_result['score']
                    })
                except Exception as e:
                    logger.exception('Social model failed')
        else:
            # If no features provided, return last known score from database
            if 'credit_score' in customer_data:
                return jsonify({
                    'predicted_score': float(customer_data['credit_score']),
                    'predicted_group': customer_data.get('credit_group', get_fico_group(customer_data['credit_score'])),
                    'customer_info': {
                        'full_name': customer_data.get('full_name'),
                        'national_id': customer_data.get('national_id'),
                        'email': customer_data.get('email'),
                        'phone_number': customer_data.get('phone_number')
                    },
                    'last_prediction': customer_data.get('last_prediction'),
                    'source': 'database_cache'
                })
            else:
                return jsonify({'error': 'No prediction available. Features required for new prediction.'}), 400

        # Aggregate scores
        scores = [r['score'] for r in results if 'score' in r]

        if not scores:
            return jsonify({'error': 'No valid predictions from models'}), 500

        aggregated_score = sum(scores) / len(scores)
        predicted_group = get_fico_group(aggregated_score)

        # Prepare response
        result = {
            'predicted_score': aggregated_score,
            'predicted_group': predicted_group,
            'customer_info': {
                'full_name': customer_data.get('full_name') or data.get('full_name'),
                'national_id': customer_data.get('national_id'),
                'email': customer_data.get('email'),
                'phone_number': customer_data.get('phone_number')
            },
            'models_used': [r['model'] for r in results],
            'model_scores': {r['model']: r['score'] for r in results},
            'timestamp': datetime.utcnow().isoformat()
        }

        # Update customer record
        update_customer_prediction(customer_data, result)

        return jsonify(result)

    except Exception as e:
        logger.exception('Existing customer prediction failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/search', methods=['GET', 'POST'])
def search():
    """
    Search customer by identifiers

    GET: /api/v1/search?national_id=001234567890
    GET: /api/v1/search?email=test@example.com
    GET: /api/v1/search?phone_number=0912345678

    POST body:
    {
        "full_name": "Nguyen Van A",
        "national_id": "001234567890",
        "email": "nguyenvana@example.com",
        "phone_number": "0912345678"
    }

    Response:
    {
        "found": true,
        "customer": {
            "full_name": "Nguyen Van A",
            "national_id": "001234567890",
            "email": "nguyenvana@example.com",
            "phone_number": "0912345678",
            "credit_score": 735.4,
            "credit_group": "Very Good",
            "last_prediction": "2025-11-08T10:30:00Z",
            "created_at": "2025-11-01T08:00:00Z"
        }
    }
    """
    try:
        # Get search parameters
        if request.method == 'GET':
            national_id = request.args.get('national_id')
            email = request.args.get('email')
            phone_number = request.args.get('phone_number')
        else:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Missing request body'}), 400
            national_id = data.get('national_id')
            email = data.get('email')
            phone_number = data.get('phone_number')

        # Validate at least one identifier provided
        if not any([national_id, email, phone_number]):
            return jsonify({'error': 'At least one identifier required: national_id, email, or phone_number'}), 400

        # Search customer
        exists, customer_data = check_customer_exists(
            national_id=national_id,
            email=email,
            phone_number=phone_number
        )

        if not exists:
            return jsonify({
                'found': False,
                'message': 'Customer not found in database'
            })

        # Return customer data
        return jsonify({
            'found': True,
            'customer': {
                'full_name': customer_data.get('full_name'),
                'national_id': customer_data.get('national_id'),
                'email': customer_data.get('email'),
                'phone_number': customer_data.get('phone_number'),
                'credit_score': float(customer_data.get('credit_score', 0)) if customer_data.get('credit_score') else None,
                'credit_group': customer_data.get('credit_group'),
                'last_prediction': customer_data.get('last_prediction'),
                'created_at': customer_data.get('created_at')
            }
        })

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
