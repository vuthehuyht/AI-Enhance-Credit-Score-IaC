# API Gateway + Lambda - DEPRECATED
#
# All resources disabled. Using Flask API on EC2 instead.
# See compute.tf and flask-app/ directory for new implementation.
#
# Flask API Endpoints:
# - POST /api/v1/predict (smart routing: existing vs new customers)
# - GET/POST /api/v1/search (customer history lookup)
#
# Cost Benefits:
# - No API Gateway charges
# - No Lambda invocation fees
# - No SageMaker endpoint costs
# - Single EC2 instance handles inference
