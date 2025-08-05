# Import all models to make them available
# Import everything from the models.py file in the parent api directory
import sys
import os

# Get the path to the parent api directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Import the models module
import importlib.util
spec = importlib.util.spec_from_file_location("api_models", os.path.join(parent_dir, "models.py"))
api_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_models)

# Import all classes from the loaded module
PredictionRequest = api_models.PredictionRequest
PredictionResponse = api_models.PredictionResponse
BatchPredictionRequest = api_models.BatchPredictionRequest
BatchPredictionResponse = api_models.BatchPredictionResponse
ExperimentRequest = api_models.ExperimentRequest
ExperimentResponse = api_models.ExperimentResponse
InsightRequest = api_models.InsightRequest
InsightResponse = api_models.InsightResponse
HealthCheckResponse = api_models.HealthCheckResponse
ErrorResponse = api_models.ErrorResponse
ModelVersion = api_models.ModelVersion
PredictionTarget = api_models.PredictionTarget

# Re-export all models
__all__ = [
    'PredictionRequest',
    'PredictionResponse',
    'BatchPredictionRequest',
    'BatchPredictionResponse',
    'ExperimentRequest',
    'ExperimentResponse',
    'InsightRequest',
    'InsightResponse',
    'HealthCheckResponse',
    'ErrorResponse',
    'ModelVersion',
    'PredictionTarget'
]