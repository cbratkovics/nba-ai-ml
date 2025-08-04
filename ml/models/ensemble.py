"""
Production ensemble model for NBA predictions
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import BayesianRidge
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import logging
from datetime import datetime

# Conditional MLflow import
MLFLOW_AVAILABLE = False
if os.getenv('ENVIRONMENT', 'development') != 'production':
    try:
        import mlflow
        import mlflow.sklearn
        import mlflow.pytorch
        MLFLOW_AVAILABLE = True
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("MLflow not available, continuing without experiment tracking")

logger = logging.getLogger(__name__)


class NeuralNetwork(nn.Module):
    """Deep learning model for NBA predictions"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class NeuralNetRegressor:
    """Scikit-learn compatible neural network regressor"""
    
    def __init__(self, input_dim: int, epochs: int = 100, batch_size: int = 32):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        """Train the neural network"""
        self.model = NeuralNetwork(self.input_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        return predictions


class NBAEnsemble:
    """
    Production ensemble with uncertainty quantification
    Combines multiple models for robust predictions
    """
    
    def __init__(self, target: str = 'PTS', use_neural_net: bool = True):
        self.target = target
        self.use_neural_net = use_neural_net
        
        # Initialize base models
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=7,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'lgbm': lgb.LGBMRegressor(
                n_estimators=800,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
        }
        
        # Meta learner for stacking
        self.meta_learner = BayesianRidge()
        
        # Track performance metrics
        self.metrics = {}
        self.feature_importance = {}
        
    def train(self, 
             X_train: pd.DataFrame,
             y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None,
             track_mlflow: bool = True) -> Dict[str, float]:
        """
        Train ensemble model with MLflow tracking
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            track_mlflow: Whether to track with MLflow
            
        Returns:
            Dictionary of performance metrics
        """
        if track_mlflow and MLFLOW_AVAILABLE:
            mlflow.start_run(run_name=f"nba_ensemble_{self.target}_{datetime.now():%Y%m%d_%H%M%S}")
            mlflow.log_param("target", self.target)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_samples", X_train.shape[0])
        
        # Add neural network if specified
        if self.use_neural_net:
            self.models['nn'] = NeuralNetRegressor(input_dim=X_train.shape[1])
        
        # Train base models
        base_predictions_train = []
        base_predictions_val = [] if X_val is not None else None
        
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions
            train_pred = model.predict(X_train)
            base_predictions_train.append(train_pred)
            
            if X_val is not None:
                val_pred = model.predict(X_val)
                base_predictions_val.append(val_pred)
                
                # Calculate individual model metrics
                val_r2 = r2_score(y_val, val_pred)
                val_mae = mean_absolute_error(y_val, val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                
                self.metrics[f'{name}_r2'] = val_r2
                self.metrics[f'{name}_mae'] = val_mae
                self.metrics[f'{name}_rmse'] = val_rmse
                
                logger.info(f"{name} - R2: {val_r2:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
                
                if track_mlflow and MLFLOW_AVAILABLE:
                    mlflow.log_metric(f"{name}_r2", val_r2)
                    mlflow.log_metric(f"{name}_mae", val_mae)
                    mlflow.log_metric(f"{name}_rmse", val_rmse)
            
            # Extract feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        # Train meta learner (stacking)
        logger.info("Training meta learner...")
        base_predictions_train = np.column_stack(base_predictions_train)
        self.meta_learner.fit(base_predictions_train, y_train)
        
        # Final ensemble predictions
        if X_val is not None:
            base_predictions_val = np.column_stack(base_predictions_val)
            ensemble_pred = self.meta_learner.predict(base_predictions_val)
            
            # Calculate ensemble metrics
            ensemble_r2 = r2_score(y_val, ensemble_pred)
            ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
            ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            
            self.metrics['ensemble_r2'] = ensemble_r2
            self.metrics['ensemble_mae'] = ensemble_mae
            self.metrics['ensemble_rmse'] = ensemble_rmse
            
            logger.info(f"Ensemble - R2: {ensemble_r2:.4f}, MAE: {ensemble_mae:.4f}, RMSE: {ensemble_rmse:.4f}")
            
            if track_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_metric("ensemble_r2", ensemble_r2)
                mlflow.log_metric("ensemble_mae", ensemble_mae)
                mlflow.log_metric("ensemble_rmse", ensemble_rmse)
                
                # Log model
                mlflow.sklearn.log_model(self, "ensemble_model")
        
        # Perform cross-validation
        self._perform_cross_validation(X_train, y_train, track_mlflow)
        
        if track_mlflow and MLFLOW_AVAILABLE:
            mlflow.end_run()
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame, return_uncertainty: bool = False) -> np.ndarray:
        """
        Make predictions with optional uncertainty quantification
        
        Args:
            X: Features for prediction
            return_uncertainty: Whether to return prediction intervals
            
        Returns:
            Predictions (and optionally uncertainty intervals)
        """
        base_predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            base_predictions.append(pred)
        
        base_predictions = np.column_stack(base_predictions)
        ensemble_pred = self.meta_learner.predict(base_predictions)
        
        if return_uncertainty:
            # Calculate prediction intervals using base model variance
            std_dev = np.std(base_predictions, axis=1)
            lower_bound = ensemble_pred - 1.96 * std_dev
            upper_bound = ensemble_pred + 1.96 * std_dev
            
            return ensemble_pred, lower_bound, upper_bound
        
        return ensemble_pred
    
    def _perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, track_mlflow: bool):
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]
            
            # Train simplified ensemble for CV
            fold_predictions = []
            for name, model in self.models.items():
                if name != 'nn':  # Skip NN for faster CV
                    model_copy = model.__class__(**model.get_params())
                    model_copy.fit(X_fold_train, y_fold_train)
                    fold_predictions.append(model_copy.predict(X_fold_val))
            
            # Simple average for CV
            ensemble_fold_pred = np.mean(fold_predictions, axis=0)
            fold_score = r2_score(y_fold_val, ensemble_fold_pred)
            cv_scores.append(fold_score)
            
            logger.info(f"CV Fold {fold+1} R2: {fold_score:.4f}")
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        self.metrics['cv_r2_mean'] = cv_mean
        self.metrics['cv_r2_std'] = cv_std
        
        logger.info(f"Cross-validation R2: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        if track_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metric("cv_r2_mean", cv_mean)
            mlflow.log_metric("cv_r2_std", cv_std)
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get aggregated feature importance"""
        importance_df = pd.DataFrame()
        
        for name, importance in self.feature_importance.items():
            importance_df[name] = importance
        
        importance_df['feature'] = feature_names[:len(importance_df)]
        importance_df['mean_importance'] = importance_df.drop('feature', axis=1).mean(axis=1)
        importance_df = importance_df.sort_values('mean_importance', ascending=False)
        
        return importance_df
    
    def save(self, path: str):
        """Save ensemble model"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load ensemble model"""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model


class ModelExperiment:
    """A/B testing framework for model variants"""
    
    def __init__(self, control_model: NBAEnsemble, treatment_model: NBAEnsemble):
        self.control_model = control_model
        self.treatment_model = treatment_model
        self.results = {
            'control': {'predictions': [], 'actuals': []},
            'treatment': {'predictions': [], 'actuals': []}
        }
        
    def route_prediction(self, user_id: str, traffic_split: float = 0.5) -> str:
        """Route user to control or treatment model"""
        # Hash-based assignment for consistency
        hash_value = hash(user_id) % 100
        return 'treatment' if hash_value < traffic_split * 100 else 'control'
    
    def predict(self, X: pd.DataFrame, user_id: str) -> Tuple[np.ndarray, str]:
        """Make prediction with assigned model"""
        variant = self.route_prediction(user_id)
        
        if variant == 'treatment':
            predictions = self.treatment_model.predict(X)
        else:
            predictions = self.control_model.predict(X)
        
        return predictions, variant
    
    def log_outcome(self, variant: str, prediction: float, actual: float):
        """Log prediction outcome for analysis"""
        self.results[variant]['predictions'].append(prediction)
        self.results[variant]['actuals'].append(actual)
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze A/B test results"""
        analysis = {}
        
        for variant in ['control', 'treatment']:
            preds = np.array(self.results[variant]['predictions'])
            actuals = np.array(self.results[variant]['actuals'])
            
            if len(preds) > 0:
                analysis[variant] = {
                    'r2_score': r2_score(actuals, preds),
                    'mae': mean_absolute_error(actuals, preds),
                    'rmse': np.sqrt(mean_squared_error(actuals, preds)),
                    'n_samples': len(preds)
                }
        
        # Calculate lift
        if 'control' in analysis and 'treatment' in analysis:
            analysis['lift'] = {
                'r2_lift': analysis['treatment']['r2_score'] - analysis['control']['r2_score'],
                'mae_improvement': analysis['control']['mae'] - analysis['treatment']['mae']
            }
        
        return analysis