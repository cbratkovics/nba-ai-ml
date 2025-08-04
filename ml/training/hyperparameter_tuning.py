"""
Hyperparameter optimization for NBA ML models
Uses Optuna for Bayesian optimization
"""
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

try:
    import optuna
    from optuna import Trial
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    Trial = None

logger = logging.getLogger(__name__)


def get_search_space(model_type: str) -> Dict[str, Any]:
    """
    Define hyperparameter search spaces for each model type
    
    Args:
        model_type: Type of model (random_forest, xgboost, lightgbm)
        
    Returns:
        Dictionary defining search space
    """
    spaces = {
        "random_forest": {
            "n_estimators": [100, 300, 500, 700, 1000],
            "max_depth": [10, 15, 20, 25, 30, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
            "bootstrap": [True, False]
        },
        "xgboost": {
            "n_estimators": [100, 300, 500, 700, 1000],
            "max_depth": [3, 5, 7, 9, 11],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "gamma": [0, 0.1, 0.2, 0.3, 0.4],
            "reg_alpha": [0, 0.001, 0.01, 0.1, 1],
            "reg_lambda": [0, 0.001, 0.01, 0.1, 1]
        },
        "lightgbm": {
            "n_estimators": [100, 300, 500, 700, 1000],
            "num_leaves": [31, 50, 70, 100, 150],
            "max_depth": [-1, 10, 20, 30],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
            "feature_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
            "bagging_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
            "bagging_freq": [0, 1, 3, 5, 7],
            "lambda_l1": [0, 0.001, 0.01, 0.1, 1],
            "lambda_l2": [0, 0.001, 0.01, 0.1, 1],
            "min_child_samples": [5, 10, 20, 30, 50]
        }
    }
    return spaces.get(model_type, {})


class HyperparameterOptimizer:
    """
    Bayesian hyperparameter optimization for NBA models
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize optimizer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default hyperparameters")
    
    async def optimize(self,
                       model_type: str,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: pd.DataFrame,
                       y_val: pd.Series,
                       n_trials: int = 100,
                       target_name: str = "target") -> Dict[str, Any]:
        """
        Run hyperparameter optimization
        
        Args:
            model_type: Type of model to optimize
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            n_trials: Number of optimization trials
            target_name: Name of target variable
            
        Returns:
            Dictionary of best hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            return self._get_default_params(model_type)
        
        logger.info(f"Starting hyperparameter optimization for {model_type}")
        
        # Create Optuna study
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.random_state),
            study_name=f"{model_type}_{target_name}_optimization"
        )
        
        # Define objective function
        def objective(trial: Trial) -> float:
            params = self._suggest_params(trial, model_type)
            
            # Train model with suggested params
            if model_type == "random_forest":
                model = RandomForestRegressor(**params, random_state=self.random_state, n_jobs=-1)
            elif model_type == "xgboost":
                model = xgb.XGBRegressor(**params, random_state=self.random_state, n_jobs=-1, verbosity=0)
            elif model_type == "lightgbm":
                model = lgb.LGBMRegressor(**params, random_state=self.random_state, n_jobs=-1, verbosity=-1)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Use negative R² as loss (Optuna minimizes)
            mae = mean_absolute_error(y_val, y_pred)
            
            return mae
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best {model_type} params: {best_params}")
        logger.info(f"Best validation MAE: {best_value:.4f}")
        
        return best_params
    
    def _suggest_params(self, trial: Trial, model_type: str) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial
        
        Args:
            trial: Optuna trial object
            model_type: Type of model
            
        Returns:
            Dictionary of suggested parameters
        """
        params = {}
        
        if model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_categorical("max_depth", [10, 15, 20, 25, 30, None]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
            }
            
        elif model_type == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 11, step=2),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1, log=True)
            }
            
        elif model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "num_leaves": trial.suggest_int("num_leaves", 31, 200),
                "max_depth": trial.suggest_categorical("max_depth", [-1, 10, 20, 30]),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
                "lambda_l1": trial.suggest_float("lambda_l1", 0, 1, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 0, 1, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50)
            }
        
        return params
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """
        Get default parameters when Optuna is not available
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of default parameters
        """
        defaults = {
            "random_forest": {
                "n_estimators": 500,
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True
            },
            "xgboost": {
                "n_estimators": 500,
                "max_depth": 7,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0.1,
                "reg_alpha": 0.01,
                "reg_lambda": 0.01
            },
            "lightgbm": {
                "n_estimators": 500,
                "num_leaves": 70,
                "max_depth": -1,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l1": 0.01,
                "lambda_l2": 0.01,
                "min_child_samples": 20
            }
        }
        
        return defaults.get(model_type, {})
    
    def grid_search(self,
                   model_type: str,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: pd.DataFrame,
                   y_val: pd.Series,
                   param_grid: Optional[Dict] = None) -> Tuple[Dict[str, Any], float]:
        """
        Fallback grid search when Optuna is not available
        
        Args:
            model_type: Type of model
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            param_grid: Custom parameter grid
            
        Returns:
            Tuple of (best_params, best_score)
        """
        if param_grid is None:
            param_grid = self._get_simplified_grid(model_type)
        
        best_params = None
        best_score = float('-inf')
        
        # Generate all parameter combinations
        from itertools import product
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combination in product(*values):
            params = dict(zip(keys, combination))
            
            # Train model
            if model_type == "random_forest":
                model = RandomForestRegressor(**params, random_state=self.random_state, n_jobs=-1)
            elif model_type == "xgboost":
                model = xgb.XGBRegressor(**params, random_state=self.random_state, n_jobs=-1, verbosity=0)
            elif model_type == "lightgbm":
                model = lgb.LGBMRegressor(**params, random_state=self.random_state, n_jobs=-1, verbosity=-1)
            else:
                continue
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = r2_score(y_val, y_pred)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        logger.info(f"Grid search best params: {best_params}")
        logger.info(f"Grid search best R²: {best_score:.4f}")
        
        return best_params, best_score
    
    def _get_simplified_grid(self, model_type: str) -> Dict[str, list]:
        """
        Get simplified parameter grid for grid search
        
        Args:
            model_type: Type of model
            
        Returns:
            Simplified parameter grid
        """
        grids = {
            "random_forest": {
                "n_estimators": [300, 500],
                "max_depth": [15, 20, None],
                "min_samples_split": [5, 10]
            },
            "xgboost": {
                "n_estimators": [300, 500],
                "max_depth": [5, 7],
                "learning_rate": [0.05, 0.1]
            },
            "lightgbm": {
                "n_estimators": [300, 500],
                "num_leaves": [50, 70],
                "learning_rate": [0.05, 0.1]
            }
        }
        
        return grids.get(model_type, {})