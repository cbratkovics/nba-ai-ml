"""
Production training pipeline for NBA ML models
Orchestrates complete training workflow with MLOps best practices
"""
import os
import sys
import asyncio
import yaml
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

# Conditional MLflow import
MLFLOW_AVAILABLE = False
if os.getenv('ENVIRONMENT', 'development') != 'production':
    try:
        import mlflow
        import mlflow.sklearn
        MLFLOW_AVAILABLE = True
    except ImportError:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from database.connection import DatabaseManager, init_db
from database.models import Player, Game, PlayerGameLog, Model as ModelDB
from ml.data.processors.data_validator import NBADataValidator, ValidationStatus
from ml.data.processors.feature_engineer import NBAFeatureEngineer
from ml.models.ensemble import NBAEnsemble
from ml.training.hyperparameter_tuning import HyperparameterOptimizer
from ml.training.validation import TimeSeriesValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionTrainingPipeline:
    """
    Automated training pipeline with MLOps best practices
    Handles data loading, validation, training, and model registration
    """
    
    def __init__(self, config_path: str = "config/training.yaml"):
        """
        Initialize training pipeline
        
        Args:
            config_path: Path to training configuration file
        """
        self.config = self._load_config(config_path)
        self.mlflow_tracking_uri = None
        if os.getenv('ENVIRONMENT', 'development') != 'production' and MLFLOW_AVAILABLE:
            self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
            # Initialize MLflow
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        self.data_validator = NBADataValidator()
        self.feature_engineer = NBAFeatureEngineer()
        self.time_validator = TimeSeriesValidator()
        
        # Database connection
        self.db_manager = None
        
        logger.info(f"Training pipeline initialized with config: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Use default configuration
            return self._get_default_config()
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict:
        """Get default training configuration"""
        return {
            "training": {
                "targets": ["PTS", "REB", "AST"],
                "seasons": ["2021-22", "2022-23", "2023-24"],
                "data": {
                    "min_games_per_player": 20,
                    "min_minutes_per_game": 10,
                    "exclude_garbage_time": True
                },
                "features": {
                    "engineering": {
                        "rolling_windows": [3, 5, 10, 20],
                        "include_advanced_stats": True,
                        "include_matchup_history": True
                    }
                },
                "models": {
                    "random_forest": {
                        "enabled": True,
                        "hyperparameter_tuning": True,
                        "n_trials": 50
                    },
                    "xgboost": {
                        "enabled": True,
                        "hyperparameter_tuning": True,
                        "n_trials": 50
                    },
                    "lightgbm": {
                        "enabled": True,
                        "hyperparameter_tuning": True,
                        "n_trials": 50
                    },
                    "neural_network": {
                        "enabled": False,
                        "epochs": 100,
                        "batch_size": 32
                    }
                },
                "evaluation": {
                    "metrics": ["r2", "mae", "rmse"],
                    "confidence_intervals": True,
                    "backtesting_seasons": 1
                },
                "mlflow": {
                    "experiment_name": "nba_player_prediction",
                    "artifact_location": "./mlruns"
                },
                "performance_targets": {
                    "min_r2_score": 0.90,
                    "max_mae": 4.0,
                    "max_rmse": 5.5
                }
            }
        }
    
    async def train(self,
                   target: str = "PTS",
                   seasons: Optional[List[str]] = None,
                   experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete training pipeline with MLflow tracking
        
        Args:
            target: Target variable to predict (PTS, REB, AST, etc.)
            seasons: List of seasons to use for training
            experiment_name: MLflow experiment name
            
        Returns:
            Dictionary with training results and model metadata
        """
        logger.info(f"Starting training pipeline for target: {target}")
        
        # Setup
        seasons = seasons or self.config["training"]["seasons"]
        experiment_name = experiment_name or self.config["training"]["mlflow"]["experiment_name"]
        
        # Set MLflow experiment
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
        
        # Start MLflow run or use context manager
        mlflow_context = mlflow.start_run(run_name=f"{target}_training_{datetime.now():%Y%m%d_%H%M%S}") if MLFLOW_AVAILABLE else None
        
        try:
            if mlflow_context:
                mlflow_context.__enter__()
            
            # Log parameters
            if MLFLOW_AVAILABLE:
                mlflow.log_param("target", target)
                mlflow.log_param("seasons", seasons)
                mlflow.log_param("config", self.config)
            
            try:
                # Step 1: Load data
                logger.info("Loading training data...")
                df = await self.load_training_data(seasons)
                logger.info(f"Loaded {len(df)} samples")
                
                # Step 2: Validate data
                logger.info("Validating data quality...")
                validation_result = self.data_validator.validate_player_game_log(df)
                
                if validation_result.status == ValidationStatus.FAILED:
                    logger.error(f"Data validation failed: {validation_result.errors}")
                    if MLFLOW_AVAILABLE:
                        mlflow.log_metric("data_validation_passed", 0)
                    raise ValueError("Data validation failed")
                
                if MLFLOW_AVAILABLE:
                    mlflow.log_metric("data_validation_passed", 1)
                    mlflow.log_metric("data_validation_warnings", validation_result.warning_checks)
                
                # Step 3: Engineer features
                logger.info("Engineering features...")
                features_df = self.feature_engineer.create_features(df)
                feature_columns = [col for col in features_df.columns if col not in 
                                 ['GAME_DATE', 'GAME_ID', 'PLAYER_ID', 'TEAM_ID', target]]
                
                logger.info(f"Created {len(feature_columns)} features")
                if MLFLOW_AVAILABLE:
                    mlflow.log_metric("n_features", len(feature_columns))
                
                # Step 4: Create time-based splits
                logger.info("Creating time-based data splits...")
                train_data, val_data, test_data = self.create_time_based_splits(
                    features_df, target_col=target
                )
                
                X_train, y_train = train_data
                X_val, y_val = val_data
                X_test, y_test = test_data
                
                logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
                if MLFLOW_AVAILABLE:
                    mlflow.log_metric("train_samples", len(X_train))
                    mlflow.log_metric("val_samples", len(X_val))
                    mlflow.log_metric("test_samples", len(X_test))
                
                # Step 5: Hyperparameter tuning (if enabled)
                best_params = {}
                if self.config["training"]["models"]["xgboost"]["hyperparameter_tuning"]:
                    logger.info("Running hyperparameter optimization...")
                    optimizer = HyperparameterOptimizer()
                    best_params = await self.hyperparameter_tuning(
                        X_train, y_train, X_val, y_val, target
                    )
                    if MLFLOW_AVAILABLE:
                        mlflow.log_params(best_params)
                
                # Step 6: Train ensemble
                logger.info("Training ensemble model...")
                ensemble = self.train_ensemble(X_train, y_train, best_params, target)
                
                # Step 7: Evaluate model
                logger.info("Evaluating model performance...")
                metrics = self.evaluate_model(ensemble, X_test, y_test)
                
                # Log metrics
                for metric_name, value in metrics.items():
                    if MLFLOW_AVAILABLE:
                        mlflow.log_metric(metric_name, value)
                
                logger.info(f"Test R²: {metrics['test_r2']:.4f}, MAE: {metrics['test_mae']:.4f}")
                
                # Step 8: Check performance targets
                if not self._check_performance_targets(metrics):
                    logger.warning("Model did not meet performance targets")
                    if MLFLOW_AVAILABLE:
                        mlflow.log_metric("meets_targets", 0)
                else:
                    logger.info("Model meets all performance targets!")
                    if MLFLOW_AVAILABLE:
                        mlflow.log_metric("meets_targets", 1)
                
                # Step 9: Register model
                logger.info("Registering model...")
                model_info = self.register_model(
                    ensemble, metrics, target, feature_columns
                )
                
                # Step 10: Generate reports
                self._generate_training_report(metrics, validation_result, model_info)
                
                return {
                    "model": ensemble,
                    "metrics": metrics,
                    "model_info": model_info,
                    "validation": validation_result.to_dict(),
                    "feature_columns": feature_columns
                }
                
            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                if MLFLOW_AVAILABLE:
                    mlflow.log_metric("training_failed", 1)
                raise
        finally:
            if mlflow_context:
                mlflow_context.__exit__(None, None, None)
    
    async def load_training_data(self, seasons: List[str]) -> pd.DataFrame:
        """
        Load data from PostgreSQL with partitions
        
        Args:
            seasons: List of seasons to load
            
        Returns:
            DataFrame with player game logs
        """
        if not self.db_manager:
            self.db_manager = init_db()
        
        all_data = []
        
        with self.db_manager.get_db(read_only=True) as session:
            for season in seasons:
                logger.info(f"Loading season {season}")
                
                # Query player game logs
                query = session.query(PlayerGameLog).filter(
                    PlayerGameLog.minutes_played >= self.config["training"]["data"]["min_minutes_per_game"]
                )
                
                # Convert to DataFrame
                season_data = pd.read_sql(query.statement, session.bind)
                
                if not season_data.empty:
                    season_data['season'] = season
                    all_data.append(season_data)
        
        if not all_data:
            # Use mock data for demo
            logger.warning("No data in database, generating mock data...")
            return self._generate_mock_training_data(seasons)
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Apply filters
        min_games = self.config["training"]["data"]["min_games_per_player"]
        player_game_counts = df.groupby('player_id').size()
        valid_players = player_game_counts[player_game_counts >= min_games].index
        df = df[df['player_id'].isin(valid_players)]
        
        return df
    
    def _generate_mock_training_data(self, seasons: List[str]) -> pd.DataFrame:
        """Generate mock training data for demo"""
        np.random.seed(42)
        
        n_players = 50
        n_games_per_player = 82
        
        data = []
        
        for season in seasons:
            for player_id in range(n_players):
                for game_num in range(n_games_per_player):
                    # Generate realistic stats
                    minutes = np.random.uniform(15, 38)
                    usage_factor = minutes / 35
                    
                    game_data = {
                        'player_id': f"player_{player_id}",
                        'game_id': f"game_{season}_{player_id}_{game_num}",
                        'game_date': pd.Timestamp(f"2023-01-01") + pd.Timedelta(days=game_num*2),
                        'season': season,
                        'minutes_played': minutes,
                        'points': np.random.poisson(20 * usage_factor),
                        'rebounds': np.random.poisson(8 * usage_factor),
                        'assists': np.random.poisson(5 * usage_factor),
                        'field_goals_made': np.random.poisson(7 * usage_factor),
                        'field_goals_attempted': np.random.poisson(15 * usage_factor),
                        'three_pointers_made': np.random.poisson(2 * usage_factor),
                        'three_pointers_attempted': np.random.poisson(6 * usage_factor),
                        'free_throws_made': np.random.poisson(4 * usage_factor),
                        'free_throws_attempted': np.random.poisson(5 * usage_factor),
                        'steals': np.random.poisson(1.5),
                        'blocks': np.random.poisson(1),
                        'turnovers': np.random.poisson(3),
                        'personal_fouls': np.random.poisson(2.5),
                        'plus_minus': np.random.randint(-15, 15),
                        'home_game': np.random.choice([0, 1]),
                        'rest_days': np.random.choice([0, 1, 2, 3]),
                        'back_to_back': np.random.choice([0, 1], p=[0.7, 0.3])
                    }
                    
                    # Calculate percentages
                    if game_data['field_goals_attempted'] > 0:
                        game_data['field_goal_percentage'] = (
                            game_data['field_goals_made'] / game_data['field_goals_attempted']
                        )
                    else:
                        game_data['field_goal_percentage'] = 0
                    
                    # Ensure data consistency
                    game_data['field_goals_made'] = min(
                        game_data['field_goals_made'],
                        game_data['field_goals_attempted']
                    )
                    game_data['three_pointers_made'] = min(
                        game_data['three_pointers_made'],
                        game_data['three_pointers_attempted'],
                        game_data['field_goals_made']
                    )
                    game_data['free_throws_made'] = min(
                        game_data['free_throws_made'],
                        game_data['free_throws_attempted']
                    )
                    
                    # Rename columns to match database schema
                    game_data['PTS'] = game_data.pop('points')
                    game_data['REB'] = game_data.pop('rebounds')
                    game_data['AST'] = game_data.pop('assists')
                    game_data['FGM'] = game_data.pop('field_goals_made')
                    game_data['FGA'] = game_data.pop('field_goals_attempted')
                    game_data['FG3M'] = game_data.pop('three_pointers_made')
                    game_data['FG3A'] = game_data.pop('three_pointers_attempted')
                    game_data['FTM'] = game_data.pop('free_throws_made')
                    game_data['FTA'] = game_data.pop('free_throws_attempted')
                    game_data['STL'] = game_data.pop('steals')
                    game_data['BLK'] = game_data.pop('blocks')
                    game_data['TOV'] = game_data.pop('turnovers')
                    game_data['PF'] = game_data.pop('personal_fouls')
                    game_data['MIN'] = game_data.pop('minutes_played')
                    game_data['FG_PCT'] = game_data.pop('field_goal_percentage')
                    game_data['PLUS_MINUS'] = game_data.pop('plus_minus')
                    game_data['HOME_GAME'] = game_data.pop('home_game')
                    game_data['REST_DAYS'] = game_data.pop('rest_days')
                    game_data['BACK_TO_BACK'] = game_data.pop('back_to_back')
                    game_data['GAME_DATE'] = game_data.pop('game_date')
                    game_data['GAME_ID'] = game_data.pop('game_id')
                    game_data['PLAYER_ID'] = game_data.pop('player_id')
                    
                    # Add team columns
                    game_data['TEAM_ID'] = f"team_{player_id % 30}"
                    game_data['DREB'] = game_data['REB'] - np.random.randint(0, 3)
                    game_data['OREB'] = game_data['REB'] - game_data['DREB']
                    
                    data.append(game_data)
        
        return pd.DataFrame(data)
    
    def create_time_based_splits(self, 
                                df: pd.DataFrame,
                                target_col: str) -> Tuple[Tuple[pd.DataFrame, pd.Series], ...]:
        """
        Create time-based train/validation/test splits
        
        Args:
            df: Feature DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        # Sort by date
        df = df.sort_values('GAME_DATE')
        
        # Remove non-feature columns
        feature_cols = [col for col in df.columns if col not in 
                       ['GAME_DATE', 'GAME_ID', 'PLAYER_ID', 'TEAM_ID', target_col]]
        
        # 70% train, 15% validation, 15% test
        n = len(df)
        train_size = int(0.70 * n)
        val_size = int(0.15 * n)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        # Validate no data leakage
        self.time_validator.validate_no_future_leakage(
            train_df['GAME_DATE'].values,
            test_df['GAME_DATE'].values
        )
        
        # Extract features and targets
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    async def hyperparameter_tuning(self, 
                                   X_train: pd.DataFrame,
                                   y_train: pd.Series,
                                   X_val: pd.DataFrame,
                                   y_val: pd.Series,
                                   target: str) -> Dict[str, Any]:
        """
        Optuna-based hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            target: Target name
            
        Returns:
            Dictionary of best hyperparameters
        """
        optimizer = HyperparameterOptimizer()
        
        best_params = {}
        
        # Optimize each model type
        for model_type in ["random_forest", "xgboost", "lightgbm"]:
            if self.config["training"]["models"][model_type]["enabled"]:
                logger.info(f"Optimizing {model_type}...")
                
                n_trials = self.config["training"]["models"][model_type].get("n_trials", 50)
                
                params = await optimizer.optimize(
                    model_type=model_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    n_trials=n_trials,
                    target_name=target
                )
                
                best_params[model_type] = params
                logger.info(f"Best params for {model_type}: {params}")
        
        return best_params
    
    def train_ensemble(self, 
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      best_params: Dict[str, Any],
                      target: str) -> NBAEnsemble:
        """
        Train the full ensemble model
        
        Args:
            X_train: Training features
            y_train: Training target
            best_params: Optimized hyperparameters
            target: Target name
            
        Returns:
            Trained ensemble model
        """
        # Initialize ensemble
        ensemble = NBAEnsemble(
            target=target,
            use_neural_net=self.config["training"]["models"]["neural_network"]["enabled"]
        )
        
        # Apply best hyperparameters
        if best_params:
            for model_type, params in best_params.items():
                if model_type in ensemble.models:
                    ensemble.models[model_type].set_params(**params)
        
        # Train ensemble
        ensemble.train(
            X_train, y_train,
            track_mlflow=False  # We're already tracking in the parent run
        )
        
        return ensemble
    
    def evaluate_model(self, 
                      model: NBAEnsemble,
                      X_test: pd.DataFrame,
                      y_test: pd.Series) -> Dict[str, float]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        if self.config["training"]["evaluation"]["confidence_intervals"]:
            y_pred, lower, upper = model.predict(X_test, return_uncertainty=True)
            
            # Calculate coverage
            coverage = np.mean((y_test >= lower) & (y_test <= upper))
            interval_width = np.mean(upper - lower)
        else:
            y_pred = model.predict(X_test)
            coverage = None
            interval_width = None
        
        # Calculate metrics
        metrics = {
            "test_r2": r2_score(y_test, y_pred),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "test_mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        if coverage is not None:
            metrics["test_coverage"] = coverage
            metrics["test_interval_width"] = interval_width
        
        # Calculate metrics by player (check for bias)
        if 'PLAYER_ID' in X_test.columns:
            player_metrics = []
            for player_id in X_test['PLAYER_ID'].unique()[:10]:  # Check top 10 players
                mask = X_test['PLAYER_ID'] == player_id
                if mask.sum() > 5:
                    player_r2 = r2_score(y_test[mask], y_pred[mask])
                    player_metrics.append(player_r2)
            
            metrics["test_r2_std_by_player"] = np.std(player_metrics) if player_metrics else 0
        
        return metrics
    
    def register_model(self, 
                      model: NBAEnsemble,
                      metrics: Dict[str, float],
                      target: str,
                      feature_columns: List[str]) -> Dict[str, Any]:
        """
        Register model in MLflow and database
        
        Args:
            model: Trained model
            metrics: Evaluation metrics
            target: Target variable
            feature_columns: List of feature names
            
        Returns:
            Model registration info
        """
        # Save model to MLflow
        if MLFLOW_AVAILABLE:
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=f"nba_{target.lower()}_predictor"
            )
            
            # Save feature names
            mlflow.log_text("\n".join(feature_columns), "features.txt")
        
        # Save model locally
        model_path = f"models/{target.lower()}_ensemble_v{datetime.now():%Y%m%d}.pkl"
        os.makedirs("models", exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # Register in database
        if self.db_manager:
            with self.db_manager.transaction() as session:
                model_db = ModelDB(
                    name=f"nba_{target.lower()}_predictor",
                    version=f"v{datetime.now():%Y%m%d_%H%M%S}",
                    target=target,
                    model_type="ensemble",
                    framework="sklearn",
                    artifact_uri=model_path,
                    val_r2_score=metrics.get("test_r2"),
                    val_mae=metrics.get("test_mae"),
                    val_rmse=metrics.get("test_rmse"),
                    feature_count=len(feature_columns),
                    status="staged",
                    created_by="training_pipeline"
                )
                session.add(model_db)
                session.flush()
                
                model_id = model_db.id
        else:
            model_id = None
        
        return {
            "model_id": model_id,
            "model_path": model_path,
            "mlflow_run_id": mlflow.active_run().info.run_id if MLFLOW_AVAILABLE and mlflow.active_run() else None,
            "version": f"v{datetime.now():%Y%m%d_%H%M%S}"
        }
    
    def _check_performance_targets(self, metrics: Dict[str, float]) -> bool:
        """Check if model meets performance targets"""
        targets = self.config["training"]["performance_targets"]
        
        checks = [
            metrics.get("test_r2", 0) >= targets.get("min_r2_score", 0.90),
            metrics.get("test_mae", 999) <= targets.get("max_mae", 4.0),
            metrics.get("test_rmse", 999) <= targets.get("max_rmse", 5.5)
        ]
        
        return all(checks)
    
    def _generate_training_report(self, 
                                 metrics: Dict[str, float],
                                 validation_result: Any,
                                 model_info: Dict[str, Any]):
        """Generate training report"""
        report = []
        report.append("=" * 60)
        report.append("TRAINING REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now()}")
        report.append(f"Model ID: {model_info.get('model_id')}")
        report.append(f"MLflow Run: {model_info.get('mlflow_run_id')}")
        report.append("")
        report.append("Performance Metrics:")
        for metric, value in metrics.items():
            report.append(f"  {metric}: {value:.4f}")
        report.append("")
        report.append(f"Data Validation: {validation_result.status.value}")
        report.append(f"Performance Targets Met: {self._check_performance_targets(metrics)}")
        
        report_text = "\n".join(report)
        
        # Log to MLflow
        if MLFLOW_AVAILABLE:
            mlflow.log_text(report_text, "training_report.txt")
        
        # Save locally
        os.makedirs("reports", exist_ok=True)
        report_path = f"reports/training_report_{datetime.now():%Y%m%d_%H%M%S}.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Training report saved to {report_path}")


async def main():
    """Example usage"""
    pipeline = ProductionTrainingPipeline()
    
    # Train model for points prediction
    result = await pipeline.train(
        target="PTS",
        seasons=["2022-23", "2023-24"],
        experiment_name="nba_points_prediction"
    )
    
    print(f"Training completed!")
    print(f"Test R²: {result['metrics']['test_r2']:.4f}")
    print(f"Test MAE: {result['metrics']['test_mae']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())