#!/usr/bin/env python3
"""
Production NBA Model Training with Real Data
Trains high-accuracy prediction models using collected NBA data
Implements advanced MLOps practices for portfolio showcase
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import DatabaseManager, init_db
from database.models import PlayerGameLog, Player, Team, Game
from ml.training.pipeline import ProductionTrainingPipeline
from ml.training.validation import TimeSeriesValidator
from ml.training.hyperparameter_tuning import HyperparameterOptimizer
from ml.data.processors.data_validator import NBADataValidator
from ml.data.processors.feature_engineering import FeatureEngineer
from sqlalchemy import and_, func, text
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/production_training_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionModelTrainer:
    """Production-grade NBA model trainer with real data"""
    
    def __init__(self, config_path: str = "config/training.yaml"):
        """
        Initialize production trainer
        
        Args:
            config_path: Path to training configuration
        """
        self.config = self._load_config(config_path)
        self.db_manager = None
        self.validator = NBADataValidator()
        self.time_validator = TimeSeriesValidator()
        self.optimizer = HyperparameterOptimizer()
        self.feature_engineer = FeatureEngineer()
        
        # Performance tracking
        self.performance_metrics = {}
        self.models_trained = 0
        self.models_meeting_target = 0
        
        logger.info("Production Model Trainer initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def load_real_data(self, 
                           seasons: List[str],
                           min_games: int = 20,
                           min_minutes: float = 10.0) -> pd.DataFrame:
        """
        Load real NBA data from database
        
        Args:
            seasons: List of seasons to load
            min_games: Minimum games per player
            min_minutes: Minimum minutes per game
            
        Returns:
            DataFrame with real NBA data
        """
        logger.info(f"Loading real NBA data for seasons: {seasons}")
        
        if not self.db_manager:
            self.db_manager = init_db()
        
        with self.db_manager.get_db(read_only=True) as session:
            # Build query for player game logs
            query = session.query(
                PlayerGameLog,
                Player.full_name.label('player_name'),
                Player.nba_player_id,
                Team.abbreviation.label('team_abbr'),
                Team.full_name.label('team_name')
            ).join(
                Player, PlayerGameLog.player_id == Player.id
            ).join(
                Team, PlayerGameLog.team_id == Team.id
            ).join(
                Game, PlayerGameLog.game_id == Game.id
            )
            
            # Filter by seasons
            if seasons:
                query = query.filter(Game.season.in_(seasons))
            
            # Filter by minimum minutes
            query = query.filter(PlayerGameLog.minutes_played >= min_minutes)
            
            # Execute query
            results = query.all()
            
            if not results:
                logger.warning(f"No data found for seasons {seasons}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data_records = []
            for log, player_name, nba_player_id, team_abbr, team_name in results:
                record = {
                    'player_id': nba_player_id,
                    'player_name': player_name,
                    'game_id': log.game_id,
                    'game_date': log.game_date,
                    'team': team_abbr,
                    'team_name': team_name,
                    'home_game': log.home_game,
                    
                    # Basic stats
                    'MIN': log.minutes_played,
                    'PTS': log.points,
                    'REB': log.rebounds,
                    'OREB': log.offensive_rebounds,
                    'DREB': log.defensive_rebounds,
                    'AST': log.assists,
                    'STL': log.steals,
                    'BLK': log.blocks,
                    'TOV': log.turnovers,
                    'PF': log.personal_fouls,
                    
                    # Shooting stats
                    'FGM': log.field_goals_made,
                    'FGA': log.field_goals_attempted,
                    'FG_PCT': log.field_goal_percentage,
                    'FG3M': log.three_pointers_made,
                    'FG3A': log.three_pointers_attempted,
                    'FG3_PCT': log.three_point_percentage,
                    'FTM': log.free_throws_made,
                    'FTA': log.free_throws_attempted,
                    'FT_PCT': log.free_throw_percentage,
                    
                    # Advanced
                    'PLUS_MINUS': log.plus_minus
                }
                data_records.append(record)
            
            df = pd.DataFrame(data_records)
            
            # Filter players with minimum games
            player_games = df.groupby('player_id').size()
            valid_players = player_games[player_games >= min_games].index
            df = df[df['player_id'].isin(valid_players)]
            
            # Sort by date
            df['game_date'] = pd.to_datetime(df['game_date'])
            df = df.sort_values(['player_id', 'game_date'])
            
            logger.info(f"Loaded {len(df)} game logs for {df['player_id'].nunique()} players")
            
            return df
    
    async def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer advanced features for production models
        
        Args:
            df: Raw game log data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering advanced features...")
        
        # Calculate rolling averages
        rolling_windows = self.config['training']['features']['engineering']['rolling_windows']
        
        stat_columns = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FT_PCT', 'MIN']
        
        for window in rolling_windows:
            for col in stat_columns:
                if col in df.columns:
                    # Rolling mean
                    df[f'{col}_avg_{window}'] = df.groupby('player_id')[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )
                    
                    # Rolling std
                    df[f'{col}_std_{window}'] = df.groupby('player_id')[col].transform(
                        lambda x: x.rolling(window, min_periods=1).std().shift(1)
                    )
        
        # Calculate days of rest
        df['days_rest'] = df.groupby('player_id')['game_date'].diff().dt.days.fillna(3)
        df['days_rest'] = df['days_rest'].clip(0, 7)  # Cap at 7 days
        
        # Back-to-back games
        df['back_to_back'] = (df['days_rest'] <= 1).astype(int)
        
        # Season progress (0 to 1)
        df['season_progress'] = df.groupby('player_id').cumcount() / df.groupby('player_id')['player_id'].transform('count')
        
        # Opponent strength (simplified - would use team stats in production)
        df['is_strong_opponent'] = 0  # Placeholder
        
        # Player consistency metrics
        for col in ['PTS', 'REB', 'AST']:
            if col in df.columns:
                df[f'{col}_consistency'] = df.groupby('player_id')[col].transform(
                    lambda x: 1 / (x.rolling(10, min_periods=5).std() + 1)
                ).shift(1)
        
        # Recent form (last 5 games vs season average)
        for col in ['PTS', 'REB', 'AST']:
            if col in df.columns:
                season_avg = df.groupby('player_id')[col].transform('mean')
                recent_avg = df.groupby('player_id')[col].transform(
                    lambda x: x.rolling(5, min_periods=3).mean()
                ).shift(1)
                df[f'{col}_form'] = (recent_avg / (season_avg + 1)) - 1
        
        # Advanced efficiency metrics
        df['true_shooting_pct'] = df.apply(
            lambda x: x['PTS'] / (2 * (x['FGA'] + 0.44 * x['FTA'])) if x['FGA'] + x['FTA'] > 0 else 0,
            axis=1
        )
        
        df['usage_rate'] = df.apply(
            lambda x: 100 * ((x['FGA'] + 0.44 * x['FTA'] + x['TOV']) * 48) / (x['MIN'] * 5) if x['MIN'] > 0 else 0,
            axis=1
        )
        
        # Fill missing values
        df = df.fillna(method='ffill').fillna(0)
        
        # Remove first N games per player (not enough history)
        min_games_for_features = max(rolling_windows) + 1
        df = df.groupby('player_id').apply(
            lambda x: x.iloc[min_games_for_features:] if len(x) > min_games_for_features else pd.DataFrame()
        ).reset_index(drop=True)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        
        return df
    
    async def train_target_model(self,
                               df: pd.DataFrame,
                               target: str,
                               test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train production model for a specific target
        
        Args:
            df: Feature-engineered DataFrame
            target: Target variable (PTS, REB, AST, etc.)
            test_size: Test set size
            
        Returns:
            Dictionary with model performance and metadata
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {target} Prediction Model")
        logger.info(f"{'='*60}")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in [
            'player_id', 'player_name', 'game_id', 'game_date', 'team', 'team_name',
            'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'FTM'
        ]]
        
        # For targets other than PTS, REB, AST, keep those as features
        if target != 'PTS':
            feature_cols = [col for col in feature_cols if 'PTS' not in col or 'avg' in col or 'std' in col]
        if target != 'REB':
            feature_cols = [col for col in feature_cols if 'REB' not in col or 'avg' in col or 'std' in col]
        if target != 'AST':
            feature_cols = [col for col in feature_cols if 'AST' not in col or 'avg' in col or 'std' in col]
        
        X = df[feature_cols]
        y = df[target]
        
        # Remove outliers (games with 0 minutes or extreme values)
        valid_idx = (df['MIN'] > 5) & (y < y.quantile(0.999)) & (y > 0)
        X = X[valid_idx]
        y = y[valid_idx]
        game_dates = df[valid_idx]['game_date']
        
        logger.info(f"Training with {len(X)} samples, {len(feature_cols)} features")
        
        # Time-based train/test split
        split_date = game_dates.quantile(1 - test_size)
        train_idx = game_dates < split_date
        test_idx = ~train_idx
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Validate no data leakage
        train_dates = game_dates[train_idx]
        test_dates = game_dates[test_idx]
        self.time_validator.validate_no_future_leakage(train_dates.values, test_dates.values)
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame for feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        # Further split train into train/val for hyperparameter tuning
        val_split_date = train_dates.quantile(0.85)
        val_idx = train_dates >= val_split_date
        train_idx_final = train_dates < val_split_date
        
        X_train_final = X_train_scaled[train_idx_final]
        X_val = X_train_scaled[val_idx]
        y_train_final = y_train[train_idx_final]
        y_val = y_train[val_idx]
        
        # Initialize MLflow
        mlflow.set_experiment(f"nba_{target.lower()}_production")
        
        best_model = None
        best_score = -np.inf
        best_params = {}
        model_results = {}
        
        with mlflow.start_run(run_name=f"{target}_ensemble_{datetime.now():%Y%m%d_%H%M%S}"):
            mlflow.log_param("target", target)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("n_train_samples", len(X_train_final))
            mlflow.log_param("n_val_samples", len(X_val))
            mlflow.log_param("n_test_samples", len(X_test))
            
            # Train multiple models
            models_to_train = []
            if self.config['training']['models']['random_forest']['enabled']:
                models_to_train.append('random_forest')
            if self.config['training']['models']['xgboost']['enabled']:
                models_to_train.append('xgboost')
            if self.config['training']['models']['lightgbm']['enabled']:
                models_to_train.append('lightgbm')
            
            for model_type in models_to_train:
                logger.info(f"\nTraining {model_type}...")
                
                # Hyperparameter optimization
                if self.config['training']['models'][model_type].get('hyperparameter_tuning', True):
                    n_trials = self.config['training']['models'][model_type].get('n_trials', 50)
                    
                    best_params_model = await self.optimizer.optimize(
                        model_type=model_type,
                        X_train=X_train_final,
                        y_train=y_train_final,
                        X_val=X_val,
                        y_val=y_val,
                        n_trials=n_trials,
                        target_name=target
                    )
                else:
                    best_params_model = self.optimizer._get_default_params(model_type)
                
                # Train final model with best params
                if model_type == 'random_forest':
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(**best_params_model, random_state=42, n_jobs=-1)
                elif model_type == 'xgboost':
                    import xgboost as xgb
                    model = xgb.XGBRegressor(**best_params_model, random_state=42, n_jobs=-1)
                elif model_type == 'lightgbm':
                    import lightgbm as lgb
                    model = lgb.LGBMRegressor(**best_params_model, random_state=42, n_jobs=-1)
                
                # Train on full training set
                model.fit(X_train_scaled, y_train)
                
                # Evaluate on test set
                y_pred = model.predict(X_test_scaled)
                
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                model_results[model_type] = {
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'params': best_params_model
                }
                
                logger.info(f"{model_type} Results:")
                logger.info(f"  R² Score: {r2:.4f}")
                logger.info(f"  MAE: {mae:.2f}")
                logger.info(f"  RMSE: {rmse:.2f}")
                logger.info(f"  MAPE: {mape:.1f}%")
                
                # Log to MLflow
                mlflow.log_metrics({
                    f"{model_type}_r2": r2,
                    f"{model_type}_mae": mae,
                    f"{model_type}_rmse": rmse,
                    f"{model_type}_mape": mape
                })
                
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_params = best_params_model
                    best_model_type = model_type
            
            # Ensemble models if multiple were trained
            if len(model_results) > 1 and self.config['training']['ensemble']['use_stacking']:
                logger.info("\nCreating ensemble model...")
                
                from sklearn.ensemble import VotingRegressor
                from sklearn.linear_model import BayesianRidge
                
                # Create ensemble
                estimators = []
                for model_type in models_to_train:
                    if model_type == 'random_forest':
                        model = RandomForestRegressor(**model_results[model_type]['params'], random_state=42, n_jobs=-1)
                    elif model_type == 'xgboost':
                        model = xgb.XGBRegressor(**model_results[model_type]['params'], random_state=42, n_jobs=-1)
                    elif model_type == 'lightgbm':
                        model = lgb.LGBMRegressor(**model_results[model_type]['params'], random_state=42, n_jobs=-1)
                    
                    model.fit(X_train_scaled, y_train)
                    estimators.append((model_type, model))
                
                ensemble = VotingRegressor(estimators=estimators)
                ensemble.fit(X_train_scaled, y_train)
                
                # Evaluate ensemble
                y_pred_ensemble = ensemble.predict(X_test_scaled)
                
                r2_ensemble = r2_score(y_test, y_pred_ensemble)
                mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
                rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
                mape_ensemble = np.mean(np.abs((y_test - y_pred_ensemble) / y_test)) * 100
                
                logger.info(f"\nEnsemble Results:")
                logger.info(f"  R² Score: {r2_ensemble:.4f}")
                logger.info(f"  MAE: {mae_ensemble:.2f}")
                logger.info(f"  RMSE: {rmse_ensemble:.2f}")
                logger.info(f"  MAPE: {mape_ensemble:.1f}%")
                
                mlflow.log_metrics({
                    "ensemble_r2": r2_ensemble,
                    "ensemble_mae": mae_ensemble,
                    "ensemble_rmse": rmse_ensemble,
                    "ensemble_mape": mape_ensemble
                })
                
                if r2_ensemble > best_score:
                    best_model = ensemble
                    best_score = r2_ensemble
                    best_model_type = 'ensemble'
            
            # Save best model
            model_dir = Path("models") / target.lower()
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / f"model_{datetime.now():%Y%m%d_%H%M%S}.pkl"
            scaler_path = model_dir / f"scaler_{datetime.now():%Y%m%d_%H%M%S}.pkl"
            
            joblib.dump(best_model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Log artifacts
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(scaler_path))
            
            # Log feature importance if available
            if hasattr(best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_path = model_dir / f"feature_importance_{datetime.now():%Y%m%d_%H%M%S}.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))
                
                logger.info(f"\nTop 10 Features:")
                for idx, row in importance_df.head(10).iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            # Time series validation
            logger.info("\nPerforming time series cross-validation...")
            cv_results = self.time_validator.time_series_cross_validation(
                X=X_train_scaled,
                y=y_train,
                model=best_model,
                dates=train_dates
            )
            
            mlflow.log_metrics({
                "cv_mean_r2": cv_results['mean_r2'],
                "cv_std_r2": cv_results['std_r2'],
                "cv_mean_mae": cv_results['mean_mae'],
                "cv_std_mae": cv_results['std_mae']
            })
            
            # Check if meets performance targets
            meets_target = (
                best_score >= self.config['training']['performance_targets']['min_r2_score'] and
                mae_ensemble <= self.config['training']['performance_targets']['max_mae'] if 'mae_ensemble' in locals() else True
            )
            
            if meets_target:
                self.models_meeting_target += 1
                logger.info(f"\n✓ {target} model MEETS performance targets!")
            else:
                logger.info(f"\n✗ {target} model does not meet performance targets")
            
            self.models_trained += 1
            
            # Store performance metrics
            self.performance_metrics[target] = {
                'best_model_type': best_model_type,
                'r2_score': best_score,
                'mae': mae_ensemble if 'mae_ensemble' in locals() else model_results[best_model_type]['mae'],
                'rmse': rmse_ensemble if 'rmse_ensemble' in locals() else model_results[best_model_type]['rmse'],
                'mape': mape_ensemble if 'mape_ensemble' in locals() else model_results[best_model_type]['mape'],
                'cv_mean_r2': cv_results['mean_r2'],
                'cv_std_r2': cv_results['std_r2'],
                'meets_target': meets_target,
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'n_features': len(feature_cols),
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test)
            }
            
            mlflow.log_param("meets_target", meets_target)
            mlflow.log_param("model_path", str(model_path))
            
        return self.performance_metrics[target]
    
    async def train_all_models(self,
                             targets: List[str],
                             seasons: List[str]) -> Dict[str, Any]:
        """
        Train models for all targets
        
        Args:
            targets: List of targets to train
            seasons: List of seasons to use
            
        Returns:
            Dictionary with all model results
        """
        logger.info("="*60)
        logger.info("PRODUCTION NBA MODEL TRAINING")
        logger.info("="*60)
        logger.info(f"Targets: {targets}")
        logger.info(f"Seasons: {seasons}")
        
        # Load real data
        df = await self.load_real_data(
            seasons=seasons,
            min_games=self.config['training']['data']['min_games_per_player'],
            min_minutes=self.config['training']['data']['min_minutes_per_game']
        )
        
        if df.empty:
            logger.error("No data loaded. Please run data collection first.")
            return {}
        
        # Validate data quality
        logger.info("\nValidating data quality...")
        validation_result = self.validator.validate_player_game_log(df)
        logger.info(f"Validation Status: {validation_result.status.value}")
        
        if validation_result.status.value == "FAILED":
            logger.error("Data validation failed. Please check data quality.")
            return {}
        
        # Engineer features
        df = await self.engineer_features(df)
        
        # Train models for each target
        results = {}
        for target in targets:
            if target in df.columns:
                result = await self.train_target_model(df, target)
                results[target] = result
            else:
                logger.warning(f"Target {target} not found in data")
        
        # Generate final report
        self.generate_production_report(results)
        
        return results
    
    def generate_production_report(self, results: Dict[str, Any]):
        """Generate comprehensive production training report"""
        
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_report = {
            "timestamp": timestamp,
            "models_trained": self.models_trained,
            "models_meeting_target": self.models_meeting_target,
            "target_accuracy": self.config['training']['performance_targets']['min_r2_score'],
            "results": results,
            "summary": {
                "average_r2": np.mean([r['r2_score'] for r in results.values()]) if results else 0,
                "average_mae": np.mean([r['mae'] for r in results.values()]) if results else 0,
                "best_model": max(results.items(), key=lambda x: x[1]['r2_score']) if results else None
            }
        }
        
        json_path = report_dir / f"production_training_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        # Text report
        text_path = report_dir / f"production_training_{timestamp}.txt"
        with open(text_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("NBA PRODUCTION MODEL TRAINING REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Models Trained: {self.models_trained}\n")
            f.write(f"Models Meeting Target (≥{self.config['training']['performance_targets']['min_r2_score']} R²): {self.models_meeting_target}\n\n")
            
            f.write("INDIVIDUAL MODEL RESULTS\n")
            f.write("-"*60 + "\n\n")
            
            for target, result in results.items():
                f.write(f"{target} PREDICTION MODEL\n")
                f.write(f"  Model Type: {result['best_model_type']}\n")
                f.write(f"  R² Score: {result['r2_score']:.4f} {'✓' if result['meets_target'] else '✗'}\n")
                f.write(f"  MAE: {result['mae']:.2f}\n")
                f.write(f"  RMSE: {result['rmse']:.2f}\n")
                f.write(f"  MAPE: {result['mape']:.1f}%\n")
                f.write(f"  CV Mean R²: {result['cv_mean_r2']:.4f} ± {result['cv_std_r2']:.4f}\n")
                f.write(f"  Features: {result['n_features']}\n")
                f.write(f"  Training Samples: {result['n_train_samples']:,}\n")
                f.write(f"  Test Samples: {result['n_test_samples']:,}\n")
                f.write(f"  Model Path: {result['model_path']}\n\n")
            
            if json_report['summary']['best_model']:
                best_target, best_result = json_report['summary']['best_model']
                f.write("="*60 + "\n")
                f.write(f"BEST MODEL: {best_target} (R² = {best_result['r2_score']:.4f})\n")
                f.write("="*60 + "\n")
        
        logger.info(f"\nReports saved to:")
        logger.info(f"  - {json_path}")
        logger.info(f"  - {text_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Models Trained: {self.models_trained}")
        logger.info(f"Models Meeting Target: {self.models_meeting_target}/{self.models_trained}")
        
        if self.models_meeting_target > 0:
            logger.info(f"\n✓ SUCCESS: {self.models_meeting_target} model(s) achieved ≥{self.config['training']['performance_targets']['min_r2_score']} R² score!")
        else:
            logger.info(f"\n⚠ WARNING: No models met the {self.config['training']['performance_targets']['min_r2_score']} R² target")


async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train production NBA prediction models')
    parser.add_argument('--targets', nargs='+', default=['PTS', 'REB', 'AST'],
                       help='Targets to train models for')
    parser.add_argument('--seasons', nargs='+', default=['2023-24', '2024-25'],
                       help='Seasons to use for training')
    parser.add_argument('--config', default='config/training.yaml',
                       help='Path to training configuration')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate data without training')
    
    args = parser.parse_args()
    
    # Create necessary directories
    for dir_name in ['logs', 'reports', 'models', 'mlruns']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Initialize trainer
    trainer = ProductionModelTrainer(config_path=args.config)
    
    if args.validate_only:
        # Load and validate data only
        logger.info("Running data validation only...")
        df = await trainer.load_real_data(args.seasons)
        
        if not df.empty:
            validation_result = trainer.validator.validate_player_game_log(df)
            logger.info(f"Validation Status: {validation_result.status.value}")
            logger.info(f"Passed Checks: {validation_result.passed_checks}")
            logger.info(f"Failed Checks: {validation_result.failed_checks}")
            logger.info(f"Warning Checks: {validation_result.warning_checks}")
            
            if validation_result.errors:
                logger.error("Validation Errors:")
                for error in validation_result.errors[:10]:
                    logger.error(f"  - {error}")
        else:
            logger.error("No data found to validate")
        
        return
    
    # Train models
    results = await trainer.train_all_models(
        targets=args.targets,
        seasons=args.seasons
    )
    
    if results:
        logger.info("\n✓ Production model training completed successfully!")
        logger.info(f"  Models trained: {len(results)}")
        logger.info(f"  Average R² score: {np.mean([r['r2_score'] for r in results.values()]):.4f}")
    else:
        logger.error("\n✗ Production model training failed")


if __name__ == "__main__":
    asyncio.run(main())