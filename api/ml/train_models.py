"""
Model training script for NBA player predictions
"""
import logging
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from typing import Dict, List, Tuple, Any

from sqlalchemy import select, and_
from database.connection import get_db_session
from api.models.game_data import GameLog, Player
from api.features.player_features import FeatureEngineer

logger = logging.getLogger(__name__)


class NBAModelTrainer:
    """Train models for NBA player predictions"""
    
    def __init__(self, model_save_path: str = "./models"):
        self.model_save_path = model_save_path
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.scalers = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(model_save_path, exist_ok=True)
    
    async def prepare_training_data(
        self, 
        seasons: List[str] = None,
        min_games_per_player: int = 20
    ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Prepare training data from historical game logs
        
        Args:
            seasons: List of seasons to use for training
            min_games_per_player: Minimum games required per player
            
        Returns:
            Tuple of (features_df, targets_dict)
        """
        if seasons is None:
            seasons = ['2022-23', '2023-24']
        
        logger.info(f"Preparing training data for seasons: {seasons}")
        
        async with get_db_session() as session:
            # Get all game logs for specified seasons
            result = await session.execute(
                select(GameLog)
                .where(GameLog.season.in_(seasons))
                .order_by(GameLog.player_id, GameLog.game_date)
            )
            game_logs = result.scalars().all()
            
            if not game_logs:
                raise ValueError("No game logs found for training")
            
            logger.info(f"Found {len(game_logs)} game logs")
            
            # Convert to DataFrame
            games_df = pd.DataFrame([g.__dict__ for g in game_logs])
            games_df = games_df.drop('_sa_instance_state', axis=1, errors='ignore')
            
            # Group by player
            feature_rows = []
            target_points = []
            target_rebounds = []
            target_assists = []
            
            for player_id, player_games in games_df.groupby('player_id'):
                if len(player_games) < min_games_per_player:
                    continue
                
                player_games = player_games.sort_values('game_date')
                
                # For each game, calculate features using games before it
                for i in range(20, len(player_games)):  # Start from game 20 to have enough history
                    current_game = player_games.iloc[i]
                    historical_games = player_games.iloc[:i]
                    
                    # Calculate features manually (simplified version)
                    features = {
                        'pts_last_5': historical_games.tail(5)['points'].mean(),
                        'pts_last_10': historical_games.tail(10)['points'].mean(),
                        'pts_last_20': historical_games.tail(20)['points'].mean(),
                        'pts_season': historical_games['points'].mean(),
                        
                        'reb_last_5': historical_games.tail(5)['rebounds'].mean(),
                        'reb_last_10': historical_games.tail(10)['rebounds'].mean(),
                        'reb_last_20': historical_games.tail(20)['rebounds'].mean(),
                        'reb_season': historical_games['rebounds'].mean(),
                        
                        'ast_last_5': historical_games.tail(5)['assists'].mean(),
                        'ast_last_10': historical_games.tail(10)['assists'].mean(),
                        'ast_last_20': historical_games.tail(20)['assists'].mean(),
                        'ast_season': historical_games['assists'].mean(),
                        
                        'fg_pct_last_10': historical_games.tail(10)['field_goal_pct'].mean(),
                        'ft_pct_last_10': historical_games.tail(10)['free_throw_pct'].mean(),
                        'fg3_pct_last_10': historical_games.tail(10)['three_point_pct'].mean(),
                        
                        'minutes_last_10': historical_games.tail(10)['minutes_played'].mean(),
                        'games_played_season': len(historical_games),
                        'days_since_last_game': (current_game['game_date'] - historical_games.iloc[-1]['game_date']).days,
                        'back_to_backs_last_10': self._count_back_to_backs(historical_games.tail(10)),
                        
                        'home_ppg': historical_games[historical_games['is_home'] == True]['points'].mean(),
                        'away_ppg': historical_games[historical_games['is_home'] == False]['points'].mean(),
                        
                        # Simplified opponent features
                        'vs_opponent_avg_pts': historical_games[historical_games['opponent_id'] == current_game['opponent_id']]['points'].mean() 
                                              if len(historical_games[historical_games['opponent_id'] == current_game['opponent_id']]) > 0 
                                              else historical_games['points'].mean()
                    }
                    
                    feature_rows.append(features)
                    target_points.append(current_game['points'])
                    target_rebounds.append(current_game['rebounds'])
                    target_assists.append(current_game['assists'])
            
            # Create feature DataFrame
            features_df = pd.DataFrame(feature_rows)
            
            # Handle any NaN values
            features_df = features_df.fillna(features_df.mean())
            
            targets = {
                'points': pd.Series(target_points),
                'rebounds': pd.Series(target_rebounds),
                'assists': pd.Series(target_assists)
            }
            
            logger.info(f"Prepared {len(features_df)} training samples")
            
            return features_df, targets
    
    def _count_back_to_backs(self, games_df: pd.DataFrame) -> int:
        """Count back-to-back games in a DataFrame"""
        if len(games_df) < 2:
            return 0
        
        count = 0
        for i in range(1, len(games_df)):
            if (games_df.iloc[i]['game_date'] - games_df.iloc[i-1]['game_date']).days == 1:
                count += 1
        return count
    
    def train_models(
        self, 
        features_df: pd.DataFrame, 
        targets: Dict[str, pd.Series],
        test_size: float = 0.2
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models for each target
        
        Args:
            features_df: Feature DataFrame
            targets: Dictionary of target variables
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary of trained models and metrics
        """
        results = {}
        
        for target_name, target_values in targets.items():
            logger.info(f"Training models for {target_name}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, target_values, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler
            self.scalers[target_name] = scaler
            
            # Train multiple models
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ),
                'ridge': Ridge(alpha=1.0, random_state=42),
                'lasso': Lasso(alpha=0.1, random_state=42)
            }
            
            model_results = {}
            
            for model_name, model in models.items():
                logger.info(f"  Training {model_name}...")
                
                # Use scaled data for linear models
                if model_name in ['ridge', 'lasso']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, 
                    X_train_scaled if model_name in ['ridge', 'lasso'] else X_train, 
                    y_train, 
                    cv=5, 
                    scoring='neg_mean_absolute_error'
                )
                cv_mae = -cv_scores.mean()
                
                model_results[model_name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_mae': cv_mae
                }
                
                logger.info(f"    {model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
            
            # Select best model based on MAE
            best_model_name = min(model_results, key=lambda x: model_results[x]['mae'])
            best_model = model_results[best_model_name]
            
            logger.info(f"  Best model for {target_name}: {best_model_name}")
            
            results[target_name] = {
                'best_model': best_model['model'],
                'best_model_name': best_model_name,
                'all_models': model_results,
                'feature_importance': self._get_feature_importance(
                    best_model['model'], 
                    features_df.columns
                ) if hasattr(best_model['model'], 'feature_importances_') else None
            }
            
            # Store the best model
            self.models[target_name] = best_model['model']
        
        return results
    
    def _get_feature_importance(self, model, feature_names) -> List[Tuple[str, float]]:
        """Get feature importance from tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp = list(zip(feature_names, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            return feature_imp[:10]  # Top 10 features
        return None
    
    def save_models(self, version: str = None):
        """Save trained models to disk"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for target_name, model in self.models.items():
            # Save model
            model_path = os.path.join(self.model_save_path, f"rf_{target_name}_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Saved model to {model_path}")
            
            # Save scaler
            if target_name in self.scalers:
                scaler_path = os.path.join(self.model_save_path, f"scaler_{target_name}.pkl")
                joblib.dump(self.scalers[target_name], scaler_path)
                logger.info(f"Saved scaler to {scaler_path}")
        
        # Save metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'targets': list(self.models.keys()),
            'feature_names': list(self.feature_engineer.feature_names)
        }
        
        import json
        metadata_path = os.path.join(self.model_save_path, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model training complete. Version: {version}")


async def main():
    """Main training pipeline"""
    trainer = NBAModelTrainer()
    
    # Prepare training data
    logger.info("Starting model training pipeline...")
    features_df, targets = await trainer.prepare_training_data(
        seasons=['2022-23', '2023-24'],
        min_games_per_player=30
    )
    
    # Train models
    results = trainer.train_models(features_df, targets)
    
    # Print results summary
    print("\n=== Model Training Results ===")
    for target, result in results.items():
        print(f"\n{target.upper()}:")
        print(f"  Best Model: {result['best_model_name']}")
        best_metrics = result['all_models'][result['best_model_name']]
        print(f"  MAE: {best_metrics['mae']:.2f}")
        print(f"  RMSE: {best_metrics['rmse']:.2f}")
        print(f"  R²: {best_metrics['r2']:.3f}")
        
        if result['feature_importance']:
            print("  Top Features:")
            for feat, imp in result['feature_importance'][:5]:
                print(f"    - {feat}: {imp:.3f}")
    
    # Save models
    trainer.save_models()
    
    print("\nModel training complete!")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())