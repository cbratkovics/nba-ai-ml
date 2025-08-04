"""
Sports-specific validation strategies for NBA ML models
Ensures proper time series validation and no data leakage
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TimeSeriesValidator:
    """
    Sports-specific validation strategies
    Handles time-based splits and prevents data leakage
    """
    
    def __init__(self, n_splits: int = 5):
        """
        Initialize validator
        
        Args:
            n_splits: Number of time series splits for cross-validation
        """
        self.n_splits = n_splits
    
    def validate_no_future_leakage(self, 
                                  train_dates: np.ndarray,
                                  test_dates: np.ndarray) -> bool:
        """
        Ensure training data doesn't include future games
        
        Args:
            train_dates: Training set dates
            test_dates: Test set dates
            
        Returns:
            True if no leakage detected
            
        Raises:
            ValueError: If future leakage is detected
        """
        # Convert to pandas datetime if needed
        if not isinstance(train_dates[0], pd.Timestamp):
            train_dates = pd.to_datetime(train_dates)
        if not isinstance(test_dates[0], pd.Timestamp):
            test_dates = pd.to_datetime(test_dates)
        
        max_train_date = train_dates.max()
        min_test_date = test_dates.min()
        
        if max_train_date >= min_test_date:
            overlap_days = (max_train_date - min_test_date).days
            raise ValueError(
                f"Data leakage detected! Training data includes games {overlap_days} days "
                f"after test data starts. Max train date: {max_train_date}, "
                f"Min test date: {min_test_date}"
            )
        
        logger.info(f"No data leakage: {(min_test_date - max_train_date).days} days gap between sets")
        return True
    
    def validate_player_coverage(self, 
                                train_players: List[str],
                                test_players: List[str],
                                min_coverage: float = 0.8) -> Dict[str, Any]:
        """
        Ensure model has seen most players during training
        
        Args:
            train_players: List of player IDs in training set
            test_players: List of player IDs in test set
            min_coverage: Minimum required coverage ratio
            
        Returns:
            Dictionary with coverage statistics
        """
        train_players_set = set(train_players)
        test_players_set = set(test_players)
        
        # Calculate coverage
        common_players = train_players_set.intersection(test_players_set)
        coverage_ratio = len(common_players) / len(test_players_set) if test_players_set else 0
        
        # Find unseen players
        unseen_players = test_players_set - train_players_set
        
        result = {
            "coverage_ratio": coverage_ratio,
            "n_train_players": len(train_players_set),
            "n_test_players": len(test_players_set),
            "n_common_players": len(common_players),
            "n_unseen_players": len(unseen_players),
            "unseen_players": list(unseen_players)[:10],  # First 10 unseen players
            "meets_minimum": coverage_ratio >= min_coverage
        }
        
        if coverage_ratio < min_coverage:
            logger.warning(
                f"Low player coverage: {coverage_ratio:.1%} "
                f"({len(unseen_players)} unseen players in test set)"
            )
        else:
            logger.info(f"Good player coverage: {coverage_ratio:.1%}")
        
        return result
    
    def backtesting_evaluation(self, 
                              model,
                              historical_data: pd.DataFrame,
                              target_col: str,
                              feature_cols: List[str],
                              n_seasons: int = 2) -> Dict[str, Any]:
        """
        Walk-forward analysis on historical seasons
        
        Args:
            model: Trained model to evaluate
            historical_data: Historical game data
            target_col: Target column name
            feature_cols: List of feature column names
            n_seasons: Number of seasons for backtesting
            
        Returns:
            Dictionary with backtesting results
        """
        # Sort by date
        historical_data = historical_data.sort_values('GAME_DATE')
        
        # Get unique seasons
        historical_data['season'] = pd.to_datetime(historical_data['GAME_DATE']).dt.year
        seasons = sorted(historical_data['season'].unique())
        
        if len(seasons) < n_seasons + 1:
            logger.warning(f"Not enough seasons for {n_seasons} season backtesting")
            n_seasons = len(seasons) - 1
        
        backtest_results = []
        
        for i in range(n_seasons):
            # Define train and test seasons
            test_season = seasons[-(i+1)]
            train_seasons = seasons[:-(i+1)]
            
            if not train_seasons:
                continue
            
            # Split data
            train_data = historical_data[historical_data['season'].isin(train_seasons)]
            test_data = historical_data[historical_data['season'] == test_season]
            
            if len(train_data) < 100 or len(test_data) < 50:
                logger.warning(f"Insufficient data for season {test_season}")
                continue
            
            # Prepare features
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]
            
            # Train model
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model_copy.predict(X_test)
            
            season_results = {
                "season": test_season,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "r2_score": r2_score(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            backtest_results.append(season_results)
            
            logger.info(
                f"Season {test_season} backtest: "
                f"R²={season_results['r2_score']:.4f}, "
                f"MAE={season_results['mae']:.2f}"
            )
        
        # Calculate summary statistics
        if backtest_results:
            r2_scores = [r['r2_score'] for r in backtest_results]
            mae_scores = [r['mae'] for r in backtest_results]
            
            summary = {
                "n_seasons_tested": len(backtest_results),
                "mean_r2": np.mean(r2_scores),
                "std_r2": np.std(r2_scores),
                "min_r2": np.min(r2_scores),
                "max_r2": np.max(r2_scores),
                "mean_mae": np.mean(mae_scores),
                "std_mae": np.std(mae_scores),
                "season_results": backtest_results
            }
        else:
            summary = {
                "n_seasons_tested": 0,
                "season_results": []
            }
        
        return summary
    
    def time_series_cross_validation(self,
                                    X: pd.DataFrame,
                                    y: pd.Series,
                                    model,
                                    dates: pd.Series = None) -> Dict[str, Any]:
        """
        Perform time series cross-validation
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model: Model to validate
            dates: Date series for temporal ordering
            
        Returns:
            Dictionary with cross-validation results
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        cv_scores = {
            'r2': [],
            'mae': [],
            'rmse': [],
            'train_size': [],
            'test_size': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Check temporal ordering if dates provided
            if dates is not None:
                train_dates = dates.iloc[train_idx]
                test_dates = dates.iloc[test_idx]
                
                try:
                    self.validate_no_future_leakage(train_dates.values, test_dates.values)
                except ValueError as e:
                    logger.error(f"Fold {fold} has data leakage: {e}")
                    continue
            
            # Train model
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_train, y_train)
            
            # Predict
            y_pred = model_copy.predict(X_test)
            
            # Calculate metrics
            cv_scores['r2'].append(r2_score(y_test, y_pred))
            cv_scores['mae'].append(mean_absolute_error(y_test, y_pred))
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            cv_scores['train_size'].append(len(X_train))
            cv_scores['test_size'].append(len(X_test))
            
            logger.info(
                f"Fold {fold}: R²={cv_scores['r2'][-1]:.4f}, "
                f"MAE={cv_scores['mae'][-1]:.2f}, "
                f"Train={len(X_train)}, Test={len(X_test)}"
            )
        
        # Calculate summary
        summary = {
            'n_folds': len(cv_scores['r2']),
            'mean_r2': np.mean(cv_scores['r2']),
            'std_r2': np.std(cv_scores['r2']),
            'mean_mae': np.mean(cv_scores['mae']),
            'std_mae': np.std(cv_scores['mae']),
            'mean_rmse': np.mean(cv_scores['rmse']),
            'std_rmse': np.std(cv_scores['rmse']),
            'fold_scores': cv_scores
        }
        
        logger.info(
            f"CV Summary: R²={summary['mean_r2']:.4f}±{summary['std_r2']:.4f}, "
            f"MAE={summary['mean_mae']:.2f}±{summary['std_mae']:.2f}"
        )
        
        return summary
    
    def validate_prediction_stability(self,
                                    model,
                                    X: pd.DataFrame,
                                    n_runs: int = 10) -> Dict[str, Any]:
        """
        Check prediction stability with different random seeds
        
        Args:
            model: Model to test
            X: Features to predict on
            n_runs: Number of runs with different seeds
            
        Returns:
            Dictionary with stability metrics
        """
        predictions = []
        
        for seed in range(n_runs):
            # Set random seed if model supports it
            if hasattr(model, 'random_state'):
                model_copy = model.__class__(**model.get_params())
                model_copy.random_state = seed
            else:
                model_copy = model
            
            # Make predictions
            y_pred = model_copy.predict(X)
            predictions.append(y_pred)
        
        # Convert to array
        predictions = np.array(predictions)
        
        # Calculate stability metrics
        mean_preds = np.mean(predictions, axis=0)
        std_preds = np.std(predictions, axis=0)
        cv_preds = std_preds / (mean_preds + 1e-10)  # Coefficient of variation
        
        stability_metrics = {
            'mean_std': np.mean(std_preds),
            'max_std': np.max(std_preds),
            'mean_cv': np.mean(cv_preds),
            'max_cv': np.max(cv_preds),
            'unstable_predictions': np.sum(cv_preds > 0.1),  # CV > 10%
            'n_samples': len(X),
            'n_runs': n_runs
        }
        
        if stability_metrics['mean_cv'] > 0.05:
            logger.warning(f"High prediction variability: mean CV = {stability_metrics['mean_cv']:.2%}")
        else:
            logger.info(f"Good prediction stability: mean CV = {stability_metrics['mean_cv']:.2%}")
        
        return stability_metrics
    
    def validate_seasonality_handling(self,
                                    model,
                                    data: pd.DataFrame,
                                    target_col: str,
                                    feature_cols: List[str]) -> Dict[str, Any]:
        """
        Check if model handles different parts of season well
        
        Args:
            model: Trained model
            data: Game data with dates
            target_col: Target column
            feature_cols: Feature columns
            
        Returns:
            Dictionary with seasonality analysis
        """
        # Add month column
        data['month'] = pd.to_datetime(data['GAME_DATE']).dt.month
        
        # Define season periods
        periods = {
            'early_season': [10, 11],  # Oct-Nov
            'mid_season': [12, 1, 2],   # Dec-Feb
            'late_season': [3, 4],      # Mar-Apr
            'playoffs': [5, 6]          # May-Jun (if applicable)
        }
        
        period_results = {}
        
        for period_name, months in periods.items():
            period_data = data[data['month'].isin(months)]
            
            if len(period_data) < 50:
                continue
            
            X_period = period_data[feature_cols]
            y_period = period_data[target_col]
            
            # Make predictions
            y_pred = model.predict(X_period)
            
            # Calculate metrics
            period_results[period_name] = {
                'n_samples': len(period_data),
                'r2_score': r2_score(y_period, y_pred),
                'mae': mean_absolute_error(y_period, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_period, y_pred))
            }
            
            logger.info(
                f"{period_name}: R²={period_results[period_name]['r2_score']:.4f}, "
                f"MAE={period_results[period_name]['mae']:.2f}"
            )
        
        # Check for significant performance differences
        if period_results:
            r2_scores = [r['r2_score'] for r in period_results.values()]
            r2_range = max(r2_scores) - min(r2_scores)
            
            if r2_range > 0.1:
                logger.warning(f"Large performance variation across season periods: R² range = {r2_range:.3f}")
        
        return period_results
    
    def validate_home_away_performance(self,
                                      model,
                                      data: pd.DataFrame,
                                      target_col: str,
                                      feature_cols: List[str]) -> Dict[str, Any]:
        """
        Check model performance for home vs away games
        
        Args:
            model: Trained model
            data: Game data
            target_col: Target column
            feature_cols: Feature columns
            
        Returns:
            Dictionary with home/away analysis
        """
        results = {}
        
        for location in ['home', 'away']:
            if 'HOME_GAME' in data.columns:
                location_data = data[data['HOME_GAME'] == (1 if location == 'home' else 0)]
            else:
                logger.warning("HOME_GAME column not found")
                return results
            
            if len(location_data) < 50:
                continue
            
            X_location = location_data[feature_cols]
            y_location = location_data[target_col]
            
            # Make predictions
            y_pred = model.predict(X_location)
            
            # Calculate metrics
            results[location] = {
                'n_samples': len(location_data),
                'r2_score': r2_score(y_location, y_pred),
                'mae': mean_absolute_error(y_location, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_location, y_pred)),
                'mean_actual': y_location.mean(),
                'mean_predicted': y_pred.mean()
            }
            
            logger.info(
                f"{location.capitalize()} games: "
                f"R²={results[location]['r2_score']:.4f}, "
                f"MAE={results[location]['mae']:.2f}"
            )
        
        # Check for bias
        if 'home' in results and 'away' in results:
            r2_diff = abs(results['home']['r2_score'] - results['away']['r2_score'])
            if r2_diff > 0.05:
                logger.warning(f"Performance differs between home/away: R² diff = {r2_diff:.3f}")
        
        return results