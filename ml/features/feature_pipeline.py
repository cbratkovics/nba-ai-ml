"""
Advanced feature engineering pipeline for NBA predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Feature engineering pipeline for NBA player predictions"""
    
    def __init__(self):
        self.feature_names = []
        self.scaler_params = {}
        
    def fit(self, df: pd.DataFrame) -> 'FeaturePipeline':
        """Fit the pipeline on training data"""
        # Calculate scaling parameters for normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.scaler_params[col] = {
                'mean': df[col].mean(),
                'std': df[col].std()
            }
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into features"""
        df = df.copy()
        
        # Core features
        df = self._add_rolling_averages(df)
        df = self._add_opponent_features(df)
        df = self._add_rest_features(df)
        df = self._add_streak_features(df)
        df = self._add_trend_features(df)
        df = self._add_matchup_history(df)
        df = self._add_advanced_stats(df)
        df = self._add_time_features(df)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in ['PLAYER_ID', 'PLAYER_NAME', 'GAME_DATE']]
        
        return df
    
    def _add_rolling_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling average features"""
        windows = [3, 5, 10, 20]
        stats = ['PTS', 'REB', 'AST', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS']
        
        for window in windows:
            for stat in stats:
                if stat in df.columns:
                    col_name = f'{stat}_MA{window}'
                    df[col_name] = df.groupby('PLAYER_ID')[stat].transform(
                        lambda x: x.rolling(window, min_periods=min(3, window)).mean().shift(1)
                    )
                    
                    # Add rolling std for volatility
                    std_name = f'{stat}_STD{window}'
                    df[std_name] = df.groupby('PLAYER_ID')[stat].transform(
                        lambda x: x.rolling(window, min_periods=min(3, window)).std().shift(1)
                    )
        
        return df
    
    def _add_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add opponent-related features"""
        if 'OPP_DEF_RATING' not in df.columns:
            df['OPP_DEF_RATING'] = 110  # League average default
        
        # Normalize opponent defensive rating
        df['OPP_DEF_RATING_NORM'] = (df['OPP_DEF_RATING'] - 110) / 10
        
        # Position vs opponent defense (simplified)
        df['MATCHUP_DIFFICULTY'] = df['OPP_DEF_RATING'] / 110
        
        return df
    
    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rest-related features"""
        if 'REST_DAYS' not in df.columns:
            df['REST_DAYS'] = 1
        
        # Categorize rest
        df['NO_REST'] = (df['REST_DAYS'] == 0).astype(int)
        df['ONE_DAY_REST'] = (df['REST_DAYS'] == 1).astype(int)
        df['TWO_PLUS_REST'] = (df['REST_DAYS'] >= 2).astype(int)
        
        # Back-to-back indicator
        if 'BACK_TO_BACK' not in df.columns:
            df['BACK_TO_BACK'] = df['NO_REST']
        
        # Games in last N days (fatigue indicator)
        df['GAMES_LAST_7_DAYS'] = df.groupby('PLAYER_ID')['GAME_DATE'].transform(
            lambda x: x.rolling('7D').count().shift(1)
        )
        
        df['GAMES_LAST_14_DAYS'] = df.groupby('PLAYER_ID')['GAME_DATE'].transform(
            lambda x: x.rolling('14D').count().shift(1)
        )
        
        return df
    
    def _add_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add hot/cold streak features"""
        # Points streak
        df['PTS_VS_AVG'] = df.groupby('PLAYER_ID').apply(
            lambda x: (x['PTS'] - x['PTS'].mean()).values
        ).explode().values
        
        df['HOT_STREAK_3'] = df.groupby('PLAYER_ID')['PTS_VS_AVG'].transform(
            lambda x: (x > 0).rolling(3, min_periods=1).sum().shift(1)
        )
        
        df['HOT_STREAK_5'] = df.groupby('PLAYER_ID')['PTS_VS_AVG'].transform(
            lambda x: (x > 0).rolling(5, min_periods=1).sum().shift(1)
        )
        
        # Shooting streak
        if 'FG_PCT' in df.columns:
            df['SHOOTING_STREAK'] = df.groupby('PLAYER_ID')['FG_PCT'].transform(
                lambda x: (x > x.mean()).rolling(3, min_periods=1).sum().shift(1)
            )
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend features"""
        # Minutes trend
        df['MIN_TREND_3'] = df.groupby('PLAYER_ID')['MIN'].transform(
            lambda x: x.diff(periods=3).shift(1)
        )
        
        df['MIN_TREND_5'] = df.groupby('PLAYER_ID')['MIN'].transform(
            lambda x: x.diff(periods=5).shift(1)
        )
        
        # Usage trend
        if 'USAGE_RATE' in df.columns:
            df['USAGE_TREND'] = df.groupby('PLAYER_ID')['USAGE_RATE'].transform(
                lambda x: x.diff(periods=3).shift(1)
            )
        else:
            # Calculate usage rate proxy
            df['USAGE_RATE'] = ((df.get('FGA', 0) + 0.44 * df.get('FTA', 0) + df.get('TOV', 0)) / 
                               df.get('MIN', 1)).replace([np.inf, -np.inf], 0)
            df['USAGE_TREND'] = df.groupby('PLAYER_ID')['USAGE_RATE'].transform(
                lambda x: x.diff(periods=3).shift(1)
            )
        
        # Performance trend (weighted recent games more)
        df['PTS_TREND'] = df.groupby('PLAYER_ID')['PTS'].transform(
            lambda x: x.ewm(span=5, adjust=False).mean().shift(1) - x.rolling(10, min_periods=5).mean().shift(1)
        )
        
        return df
    
    def _add_matchup_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add historical matchup features"""
        # Average performance against this opponent
        df['VS_OPP_PTS_AVG'] = df.groupby(['PLAYER_ID', 'OPPONENT'])['PTS'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        df['VS_OPP_REB_AVG'] = df.groupby(['PLAYER_ID', 'OPPONENT'])['REB'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        df['VS_OPP_AST_AVG'] = df.groupby(['PLAYER_ID', 'OPPONENT'])['AST'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # Games played against opponent
        df['VS_OPP_GAMES'] = df.groupby(['PLAYER_ID', 'OPPONENT']).cumcount()
        
        return df
    
    def _add_advanced_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced basketball statistics"""
        # True Shooting Percentage
        if 'FGA' in df.columns and 'FTA' in df.columns:
            df['TS_PCT'] = (df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))).replace([np.inf, -np.inf], 0)
            df['TS_PCT_MA5'] = df.groupby('PLAYER_ID')['TS_PCT'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            )
        
        # Effective Field Goal Percentage
        if 'FGM' in df.columns and 'FG3M' in df.columns:
            df['EFG_PCT'] = ((df['FGM'] + 0.5 * df['FG3M']) / df['FGA']).replace([np.inf, -np.inf], 0)
            df['EFG_PCT_MA5'] = df.groupby('PLAYER_ID')['EFG_PCT'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            )
        
        # Assist to Turnover Ratio
        if 'AST' in df.columns and 'TOV' in df.columns:
            df['AST_TO_RATIO'] = (df['AST'] / df['TOV'].replace(0, 1))
            df['AST_TO_RATIO_MA5'] = df.groupby('PLAYER_ID')['AST_TO_RATIO'].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            )
        
        # Fantasy points (DraftKings scoring)
        df['FANTASY_PTS'] = (df.get('PTS', 0) * 1.0 + 
                            df.get('REB', 0) * 1.25 + 
                            df.get('AST', 0) * 1.5 + 
                            df.get('STL', 0) * 2.0 + 
                            df.get('BLK', 0) * 2.0 - 
                            df.get('TOV', 0) * 0.5)
        
        df['FANTASY_PTS_MA5'] = df.groupby('PLAYER_ID')['FANTASY_PTS'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            
            # Day of week
            df['DAY_OF_WEEK'] = df['GAME_DATE'].dt.dayofweek
            
            # Month
            df['MONTH'] = df['GAME_DATE'].dt.month
            
            # Days since season start (experience/fatigue)
            season_start = df['GAME_DATE'].min()
            df['DAYS_INTO_SEASON'] = (df['GAME_DATE'] - season_start).dt.days
            
            # Weekend game
            df['WEEKEND_GAME'] = df['DAY_OF_WEEK'].isin([4, 5, 6]).astype(int)
        
        return df
    
    def get_feature_importance(self, model) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return pd.DataFrame()
    
    def prepare_for_prediction(self, player_id: str, game_date: str, opponent: str, 
                              historical_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for a single prediction"""
        # Filter to player's recent games
        player_data = historical_data[historical_data['PLAYER_ID'] == player_id].copy()
        
        if player_data.empty:
            # Return default features if no historical data
            return np.zeros(len(self.feature_names))
        
        # Sort by date
        player_data = player_data.sort_values('GAME_DATE')
        
        # Get most recent features
        latest_features = player_data.iloc[-1][self.feature_names].values
        
        # Update opponent-specific features
        opp_features = player_data[player_data['OPPONENT'] == opponent]
        if not opp_features.empty:
            # Update matchup history features
            latest_features = self._update_matchup_features(latest_features, opp_features)
        
        return latest_features
    
    def _update_matchup_features(self, features: np.ndarray, opp_data: pd.DataFrame) -> np.ndarray:
        """Update features with latest matchup data"""
        # This would update specific matchup-related features
        # Implementation depends on feature ordering
        return features