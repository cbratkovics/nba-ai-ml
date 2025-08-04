"""
Production feature engineering pipeline for NBA predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class NBAFeatureEngineer:
    """
    Feature engineering pipeline for NBA player predictions
    Creates 50+ features from raw game data
    """
    
    def __init__(self, scaling_method: str = 'robust'):
        self.scaler = RobustScaler() if scaling_method == 'robust' else StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features
        
        Args:
            df: Raw game data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Ensure datetime
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values('GAME_DATE', ascending=False)
        
        # Basic features
        df = self._create_basic_features(df)
        
        # Rolling statistics
        df = self._create_rolling_features(df)
        
        # Trend features
        df = self._create_trend_features(df)
        
        # Matchup features
        df = self._create_matchup_features(df)
        
        # Advanced metrics
        df = self._create_advanced_metrics(df)
        
        # Interaction features
        df = self._create_interaction_features(df)
        
        # Time-based features
        df = self._create_time_features(df)
        
        logger.info(f"Created {len(df.columns)} total features")
        self.feature_columns = [col for col in df.columns if col not in 
                               ['GAME_DATE', 'GAME_ID', 'PLAYER_ID', 'TEAM_ID']]
        
        return df
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic statistical features"""
        
        # Efficiency metrics
        df['FANTASY_POINTS'] = (df['PTS'] + 
                                df['REB'] * 1.2 + 
                                df['AST'] * 1.5 + 
                                df['STL'] * 3 + 
                                df['BLK'] * 3 - 
                                df['TOV'])
        
        # Shooting efficiency
        df['EFG_PCT'] = np.where(df['FGA'] > 0,
                                 (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'],
                                 0)
        
        df['TS_PCT'] = np.where((2 * df['FGA'] + 0.44 * df['FTA']) > 0,
                                df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'])),
                                0)
        
        # Usage and involvement
        df['USAGE_RATE'] = np.where(df['MIN'] > 0,
                                    ((df['FGA'] + 0.44 * df['FTA'] + df['TOV']) * 48) / df['MIN'],
                                    0)
        
        df['ASSIST_RATIO'] = np.where((df['FGA'] + df['FTA'] + df['AST'] + df['TOV']) > 0,
                                      df['AST'] / (df['FGA'] + df['FTA'] + df['AST'] + df['TOV']),
                                      0)
        
        # Rebounding
        df['REB_RATE'] = np.where(df['MIN'] > 0,
                                  df['REB'] / df['MIN'] * 48,
                                  0)
        
        # Rest and fatigue
        df['CUMULATIVE_MIN_5G'] = df['MIN'].rolling(window=5, min_periods=1).sum()
        df['REST_ADVANTAGE'] = df['REST_DAYS'] - 1  # Days above/below normal rest
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling average features"""
        
        # Different window sizes
        windows = [3, 5, 10, 20]
        
        # Key stats to track
        stats = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG3M', 'FG3A', 
                'FTM', 'FTA', 'STL', 'BLK', 'TOV', 'MIN', 'PLUS_MINUS']
        
        for window in windows:
            for stat in stats:
                if stat in df.columns:
                    # Rolling mean
                    df[f'{stat}_MA{window}'] = df[stat].rolling(
                        window=window, min_periods=1
                    ).mean()
                    
                    # Rolling std (volatility)
                    df[f'{stat}_STD{window}'] = df[stat].rolling(
                        window=window, min_periods=2
                    ).std()
                    
                    # Difference from rolling average
                    df[f'{stat}_DIFF_MA{window}'] = df[stat] - df[f'{stat}_MA{window}']
        
        # Weighted moving averages (more weight on recent games)
        for stat in ['PTS', 'REB', 'AST']:
            weights = np.array([0.4, 0.3, 0.2, 0.1])
            df[f'{stat}_WMA4'] = df[stat].rolling(window=4).apply(
                lambda x: np.sum(weights[-len(x):] * x) / np.sum(weights[-len(x):])
            )
        
        return df
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend and momentum features"""
        
        for stat in ['PTS', 'REB', 'AST', 'MIN']:
            if stat in df.columns:
                # Recent form (last 3 vs previous 3)
                df[f'{stat}_FORM'] = (df[stat].rolling(3).mean() - 
                                     df[stat].rolling(6).mean().shift(3))
                
                # Streak features
                df[f'{stat}_STREAK_UP'] = self._calculate_streak(df[stat], 'up')
                df[f'{stat}_STREAK_DOWN'] = self._calculate_streak(df[stat], 'down')
                
                # Trend slope (linear regression coefficient)
                df[f'{stat}_TREND'] = df[stat].rolling(window=5).apply(
                    self._calculate_trend_slope
                )
        
        # Hot/cold streaks
        df['HOT_STREAK'] = (df['PTS'] > df['PTS_MA10']).astype(int)
        df['COLD_STREAK'] = (df['PTS'] < df['PTS_MA10']).astype(int)
        
        return df
    
    def _create_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create opponent and matchup-based features"""
        
        # Opponent quality indicators (would need opponent stats in production)
        df['MATCHUP_DIFFICULTY'] = np.random.uniform(0.8, 1.2, len(df))  # Placeholder
        
        # Home/Away adjustments
        df['HOME_BOOST'] = np.where(df['HOME_GAME'] == 1, 1.05, 0.95)
        
        # Back-to-back penalty
        df['B2B_PENALTY'] = np.where(df['BACK_TO_BACK'] == 1, 0.95, 1.0)
        
        # Division/Conference games (simplified)
        df['DIVISION_GAME'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        df['CONFERENCE_GAME'] = np.random.choice([0, 1], size=len(df), p=[0.4, 0.6])
        
        return df
    
    def _create_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced basketball metrics"""
        
        # Player Impact Estimate (simplified)
        df['PIE'] = (df['PTS'] + df['FGM'] + df['FTM'] - df['FGA'] - df['FTA'] + 
                    df['DREB'] + 0.5 * df['OREB'] + df['AST'] + df['STL'] + 
                    0.5 * df['BLK'] - df['PF'] - df['TOV'])
        
        # Offensive Rating estimate
        df['OFF_RATING'] = np.where(df['MIN'] > 0,
                                   (df['PTS'] / df['MIN']) * 48 * 5,
                                   0)
        
        # Game Score (John Hollinger's formula)
        df['GAME_SCORE'] = (df['PTS'] + 0.4 * df['FGM'] - 0.7 * df['FGA'] -
                          0.4 * (df['FTA'] - df['FTM']) + 0.7 * df['OREB'] +
                          0.3 * df['DREB'] + df['STL'] + 0.7 * df['AST'] +
                          0.7 * df['BLK'] - 0.4 * df['PF'] - df['TOV'])
        
        # Versatility Index
        df['VERSATILITY'] = (
            (df['PTS'] > 10).astype(int) +
            (df['REB'] > 5).astype(int) +
            (df['AST'] > 5).astype(int) +
            (df['STL'] > 1).astype(int) +
            (df['BLK'] > 1).astype(int)
        )
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature interactions"""
        
        # Minutes x Usage interaction
        df['MIN_USAGE_INTERACTION'] = df['MIN'] * df['USAGE_RATE']
        
        # Rest x Minutes interaction
        df['REST_MIN_INTERACTION'] = df['REST_DAYS'] * df['MIN']
        
        # Home x Rest interaction
        df['HOME_REST_INTERACTION'] = df['HOME_GAME'] * df['REST_DAYS']
        
        # Form x Matchup interaction
        if 'PTS_FORM' in df.columns:
            df['FORM_MATCHUP_INTERACTION'] = df['PTS_FORM'] * df['MATCHUP_DIFFICULTY']
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        # Day of week
        df['DAY_OF_WEEK'] = df['GAME_DATE'].dt.dayofweek
        df['WEEKEND_GAME'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)
        
        # Month of season
        df['MONTH'] = df['GAME_DATE'].dt.month
        df['EARLY_SEASON'] = df['MONTH'].isin([10, 11]).astype(int)
        df['MID_SEASON'] = df['MONTH'].isin([12, 1, 2]).astype(int)
        df['LATE_SEASON'] = df['MONTH'].isin([3, 4]).astype(int)
        
        # Games into season
        df['GAMES_PLAYED'] = range(len(df), 0, -1)
        df['SEASON_FATIGUE'] = df['GAMES_PLAYED'] / 82
        
        return df
    
    def _calculate_streak(self, series: pd.Series, direction: str = 'up') -> pd.Series:
        """Calculate consecutive games streak"""
        if direction == 'up':
            compare = series > series.shift(1)
        else:
            compare = series < series.shift(1)
        
        streak = compare.astype(int)
        streak = streak.groupby((streak != streak.shift()).cumsum()).cumsum()
        return streak
    
    def _calculate_trend_slope(self, values: pd.Series) -> float:
        """Calculate linear trend slope"""
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        return coefficients[0]
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Transform features for model input
        
        Returns:
            Tuple of (transformed DataFrame, feature column names)
        """
        # Create features
        df = self.create_features(df)
        
        # Handle missing values
        df[self.feature_columns] = self.imputer.fit_transform(df[self.feature_columns])
        
        # Scale features
        df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        
        return df, self.feature_columns
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by category for analysis"""
        groups = {
            'basic_stats': [],
            'rolling_averages': [],
            'trends': [],
            'matchup': [],
            'advanced': [],
            'interactions': [],
            'time': []
        }
        
        for col in self.feature_columns:
            if '_MA' in col or '_WMA' in col:
                groups['rolling_averages'].append(col)
            elif '_TREND' in col or '_FORM' in col or '_STREAK' in col:
                groups['trends'].append(col)
            elif 'MATCHUP' in col or 'HOME' in col or 'B2B' in col:
                groups['matchup'].append(col)
            elif any(metric in col for metric in ['PIE', 'RATING', 'GAME_SCORE', 'VERSATILITY']):
                groups['advanced'].append(col)
            elif 'INTERACTION' in col:
                groups['interactions'].append(col)
            elif any(time in col for time in ['DAY', 'WEEK', 'MONTH', 'SEASON']):
                groups['time'].append(col)
            else:
                groups['basic_stats'].append(col)
        
        return groups


def create_training_features(player_data: pd.DataFrame,
                            team_data: pd.DataFrame = None,
                            opponent_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create comprehensive feature set for training
    
    Args:
        player_data: Player game logs
        team_data: Team statistics (optional)
        opponent_data: Opponent statistics (optional)
        
    Returns:
        DataFrame with all features
    """
    engineer = NBAFeatureEngineer()
    
    # Create player features
    features_df = engineer.create_features(player_data)
    
    # Add team features if available
    if team_data is not None:
        team_features = engineer.create_features(team_data)
        features_df = features_df.merge(
            team_features.add_prefix('TEAM_'),
            left_on=['GAME_ID', 'TEAM_ID'],
            right_on=['TEAM_GAME_ID', 'TEAM_TEAM_ID'],
            how='left'
        )
    
    # Add opponent features if available
    if opponent_data is not None:
        opp_features = engineer.create_features(opponent_data)
        features_df = features_df.merge(
            opp_features.add_prefix('OPP_'),
            left_on='GAME_ID',
            right_on='OPP_GAME_ID',
            how='left'
        )
    
    logger.info(f"Created {len(features_df.columns)} total features")
    return features_df