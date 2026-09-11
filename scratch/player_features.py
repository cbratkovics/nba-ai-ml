"""
Feature engineering for NBA player predictions
"""
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import select, and_, func, desc
from database.connection import get_db_session
from api.models.game_data import GameLog, Player, Team, PlayerFeatures

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Calculate features for player predictions"""
    
    def __init__(self):
        self.feature_names = [
            'pts_last_5', 'pts_last_10', 'pts_last_20', 'pts_season',
            'reb_last_5', 'reb_last_10', 'reb_last_20', 'reb_season',
            'ast_last_5', 'ast_last_10', 'ast_last_20', 'ast_season',
            'fg_pct_last_10', 'ft_pct_last_10', 'fg3_pct_last_10',
            'minutes_last_10', 'games_played_season', 'days_since_last_game',
            'back_to_backs_last_10', 'home_ppg', 'away_ppg',
            'vs_opponent_avg_pts', 'vs_opponent_avg_reb', 'vs_opponent_avg_ast'
        ]
    
    async def calculate_player_features(
        self, 
        player_id: str, 
        game_date: str, 
        opponent_team: str
    ) -> Dict[str, float]:
        """
        Calculate features for a player prediction
        
        Args:
            player_id: NBA player ID
            game_date: Date of the game to predict
            opponent_team: Opponent team abbreviation
            
        Returns:
            Dictionary of features
        """
        try:
            # Convert game_date to date object
            if isinstance(game_date, str):
                game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
            
            async with get_db_session() as session:
                # Get player's game logs before the prediction date
                result = await session.execute(
                    select(GameLog)
                    .where(
                        and_(
                            GameLog.player_id == player_id,
                            GameLog.game_date < game_date
                        )
                    )
                    .order_by(desc(GameLog.game_date))
                    .limit(50)  # Get last 50 games for calculations
                )
                game_logs = result.scalars().all()
                
                if not game_logs:
                    logger.warning(f"No game logs found for player {player_id}")
                    return self._get_default_features()
                
                # Convert to DataFrame for easier calculations
                games_df = pd.DataFrame([g.to_dict() for g in game_logs])
                games_df['game_date'] = pd.to_datetime(games_df['game_date'])
                games_df = games_df.sort_values('game_date', ascending=False)
                
                # Calculate features
                features = {}
                
                # Points averages
                features['pts_last_5'] = games_df.head(5)['points'].mean() if len(games_df) >= 5 else games_df['points'].mean()
                features['pts_last_10'] = games_df.head(10)['points'].mean() if len(games_df) >= 10 else games_df['points'].mean()
                features['pts_last_20'] = games_df.head(20)['points'].mean() if len(games_df) >= 20 else games_df['points'].mean()
                features['pts_season'] = games_df['points'].mean()
                
                # Rebounds averages
                features['reb_last_5'] = games_df.head(5)['rebounds'].mean() if len(games_df) >= 5 else games_df['rebounds'].mean()
                features['reb_last_10'] = games_df.head(10)['rebounds'].mean() if len(games_df) >= 10 else games_df['rebounds'].mean()
                features['reb_last_20'] = games_df.head(20)['rebounds'].mean() if len(games_df) >= 20 else games_df['rebounds'].mean()
                features['reb_season'] = games_df['rebounds'].mean()
                
                # Assists averages
                features['ast_last_5'] = games_df.head(5)['assists'].mean() if len(games_df) >= 5 else games_df['assists'].mean()
                features['ast_last_10'] = games_df.head(10)['assists'].mean() if len(games_df) >= 10 else games_df['assists'].mean()
                features['ast_last_20'] = games_df.head(20)['assists'].mean() if len(games_df) >= 20 else games_df['assists'].mean()
                features['ast_season'] = games_df['assists'].mean()
                
                # Shooting percentages (last 10 games)
                last_10 = games_df.head(10)
                features['fg_pct_last_10'] = last_10['field_goal_pct'].mean() if len(last_10) > 0 else 0.45
                features['ft_pct_last_10'] = last_10['free_throw_pct'].mean() if len(last_10) > 0 else 0.75
                features['fg3_pct_last_10'] = last_10['three_point_pct'].mean() if len(last_10) > 0 else 0.35
                
                # Minutes and games
                features['minutes_last_10'] = last_10['minutes_played'].mean() if len(last_10) > 0 else 30.0
                features['games_played_season'] = len(games_df)
                
                # Days since last game
                if len(games_df) > 0:
                    last_game_date = games_df.iloc[0]['game_date'].date()
                    features['days_since_last_game'] = (game_date - last_game_date).days
                else:
                    features['days_since_last_game'] = 3
                
                # Back-to-back games in last 10
                features['back_to_backs_last_10'] = self._calculate_back_to_backs(games_df.head(10))
                
                # Home/away splits
                home_games = games_df[games_df['is_home'] == True]
                away_games = games_df[games_df['is_home'] == False]
                features['home_ppg'] = home_games['points'].mean() if len(home_games) > 0 else features['pts_season']
                features['away_ppg'] = away_games['points'].mean() if len(away_games) > 0 else features['pts_season']
                
                # Get opponent-specific stats
                opponent_stats = await self._get_vs_opponent_stats(session, player_id, opponent_team, games_df)
                features.update(opponent_stats)
                
                # Ensure all features are present
                for feature_name in self.feature_names:
                    if feature_name not in features:
                        features[feature_name] = 0.0
                
                # Round all values
                features = {k: round(float(v), 3) for k, v in features.items()}
                
                return features
                
        except Exception as e:
            logger.error(f"Error calculating features for player {player_id}: {e}")
            return self._get_default_features()
    
    def _calculate_back_to_backs(self, games_df: pd.DataFrame) -> int:
        """Calculate number of back-to-back games"""
        if len(games_df) < 2:
            return 0
        
        back_to_backs = 0
        games_df = games_df.sort_values('game_date')
        
        for i in range(1, len(games_df)):
            date_diff = (games_df.iloc[i]['game_date'] - games_df.iloc[i-1]['game_date']).days
            if date_diff == 1:
                back_to_backs += 1
        
        return back_to_backs
    
    async def _get_vs_opponent_stats(
        self, 
        session, 
        player_id: str, 
        opponent_team: str, 
        all_games_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Get player's historical stats against specific opponent"""
        
        # First, get the opponent team ID
        result = await session.execute(
            select(Team).where(Team.team_abbreviation == opponent_team)
        )
        opponent = result.scalar_one_or_none()
        
        if not opponent:
            return {
                'vs_opponent_avg_pts': all_games_df['points'].mean() if len(all_games_df) > 0 else 20.0,
                'vs_opponent_avg_reb': all_games_df['rebounds'].mean() if len(all_games_df) > 0 else 5.0,
                'vs_opponent_avg_ast': all_games_df['assists'].mean() if len(all_games_df) > 0 else 5.0
            }
        
        # Get games against this opponent
        vs_opponent = all_games_df[all_games_df['opponent_id'] == opponent.team_id]
        
        if len(vs_opponent) == 0:
            # No games against this opponent, use season averages
            return {
                'vs_opponent_avg_pts': all_games_df['points'].mean() if len(all_games_df) > 0 else 20.0,
                'vs_opponent_avg_reb': all_games_df['rebounds'].mean() if len(all_games_df) > 0 else 5.0,
                'vs_opponent_avg_ast': all_games_df['assists'].mean() if len(all_games_df) > 0 else 5.0
            }
        
        return {
            'vs_opponent_avg_pts': vs_opponent['points'].mean(),
            'vs_opponent_avg_reb': vs_opponent['rebounds'].mean(),
            'vs_opponent_avg_ast': vs_opponent['assists'].mean()
        }
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature values when no data is available"""
        return {
            'pts_last_5': 20.0,
            'pts_last_10': 20.0,
            'pts_last_20': 20.0,
            'pts_season': 20.0,
            'reb_last_5': 5.0,
            'reb_last_10': 5.0,
            'reb_last_20': 5.0,
            'reb_season': 5.0,
            'ast_last_5': 5.0,
            'ast_last_10': 5.0,
            'ast_last_20': 5.0,
            'ast_season': 5.0,
            'fg_pct_last_10': 0.45,
            'ft_pct_last_10': 0.75,
            'fg3_pct_last_10': 0.35,
            'minutes_last_10': 30.0,
            'games_played_season': 0,
            'days_since_last_game': 3,
            'back_to_backs_last_10': 0,
            'home_ppg': 20.0,
            'away_ppg': 20.0,
            'vs_opponent_avg_pts': 20.0,
            'vs_opponent_avg_reb': 5.0,
            'vs_opponent_avg_ast': 5.0
        }
    
    async def save_features(self, player_id: str, calculation_date: date, features: Dict[str, float]):
        """Save calculated features to database"""
        try:
            async with get_db_session() as session:
                # Prepare feature data for insertion
                feature_data = {
                    'player_id': player_id,
                    'calculation_date': calculation_date,
                    **features
                }
                
                # Use upsert
                from sqlalchemy.dialects.postgresql import insert
                stmt = insert(PlayerFeatures).values(**feature_data)
                stmt = stmt.on_conflict_do_update(
                    constraint='uq_player_date_features',
                    set_=dict(features)
                )
                
                await session.execute(stmt)
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error saving features for player {player_id}: {e}")
            raise
    
    def prepare_features_for_model(self, features: Dict[str, float]) -> np.ndarray:
        """
        Prepare features in the correct order for model input
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Numpy array of features in correct order
        """
        # Define the exact feature order expected by the model
        feature_order = [
            'pts_last_5', 'pts_last_10', 'pts_last_20',
            'reb_last_5', 'reb_last_10', 'reb_last_20',
            'ast_last_5', 'ast_last_10', 'ast_last_20',
            'fg_pct_last_10', 'ft_pct_last_10', 'fg3_pct_last_10',
            'minutes_last_10', 'games_played_season', 'days_since_last_game',
            'home_ppg', 'away_ppg', 'back_to_backs_last_10',
            'vs_opponent_avg_pts', 'pts_season'
        ]
        
        # Extract features in order
        feature_array = []
        for feature_name in feature_order:
            if feature_name in features:
                feature_array.append(features[feature_name])
            else:
                # Use reasonable defaults if feature is missing
                if 'pts' in feature_name:
                    feature_array.append(20.0)
                elif 'reb' in feature_name:
                    feature_array.append(5.0)
                elif 'ast' in feature_name:
                    feature_array.append(5.0)
                elif 'pct' in feature_name:
                    feature_array.append(0.45)
                elif feature_name == 'minutes_last_10':
                    feature_array.append(30.0)
                elif feature_name == 'games_played_season':
                    feature_array.append(20)
                elif feature_name == 'days_since_last_game':
                    feature_array.append(2)
                else:
                    feature_array.append(0.0)
        
        return np.array(feature_array).reshape(1, -1)