"""
Data collection pipeline for NBA statistics
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert
from database.connection import get_db_session
from api.models.game_data import Player, GameLog, Team, Schedule
from api.data.nba_client import NBAStatsClient

logger = logging.getLogger(__name__)


class NBADataPipeline:
    """Pipeline for collecting and storing NBA data"""
    
    def __init__(self):
        self.client = NBAStatsClient()
        
    async def load_teams(self) -> int:
        """
        Load all NBA teams into the database
        
        Returns:
            Number of teams loaded
        """
        try:
            # Get all teams
            teams = self.client.get_all_teams()
            
            async with get_db_session() as session:
                for team_data in teams:
                    # Use upsert to handle existing teams
                    stmt = insert(Team).values(**team_data)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['team_id'],
                        set_=dict(team_data)
                    )
                    await session.execute(stmt)
                
                await session.commit()
                logger.info(f"Loaded {len(teams)} teams")
                return len(teams)
                
        except Exception as e:
            logger.error(f"Error loading teams: {e}")
            raise
    
    async def load_players(self, only_active: bool = True) -> int:
        """
        Load NBA players into the database
        
        Args:
            only_active: If True, only load active players
            
        Returns:
            Number of players loaded
        """
        try:
            # Get player list
            players = self.client.get_player_list(only_current=only_active)
            
            async with get_db_session() as session:
                loaded_count = 0
                
                for player_data in players:
                    try:
                        # Get additional player info if available
                        player_info = self.client.get_player_info(player_data['player_id'])
                        if player_info:
                            player_data.update(player_info)
                        
                        # Prepare data for insertion
                        insert_data = {
                            'player_id': player_data['player_id'],
                            'player_name': player_data['player_name'],
                            'team_id': player_data.get('team_id'),
                            'team_abbreviation': player_data.get('team_abbreviation'),
                            'position': player_data.get('position'),
                            'jersey_number': player_data.get('jersey_number'),
                            'height': player_data.get('height'),
                            'weight': player_data.get('weight'),
                            'birth_date': player_data.get('birth_date'),
                            'country': player_data.get('country'),
                            'school': player_data.get('school'),
                            'draft_year': player_data.get('draft_year'),
                            'draft_round': player_data.get('draft_round'),
                            'draft_number': player_data.get('draft_number'),
                            'is_active': player_data.get('is_active', True)
                        }
                        
                        # Use upsert
                        stmt = insert(Player).values(**insert_data)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=['player_id'],
                            set_=dict(insert_data)
                        )
                        await session.execute(stmt)
                        loaded_count += 1
                        
                        # Add small delay to avoid rate limiting
                        if loaded_count % 50 == 0:
                            await asyncio.sleep(1)
                            logger.info(f"Loaded {loaded_count} players so far...")
                            
                    except Exception as e:
                        logger.warning(f"Error loading player {player_data.get('player_name')}: {e}")
                        continue
                
                await session.commit()
                logger.info(f"Successfully loaded {loaded_count} players")
                return loaded_count
                
        except Exception as e:
            logger.error(f"Error loading players: {e}")
            raise
    
    async def load_player_game_logs(self, player_id: str, seasons: List[str]) -> int:
        """
        Load game logs for a specific player
        
        Args:
            player_id: NBA player ID
            seasons: List of seasons to load
            
        Returns:
            Number of games loaded
        """
        try:
            # Get game logs
            game_logs_df = self.client.get_player_game_logs(player_id, seasons)
            
            if game_logs_df.empty:
                logger.info(f"No game logs found for player {player_id}")
                return 0
            
            async with get_db_session() as session:
                games_loaded = 0
                
                for _, game in game_logs_df.iterrows():
                    try:
                        # Parse game data
                        game_data = {
                            'player_id': player_id,
                            'game_id': game['Game_ID'],
                            'game_date': pd.to_datetime(game['GAME_DATE']).date(),
                            'season': game.get('SEASON', seasons[0]),
                            'team_id': str(game.get('TEAM_ID', '')),
                            'opponent_id': str(game.get('OPPONENT_TEAM_ID', '')),
                            'is_home': 'vs.' in game.get('MATCHUP', ''),
                            'minutes_played': self._parse_minutes(game.get('MIN')),
                            'points': int(game.get('PTS', 0)),
                            'rebounds': int(game.get('REB', 0)),
                            'assists': int(game.get('AST', 0)),
                            'steals': int(game.get('STL', 0)),
                            'blocks': int(game.get('BLK', 0)),
                            'turnovers': int(game.get('TOV', 0)),
                            'field_goals_made': int(game.get('FGM', 0)),
                            'field_goals_attempted': int(game.get('FGA', 0)),
                            'field_goal_pct': float(game.get('FG_PCT', 0)),
                            'three_pointers_made': int(game.get('FG3M', 0)),
                            'three_pointers_attempted': int(game.get('FG3A', 0)),
                            'three_point_pct': float(game.get('FG3_PCT', 0)),
                            'free_throws_made': int(game.get('FTM', 0)),
                            'free_throws_attempted': int(game.get('FTA', 0)),
                            'free_throw_pct': float(game.get('FT_PCT', 0)),
                            'offensive_rebounds': int(game.get('OREB', 0)),
                            'defensive_rebounds': int(game.get('DREB', 0)),
                            'personal_fouls': int(game.get('PF', 0)),
                            'plus_minus': int(game.get('PLUS_MINUS', 0)),
                            'game_score': self._calculate_game_score(game)
                        }
                        
                        # Use upsert to handle duplicates
                        stmt = insert(GameLog).values(**game_data)
                        stmt = stmt.on_conflict_do_update(
                            constraint='uq_player_game',
                            set_=dict(game_data)
                        )
                        await session.execute(stmt)
                        games_loaded += 1
                        
                    except Exception as e:
                        logger.warning(f"Error loading game {game.get('Game_ID')} for player {player_id}: {e}")
                        continue
                
                await session.commit()
                logger.info(f"Loaded {games_loaded} games for player {player_id}")
                return games_loaded
                
        except Exception as e:
            logger.error(f"Error loading game logs for player {player_id}: {e}")
            raise
    
    async def load_historical_data(self, seasons: List[str] = None, sample_players: int = None):
        """
        Load historical data for all active players
        
        Args:
            seasons: List of seasons to load (default: last 3 seasons)
            sample_players: If specified, only load data for this many players (for testing)
        """
        if seasons is None:
            seasons = ['2022-23', '2023-24', '2024-25']
        
        try:
            # First, load teams
            logger.info("Loading NBA teams...")
            await self.load_teams()
            
            # Load players
            logger.info("Loading NBA players...")
            await self.load_players(only_active=True)
            
            # Get all active players from database
            async with get_db_session() as session:
                result = await session.execute(
                    select(Player).where(Player.is_active == True)
                )
                players = result.scalars().all()
            
            # Limit players if sampling
            if sample_players:
                players = players[:sample_players]
            
            logger.info(f"Loading game logs for {len(players)} players...")
            
            # Load game logs for each player
            total_games = 0
            for i, player in enumerate(players):
                try:
                    games_loaded = await self.load_player_game_logs(
                        player.player_id, 
                        seasons
                    )
                    total_games += games_loaded
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Progress: {i + 1}/{len(players)} players processed, {total_games} total games loaded")
                    
                    # Add delay to avoid rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error loading data for player {player.player_name}: {e}")
                    continue
            
            logger.info(f"Historical data load complete. Loaded {total_games} total games for {len(players)} players")
            
        except Exception as e:
            logger.error(f"Error in historical data load: {e}")
            raise
    
    async def update_daily_games(self):
        """
        Update game logs for games played yesterday
        """
        try:
            # Get yesterday's games
            yesterday_games = self.client.get_yesterday_games()
            
            if not yesterday_games:
                logger.info("No games found for yesterday")
                return
            
            logger.info(f"Found {len(yesterday_games)} games from yesterday")
            
            # Get unique player IDs from yesterday's games
            player_ids = set()
            
            for game in yesterday_games:
                # Get game logs for both teams
                for team_key in ['home_team_id', 'away_team_id']:
                    team_id = game.get(team_key)
                    if team_id:
                        # This would need to be implemented to get player IDs from team rosters
                        # For now, we'll use a simplified approach
                        pass
            
            # For each player who played yesterday, update their game log
            # This is a simplified implementation - in production, you'd want to
            # fetch the actual box scores for yesterday's games
            
            logger.info("Daily game update complete")
            
        except Exception as e:
            logger.error(f"Error in daily game update: {e}")
            raise
    
    def _parse_minutes(self, minutes_str: str) -> float:
        """Convert minutes string (MM:SS) to float"""
        if not minutes_str or pd.isna(minutes_str):
            return 0.0
        
        try:
            if ':' in str(minutes_str):
                parts = str(minutes_str).split(':')
                return float(parts[0]) + float(parts[1]) / 60
            return float(minutes_str)
        except:
            return 0.0
    
    def _calculate_game_score(self, game_row) -> float:
        """
        Calculate John Hollinger's game score
        Game Score = PTS + 0.4 * FG - 0.7 * FGA - 0.4*(FTA - FT) + 0.7 * ORB + 0.3 * DRB + STL + 0.7 * AST + 0.7 * BLK - 0.4 * PF - TOV
        """
        try:
            pts = float(game_row.get('PTS', 0))
            fgm = float(game_row.get('FGM', 0))
            fga = float(game_row.get('FGA', 0))
            ftm = float(game_row.get('FTM', 0))
            fta = float(game_row.get('FTA', 0))
            oreb = float(game_row.get('OREB', 0))
            dreb = float(game_row.get('DREB', 0))
            stl = float(game_row.get('STL', 0))
            ast = float(game_row.get('AST', 0))
            blk = float(game_row.get('BLK', 0))
            pf = float(game_row.get('PF', 0))
            tov = float(game_row.get('TOV', 0))
            
            game_score = (pts + 0.4 * fgm - 0.7 * fga - 0.4 * (fta - ftm) + 
                         0.7 * oreb + 0.3 * dreb + stl + 0.7 * ast + 
                         0.7 * blk - 0.4 * pf - tov)
            
            return round(game_score, 2)
        except:
            return 0.0