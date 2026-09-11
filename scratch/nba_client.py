"""
NBA Stats API Client for fetching real NBA data
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import time
from nba_api.stats.endpoints import (
    commonallplayers,
    playergamelog,
    teamgamelog,
    leaguegamefinder,
    playercareerstats,
    commonplayerinfo
)
from nba_api.stats.static import players, teams
import pandas as pd

logger = logging.getLogger(__name__)


class NBAStatsClient:
    """Client for fetching NBA statistics data"""
    
    def __init__(self, rate_limit_delay: float = 0.6):
        """
        Initialize NBA Stats client
        
        Args:
            rate_limit_delay: Delay between API calls to avoid rate limiting
        """
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Ensure we don't hit rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_request)
        self._last_request_time = time.time()
    
    def get_player_list(self, only_current: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch all NBA players
        
        Args:
            only_current: If True, only return active players
            
        Returns:
            List of player dictionaries with id, name, team info
        """
        try:
            self._rate_limit()
            
            # Get all players from the current season
            all_players_data = commonallplayers.CommonAllPlayers(
                is_only_current_season=1 if only_current else 0,
                league_id='00'  # NBA
            ).get_data_frames()[0]
            
            # Convert to list of dicts
            players_list = []
            for _, player in all_players_data.iterrows():
                players_list.append({
                    'player_id': str(player['PERSON_ID']),
                    'player_name': player['DISPLAY_FIRST_LAST'],
                    'team_id': str(player['TEAM_ID']) if player['TEAM_ID'] else None,
                    'team_name': player['TEAM_NAME'] if player['TEAM_NAME'] else None,
                    'team_abbreviation': player['TEAM_ABBREVIATION'] if player['TEAM_ABBREVIATION'] else None,
                    'is_active': player['ROSTERSTATUS'] == 1,
                    'from_year': player['FROM_YEAR'],
                    'to_year': player['TO_YEAR']
                })
            
            logger.info(f"Fetched {len(players_list)} players")
            return players_list
            
        except Exception as e:
            logger.error(f"Error fetching player list: {e}")
            raise
    
    def get_player_game_logs(self, player_id: str, seasons: List[str] = None) -> pd.DataFrame:
        """
        Fetch game logs for a specific player
        
        Args:
            player_id: NBA player ID
            seasons: List of seasons (e.g., ['2022-23', '2023-24', '2024-25'])
            
        Returns:
            DataFrame with game logs
        """
        if seasons is None:
            seasons = ['2024-25']  # Default to current season
        
        all_game_logs = []
        
        for season in seasons:
            try:
                self._rate_limit()
                
                # Fetch game logs for the season
                game_logs = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    season_type_all_star='Regular Season'
                ).get_data_frames()[0]
                
                if not game_logs.empty:
                    # Add season column
                    game_logs['SEASON'] = season
                    all_game_logs.append(game_logs)
                    logger.info(f"Fetched {len(game_logs)} games for player {player_id} in {season}")
                
            except Exception as e:
                logger.warning(f"Error fetching game logs for player {player_id} in {season}: {e}")
                continue
        
        if all_game_logs:
            return pd.concat(all_game_logs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_team_schedule(self, team_id: str, season: str = '2024-25') -> pd.DataFrame:
        """
        Fetch team schedule for upcoming games
        
        Args:
            team_id: NBA team ID
            season: Season to fetch schedule for
            
        Returns:
            DataFrame with team schedule
        """
        try:
            self._rate_limit()
            
            # Get team game logs (includes past and scheduled games)
            team_games = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season,
                season_type_all_star='Regular Season'
            ).get_data_frames()[0]
            
            return team_games
            
        except Exception as e:
            logger.error(f"Error fetching team schedule for {team_id}: {e}")
            raise
    
    def get_yesterday_games(self) -> List[Dict[str, Any]]:
        """
        Fetch all games completed yesterday
        
        Returns:
            List of game dictionaries
        """
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        try:
            self._rate_limit()
            
            # Use LeagueGameFinder to get games from yesterday
            games = leaguegamefinder.LeagueGameFinder(
                date_from_nullable=yesterday,
                date_to_nullable=yesterday,
                league_id_nullable='00'  # NBA
            ).get_data_frames()[0]
            
            # Process games to get unique game IDs
            unique_games = {}
            for _, game in games.iterrows():
                game_id = game['GAME_ID']
                if game_id not in unique_games:
                    unique_games[game_id] = {
                        'game_id': game_id,
                        'game_date': game['GAME_DATE'],
                        'home_team_id': None,
                        'away_team_id': None,
                        'home_team_name': None,
                        'away_team_name': None
                    }
                
                # Determine home/away based on matchup
                if '@' in game['MATCHUP']:
                    unique_games[game_id]['away_team_id'] = str(game['TEAM_ID'])
                    unique_games[game_id]['away_team_name'] = game['TEAM_NAME']
                else:
                    unique_games[game_id]['home_team_id'] = str(game['TEAM_ID'])
                    unique_games[game_id]['home_team_name'] = game['TEAM_NAME']
            
            games_list = list(unique_games.values())
            logger.info(f"Found {len(games_list)} games from yesterday")
            return games_list
            
        except Exception as e:
            logger.error(f"Error fetching yesterday's games: {e}")
            raise
    
    def get_player_recent_stats(self, player_id: str, n_games: int = 10) -> Dict[str, Any]:
        """
        Get player's recent performance statistics
        
        Args:
            player_id: NBA player ID
            n_games: Number of recent games to fetch
            
        Returns:
            Dictionary with recent stats averages
        """
        try:
            # Get current season game logs
            game_logs = self.get_player_game_logs(player_id, seasons=['2024-25'])
            
            if game_logs.empty:
                return {}
            
            # Sort by date and get most recent n games
            game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'])
            recent_games = game_logs.nlargest(n_games, 'GAME_DATE')
            
            # Calculate averages
            stats = {
                'games_played': len(recent_games),
                'avg_points': recent_games['PTS'].mean(),
                'avg_rebounds': recent_games['REB'].mean(),
                'avg_assists': recent_games['AST'].mean(),
                'avg_steals': recent_games['STL'].mean(),
                'avg_blocks': recent_games['BLK'].mean(),
                'avg_turnovers': recent_games['TOV'].mean(),
                'avg_minutes': recent_games['MIN'].mean(),
                'avg_fg_pct': recent_games['FG_PCT'].mean(),
                'avg_ft_pct': recent_games['FT_PCT'].mean(),
                'avg_fg3_pct': recent_games['FG3_PCT'].mean(),
                'last_game_date': recent_games['GAME_DATE'].max().strftime('%Y-%m-%d')
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error fetching recent stats for player {player_id}: {e}")
            return {}
    
    def get_player_info(self, player_id: str) -> Dict[str, Any]:
        """
        Get detailed player information
        
        Args:
            player_id: NBA player ID
            
        Returns:
            Dictionary with player information
        """
        try:
            self._rate_limit()
            
            player_info = commonplayerinfo.CommonPlayerInfo(
                player_id=player_id
            ).get_data_frames()[0]
            
            if not player_info.empty:
                info = player_info.iloc[0]
                return {
                    'player_id': str(info['PERSON_ID']),
                    'player_name': info['DISPLAY_FIRST_LAST'],
                    'team_id': str(info['TEAM_ID']) if info['TEAM_ID'] else None,
                    'team_name': info['TEAM_NAME'] if info['TEAM_NAME'] else None,
                    'team_abbreviation': info['TEAM_ABBREVIATION'] if info['TEAM_ABBREVIATION'] else None,
                    'jersey_number': info['JERSEY'],
                    'position': info['POSITION'],
                    'height': info['HEIGHT'],
                    'weight': info['WEIGHT'],
                    'birth_date': info['BIRTHDATE'],
                    'country': info['COUNTRY'],
                    'school': info['SCHOOL'] if pd.notna(info['SCHOOL']) else None,
                    'draft_year': info['DRAFT_YEAR'] if pd.notna(info['DRAFT_YEAR']) else None,
                    'draft_round': info['DRAFT_ROUND'] if pd.notna(info['DRAFT_ROUND']) else None,
                    'draft_number': info['DRAFT_NUMBER'] if pd.notna(info['DRAFT_NUMBER']) else None
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching player info for {player_id}: {e}")
            return {}
    
    def get_player_career_stats(self, player_id: str) -> Dict[str, pd.DataFrame]:
        """
        Get player's career statistics
        
        Args:
            player_id: NBA player ID
            
        Returns:
            Dictionary with career stats DataFrames
        """
        try:
            self._rate_limit()
            
            career_stats = playercareerstats.PlayerCareerStats(
                player_id=player_id,
                per_mode36='PerGame'
            ).get_data_frames()
            
            return {
                'season_totals': career_stats[0] if len(career_stats) > 0 else pd.DataFrame(),
                'career_totals': career_stats[1] if len(career_stats) > 1 else pd.DataFrame()
            }
            
        except Exception as e:
            logger.error(f"Error fetching career stats for player {player_id}: {e}")
            return {'season_totals': pd.DataFrame(), 'career_totals': pd.DataFrame()}
    
    @staticmethod
    def get_all_teams() -> List[Dict[str, Any]]:
        """
        Get all NBA teams
        
        Returns:
            List of team dictionaries
        """
        nba_teams = teams.get_teams()
        return [{
            'team_id': str(team['id']),
            'team_name': team['full_name'],
            'team_abbreviation': team['abbreviation'],
            'team_city': team['city'],
            'team_state': team['state'] if 'state' in team else None,
            'year_founded': team['year_founded'] if 'year_founded' in team else None
        } for team in nba_teams]
    
    @staticmethod
    def search_player(player_name: str) -> Optional[Dict[str, Any]]:
        """
        Search for a player by name
        
        Args:
            player_name: Player name to search for
            
        Returns:
            Player dictionary if found, None otherwise
        """
        try:
            player_dict = players.find_players_by_full_name(player_name)
            if player_dict:
                player = player_dict[0]
                return {
                    'player_id': str(player['id']),
                    'player_name': player['full_name'],
                    'is_active': player['is_active']
                }
            return None
        except Exception as e:
            logger.error(f"Error searching for player {player_name}: {e}")
            return None