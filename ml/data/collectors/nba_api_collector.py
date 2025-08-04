"""
Production-grade NBA data collector using free nba_api
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import (
    leaguegamefinder,
    playergamelog,
    playerprofilev2,
    teamgamelog,
    boxscoretraditionalv2,
    leaguedashteamstats
)
from nba_api.stats.static import players, teams
import redis
import json
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls: int = 300, period: int = 60):
        self.calls = calls
        self.period = period
        self.timestamps = []
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < self.period]
        
        if len(self.timestamps) >= self.calls:
            sleep_time = self.period - (now - self.timestamps[0]) + 1
            logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
            await asyncio.sleep(sleep_time)
        
        self.timestamps.append(now)


class RedisCache:
    """Redis cache for API responses"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = 3600  # 1 hour
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached data"""
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None
    
    def set(self, key: str, value: Dict, ttl: int = None):
        """Cache data with TTL"""
        ttl = ttl or self.default_ttl
        self.client.setex(key, ttl, json.dumps(value))


class NBADataCollector:
    """Production-grade NBA data collector using free APIs"""
    
    def __init__(self, use_cache: bool = True):
        self.rate_limiter = RateLimiter(calls=300, period=60)
        self.cache = RedisCache() if use_cache else None
        self.player_dict = players.get_players()
        self.team_dict = teams.get_teams()
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def collect_player_stats(self, 
                                 player_id: str,
                                 season: str = "2024-25",
                                 season_type: str = "Regular Season") -> pd.DataFrame:
        """
        Collect comprehensive player statistics
        
        Args:
            player_id: NBA player ID
            season: Season string (e.g., "2024-25")
            season_type: "Regular Season" or "Playoffs"
            
        Returns:
            DataFrame with player game logs
        """
        cache_key = f"player_stats:{player_id}:{season}:{season_type}"
        
        if self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Cache hit for player {player_id}")
                return pd.DataFrame(cached_data)
        
        await self.rate_limiter.wait_if_needed()
        
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star=season_type
            )
            
            df = gamelog.get_data_frames()[0]
            
            # Add calculated features
            df['REST_DAYS'] = self._calculate_rest_days(df)
            df['HOME_GAME'] = df['MATCHUP'].str.contains('vs').astype(int)
            df['BACK_TO_BACK'] = (df['REST_DAYS'] == 0).astype(int)
            
            if self.cache:
                self.cache.set(cache_key, df.to_dict('records'))
            
            logger.info(f"Collected {len(df)} games for player {player_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting player stats: {e}")
            raise
    
    async def collect_team_stats(self,
                                team_id: str,
                                season: str = "2024-25") -> pd.DataFrame:
        """Collect team statistics"""
        cache_key = f"team_stats:{team_id}:{season}"
        
        if self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return pd.DataFrame(cached_data)
        
        await self.rate_limiter.wait_if_needed()
        
        try:
            team_log = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season
            )
            
            df = team_log.get_data_frames()[0]
            
            # Add pace and efficiency metrics
            df['PACE'] = self._calculate_pace(df)
            df['OFF_RATING'] = (df['PTS'] / df['MIN']) * 48 * 5
            df['DEF_RATING'] = (df['PTS_OPP'] / df['MIN']) * 48 * 5 if 'PTS_OPP' in df else None
            
            if self.cache:
                self.cache.set(cache_key, df.to_dict('records'))
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting team stats: {e}")
            raise
    
    async def collect_matchup_data(self,
                                  team1_id: str,
                                  team2_id: str,
                                  last_n_games: int = 10) -> pd.DataFrame:
        """Collect head-to-head matchup data"""
        await self.rate_limiter.wait_if_needed()
        
        try:
            games = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team1_id
            ).get_data_frames()[0]
            
            # Filter for matchups against team2
            matchup_games = games[games['MATCHUP'].str.contains(
                teams.find_team_by_id(team2_id)['abbreviation']
            )].head(last_n_games)
            
            return matchup_games
            
        except Exception as e:
            logger.error(f"Error collecting matchup data: {e}")
            raise
    
    async def collect_live_data(self, game_id: str) -> Dict[str, Any]:
        """Collect live/recent game data"""
        cache_key = f"live_data:{game_id}"
        
        if self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return cached_data
        
        await self.rate_limiter.wait_if_needed()
        
        try:
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            
            player_stats = boxscore.player_stats.get_data_frame()
            team_stats = boxscore.team_stats.get_data_frame()
            
            result = {
                'player_stats': player_stats.to_dict('records'),
                'team_stats': team_stats.to_dict('records'),
                'game_id': game_id,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.cache:
                self.cache.set(cache_key, result, ttl=300)  # 5 minute TTL for live data
            
            return result
            
        except Exception as e:
            logger.error(f"Error collecting live data: {e}")
            raise
    
    async def collect_season_averages(self,
                                     player_id: str,
                                     season: str = "2024-25") -> Dict[str, float]:
        """Calculate season averages for a player"""
        df = await self.collect_player_stats(player_id, season)
        
        if df.empty:
            return {}
        
        averages = {
            'PPG': df['PTS'].mean(),
            'RPG': df['REB'].mean(),
            'APG': df['AST'].mean(),
            'SPG': df['STL'].mean(),
            'BPG': df['BLK'].mean(),
            'FG_PCT': df['FG_PCT'].mean(),
            'FG3_PCT': df['FG3_PCT'].mean(),
            'FT_PCT': df['FT_PCT'].mean(),
            'MIN': df['MIN'].mean(),
            'PLUS_MINUS': df['PLUS_MINUS'].mean()
        }
        
        return averages
    
    async def collect_advanced_stats(self,
                                    player_id: str,
                                    season: str = "2024-25") -> Dict[str, float]:
        """Calculate advanced statistics"""
        await self.rate_limiter.wait_if_needed()
        
        try:
            profile = playerprofilev2.PlayerProfileV2(
                player_id=player_id,
                per_mode36="PerGame"
            )
            
            career_stats = profile.season_totals_regular_season.get_data_frame()
            current_season = career_stats[career_stats['SEASON_ID'] == season].iloc[0]
            
            # Calculate advanced metrics
            ts_pct = self._calculate_true_shooting(
                current_season['PTS'],
                current_season['FGA'],
                current_season['FTA']
            )
            
            usage_rate = self._calculate_usage_rate(
                current_season['FGA'],
                current_season['FTA'],
                current_season['TOV'],
                current_season['MIN']
            )
            
            return {
                'TRUE_SHOOTING_PCT': ts_pct,
                'USAGE_RATE': usage_rate,
                'PER': self._estimate_per(current_season)
            }
            
        except Exception as e:
            logger.error(f"Error collecting advanced stats: {e}")
            return {}
    
    def _calculate_rest_days(self, df: pd.DataFrame) -> List[int]:
        """Calculate rest days between games"""
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        rest_days = []
        
        for i in range(len(df)):
            if i == len(df) - 1:
                rest_days.append(3)  # Default for first game of season
            else:
                days_diff = (df.iloc[i]['GAME_DATE'] - df.iloc[i+1]['GAME_DATE']).days - 1
                rest_days.append(max(0, days_diff))
        
        return rest_days
    
    def _calculate_pace(self, df: pd.DataFrame) -> pd.Series:
        """Calculate team pace"""
        possessions = 0.5 * ((df['FGA'] + 0.4 * df['FTA'] - 1.07 * 
                             (df['OREB'] / (df['OREB'] + df['DREB'])) * 
                             (df['FGA'] - df['FGM']) + df['TOV']))
        pace = 48 * possessions / df['MIN']
        return pace
    
    def _calculate_true_shooting(self, pts: float, fga: float, fta: float) -> float:
        """Calculate true shooting percentage"""
        if (2 * fga + 0.44 * fta) == 0:
            return 0
        return pts / (2 * (fga + 0.44 * fta))
    
    def _calculate_usage_rate(self, fga: float, fta: float, tov: float, min_played: float) -> float:
        """Calculate usage rate"""
        if min_played == 0:
            return 0
        return ((fga + 0.44 * fta + tov) * 48) / min_played
    
    def _estimate_per(self, row: pd.Series) -> float:
        """Estimate Player Efficiency Rating"""
        # Simplified PER calculation
        per = (row['PTS'] + row['REB'] + row['AST'] + row['STL'] + row['BLK'] -
              (row['FGA'] - row['FGM']) - (row['FTA'] - row['FTM']) - row['TOV'])
        return per / row['MIN'] * 15 if row['MIN'] > 0 else 0


async def main():
    """Example usage"""
    collector = NBADataCollector(use_cache=False)  # Disable cache for testing
    
    # Example: Collect Nikola Jokic stats
    jokic_id = "203999"
    stats = await collector.collect_player_stats(jokic_id, "2023-24")
    print(f"Collected {len(stats)} games for Jokic")
    print(stats[['GAME_DATE', 'PTS', 'REB', 'AST']].head())
    
    # Get season averages
    averages = await collector.collect_season_averages(jokic_id, "2023-24")
    print(f"\nSeason Averages: {averages}")


if __name__ == "__main__":
    asyncio.run(main())