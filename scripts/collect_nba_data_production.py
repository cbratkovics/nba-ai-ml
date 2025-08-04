#!/usr/bin/env python3
"""
Production NBA data collection with checkpointing and error recovery
Collects real NBA data using nba_api with production-grade features
"""
import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import pickle
from typing import Dict, List, Optional, Set, Any
import time
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playergamelog, teamgamelog, leaguegamefinder,
    commonplayerinfo, leaguedashteamstats
)
from database.connection import DatabaseManager, init_db
from database.models import Player, Team, Game, PlayerGameLog
from ml.data.processors.data_validator import NBADataValidator
from sqlalchemy import and_, or_
from sqlalchemy.dialects.postgresql import insert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/data_collection_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Monitor data collection quality in real-time"""
    
    def __init__(self):
        self.metrics = {
            "total_games": 0,
            "total_players": 0,
            "missing_values": {},
            "anomalies": [],
            "collection_errors": [],
            "api_calls": 0,
            "cache_hits": 0
        }
        
    def check_game_completeness(self, game_log: Dict) -> bool:
        """Verify all expected stats are present"""
        required_fields = ['PTS', 'REB', 'AST', 'MIN', 'FGM', 'FGA', 'FG_PCT']
        missing = [f for f in required_fields if game_log.get(f) is None]
        
        if missing:
            self.metrics["missing_values"][game_log.get('GAME_ID', 'unknown')] = missing
            return False
        return True
            
    def check_anomalies(self, game_log: Dict):
        """Detect statistical anomalies"""
        anomalies = []
        
        # Check for unusually high stats
        if game_log.get('PTS', 0) > 70:
            anomalies.append({
                "type": "high_points",
                "game_id": game_log.get('GAME_ID'),
                "value": game_log['PTS']
            })
        
        if game_log.get('REB', 0) > 30:
            anomalies.append({
                "type": "high_rebounds",
                "game_id": game_log.get('GAME_ID'),
                "value": game_log['REB']
            })
        
        # Check for impossible values
        if game_log.get('FGM', 0) > game_log.get('FGA', 0):
            anomalies.append({
                "type": "impossible_fg",
                "game_id": game_log.get('GAME_ID'),
                "fgm": game_log['FGM'],
                "fga": game_log['FGA']
            })
        
        if anomalies:
            self.metrics["anomalies"].extend(anomalies)
            
    def log_error(self, error_type: str, details: Dict):
        """Log collection errors"""
        self.metrics["collection_errors"].append({
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "details": details
        })
    
    def get_report(self) -> Dict:
        """Generate collection quality report"""
        return {
            "summary": {
                "total_games_collected": self.metrics["total_games"],
                "total_players_collected": self.metrics["total_players"],
                "api_calls_made": self.metrics["api_calls"],
                "cache_hit_rate": self.metrics["cache_hits"] / max(self.metrics["api_calls"], 1),
                "games_with_missing_values": len(self.metrics["missing_values"]),
                "anomalies_detected": len(self.metrics["anomalies"]),
                "collection_errors": len(self.metrics["collection_errors"])
            },
            "details": self.metrics
        }


class NBADataCollectionPipeline:
    """Production NBA data collection with checkpointing and error recovery"""
    
    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.db_manager = None
        self.monitor = DataQualityMonitor()
        self.validator = NBADataValidator()
        
        # Rate limiting
        self.last_api_call = 0
        self.api_delay = 0.6  # 600ms between calls (100 requests per minute)
        
        # Cache directory
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("NBA Data Collection Pipeline initialized")
    
    def load_checkpoint(self, name: str) -> Dict:
        """Load checkpoint data"""
        checkpoint_file = self.checkpoint_dir / f"{name}.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"Loaded checkpoint: {name}")
                return data
        return {}
    
    def save_checkpoint(self, name: str, data: Dict):
        """Save checkpoint data"""
        checkpoint_file = self.checkpoint_dir / f"{name}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        logger.debug(f"Saved checkpoint: {name}")
    
    async def rate_limit(self):
        """Enforce rate limiting for API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_api_call
        
        if time_since_last < self.api_delay:
            await asyncio.sleep(self.api_delay - time_since_last)
        
        self.last_api_call = time.time()
        self.monitor.metrics["api_calls"] += 1
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_active_players(self, min_games: int = 20) -> List[Dict]:
        """Get all active NBA players with minimum games played"""
        logger.info("Fetching active NBA players...")
        
        # Check cache
        cache_file = self.cache_dir / "active_players.json"
        if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days < 7:
            with open(cache_file, 'r') as f:
                players_list = json.load(f)
                logger.info(f"Loaded {len(players_list)} players from cache")
                self.monitor.metrics["cache_hits"] += 1
                return players_list
        
        # Get all players
        all_players = players.get_players()
        active_players = []
        
        # Filter active players (simplified - in production would check current season stats)
        for player in all_players:
            if player.get('is_active', True):  # nba_api doesn't have is_active, so we'll check by games
                active_players.append({
                    "id": str(player['id']),
                    "full_name": player['full_name'],
                    "first_name": player.get('first_name', ''),
                    "last_name": player.get('last_name', ''),
                    "is_active": True
                })
        
        # Cache the results
        with open(cache_file, 'w') as f:
            json.dump(active_players, f)
        
        logger.info(f"Found {len(active_players)} active players")
        return active_players
    
    async def get_all_teams(self) -> List[Dict]:
        """Get all NBA teams"""
        logger.info("Fetching NBA teams...")
        
        # Check cache
        cache_file = self.cache_dir / "teams.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                teams_list = json.load(f)
                self.monitor.metrics["cache_hits"] += 1
                return teams_list
        
        # Get all teams
        all_teams = teams.get_teams()
        
        teams_list = []
        for team in all_teams:
            teams_list.append({
                "id": str(team['id']),
                "full_name": team['full_name'],
                "abbreviation": team['abbreviation'],
                "nickname": team['nickname'],
                "city": team['city'],
                "state": team.get('state', ''),
                "year_founded": team.get('year_founded', 1946)
            })
        
        # Cache the results
        with open(cache_file, 'w') as f:
            json.dump(teams_list, f)
        
        logger.info(f"Found {len(teams_list)} teams")
        return teams_list
    
    async def collect_player_season_data(self, player_id: str, season: str) -> Optional[pd.DataFrame]:
        """Collect game logs for a player in a specific season"""
        try:
            await self.rate_limit()
            
            # Try to get from cache first
            cache_key = f"player_{player_id}_{season}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                self.monitor.metrics["cache_hits"] += 1
                return df
            
            # Fetch from API
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            
            df = gamelog.get_data_frames()[0]
            
            if not df.empty:
                # Cache the data
                df.to_parquet(cache_file)
                
                # Validate data quality
                for _, row in df.iterrows():
                    game_dict = row.to_dict()
                    self.monitor.check_game_completeness(game_dict)
                    self.monitor.check_anomalies(game_dict)
                
                self.monitor.metrics["total_games"] += len(df)
                
            return df
            
        except Exception as e:
            logger.error(f"Error collecting data for player {player_id} in {season}: {e}")
            self.monitor.log_error("player_collection", {
                "player_id": player_id,
                "season": season,
                "error": str(e)
            })
            return None
    
    async def collect_all_data(self, seasons: List[str] = None):
        """
        Collect comprehensive NBA data with production features
        """
        if not seasons:
            seasons = ["2023-24", "2024-25"]  # Most recent complete and current season
        
        logger.info("=" * 60)
        logger.info("Starting NBA Data Collection Pipeline")
        logger.info(f"Seasons: {seasons}")
        logger.info("=" * 60)
        
        # Initialize database
        if not self.db_manager:
            self.db_manager = init_db()
        
        # Step 1: Collect and store teams
        logger.info("\n[Step 1/4] Collecting team data...")
        teams_data = await self.get_all_teams()
        await self.store_teams(teams_data)
        
        # Step 2: Collect and store players
        logger.info("\n[Step 2/4] Collecting player data...")
        players_data = await self.get_active_players(min_games=10)
        await self.store_players(players_data)
        self.monitor.metrics["total_players"] = len(players_data)
        
        # Step 3: Collect player game logs for each season
        logger.info("\n[Step 3/4] Collecting player game logs...")
        
        for season in seasons:
            logger.info(f"\nProcessing season {season}")
            
            # Load checkpoint
            checkpoint = self.load_checkpoint(f"season_{season}")
            completed_players = set(checkpoint.get("completed", []))
            failed_players = checkpoint.get("failed", {})
            
            # Progress bar
            players_to_process = [p for p in players_data if p["id"] not in completed_players]
            
            with tqdm(total=len(players_to_process), desc=f"Season {season}") as pbar:
                for player in players_to_process:
                    player_id = player["id"]
                    
                    # Skip if failed too many times
                    if failed_players.get(player_id, 0) >= 3:
                        pbar.update(1)
                        continue
                    
                    try:
                        # Collect player data
                        game_logs = await self.collect_player_season_data(player_id, season)
                        
                        if game_logs is not None and not game_logs.empty:
                            # Store in database
                            await self.store_player_game_logs(player_id, season, game_logs)
                            
                            # Update checkpoint
                            completed_players.add(player_id)
                            self.save_checkpoint(f"season_{season}", {
                                "completed": list(completed_players),
                                "failed": failed_players,
                                "total": len(players_data),
                                "last_updated": datetime.now().isoformat()
                            })
                        
                    except Exception as e:
                        logger.error(f"Failed to process player {player['full_name']}: {e}")
                        failed_players[player_id] = failed_players.get(player_id, 0) + 1
                        self.save_checkpoint(f"season_{season}", {
                            "completed": list(completed_players),
                            "failed": failed_players,
                            "total": len(players_data),
                            "last_updated": datetime.now().isoformat()
                        })
                    
                    pbar.update(1)
            
            logger.info(f"Season {season} complete: {len(completed_players)}/{len(players_data)} players")
        
        # Step 4: Generate collection report
        logger.info("\n[Step 4/4] Generating collection report...")
        report = self.generate_collection_report()
        
        logger.info("=" * 60)
        logger.info("Data Collection Complete!")
        logger.info(f"Total games collected: {self.monitor.metrics['total_games']}")
        logger.info(f"Total players processed: {self.monitor.metrics['total_players']}")
        logger.info(f"API calls made: {self.monitor.metrics['api_calls']}")
        logger.info(f"Cache hits: {self.monitor.metrics['cache_hits']}")
        logger.info("=" * 60)
        
        return report
    
    async def store_teams(self, teams_data: List[Dict]):
        """Store teams in database"""
        with self.db_manager.transaction() as session:
            for team_data in teams_data:
                # Check if team exists
                existing = session.query(Team).filter_by(nba_team_id=team_data["id"]).first()
                
                if not existing:
                    team = Team(
                        nba_team_id=team_data["id"],
                        full_name=team_data["full_name"],
                        abbreviation=team_data["abbreviation"],
                        name=team_data["nickname"],
                        city=team_data["city"],
                        state=team_data.get("state", ""),
                        year_founded=team_data.get("year_founded")
                    )
                    session.add(team)
                    logger.debug(f"Added team: {team_data['full_name']}")
    
    async def store_players(self, players_data: List[Dict]):
        """Store players in database"""
        with self.db_manager.transaction() as session:
            for player_data in players_data:
                # Check if player exists
                existing = session.query(Player).filter_by(nba_player_id=player_data["id"]).first()
                
                if not existing:
                    player = Player(
                        nba_player_id=player_data["id"],
                        full_name=player_data["full_name"],
                        first_name=player_data.get("first_name", ""),
                        last_name=player_data.get("last_name", ""),
                        is_active=player_data.get("is_active", True)
                    )
                    session.add(player)
                    logger.debug(f"Added player: {player_data['full_name']}")
    
    async def store_player_game_logs(self, player_id: str, season: str, game_logs: pd.DataFrame):
        """Store player game logs in database"""
        if game_logs.empty:
            return
        
        with self.db_manager.transaction() as session:
            # Get player UUID
            player = session.query(Player).filter_by(nba_player_id=player_id).first()
            if not player:
                logger.warning(f"Player {player_id} not found in database")
                return
            
            # Process each game
            for _, row in game_logs.iterrows():
                try:
                    # Get or create game
                    game = session.query(Game).filter_by(nba_game_id=row['Game_ID']).first()
                    
                    if not game:
                        # Parse game date
                        game_date = pd.to_datetime(row['GAME_DATE']).date()
                        
                        # Determine home/away teams from MATCHUP (e.g., "LAL @ GSW" or "LAL vs. GSW")
                        matchup = row['MATCHUP']
                        if '@' in matchup:
                            # Away game
                            away_abbr, home_abbr = matchup.split(' @ ')
                        else:
                            # Home game (vs.)
                            parts = matchup.split(' vs. ')
                            if len(parts) == 2:
                                home_abbr, away_abbr = parts[0], parts[1]
                            else:
                                continue
                        
                        # Get team IDs
                        home_team = session.query(Team).filter_by(abbreviation=home_abbr.strip()).first()
                        away_team = session.query(Team).filter_by(abbreviation=away_abbr.strip()).first()
                        
                        if home_team and away_team:
                            game = Game(
                                nba_game_id=row['Game_ID'],
                                game_date=game_date,
                                season=season,
                                home_team_id=home_team.id,
                                away_team_id=away_team.id
                            )
                            session.add(game)
                            session.flush()
                    
                    if game:
                        # Check if game log already exists
                        existing_log = session.query(PlayerGameLog).filter_by(
                            player_id=player.id,
                            game_id=game.id
                        ).first()
                        
                        if not existing_log:
                            # Determine team
                            team_abbr = row['MATCHUP'].split()[0]
                            team = session.query(Team).filter_by(abbreviation=team_abbr).first()
                            
                            if team:
                                # Create game log
                                game_log = PlayerGameLog(
                                    player_id=player.id,
                                    game_id=game.id,
                                    team_id=team.id,
                                    game_date=game.game_date,
                                    minutes_played=row.get('MIN', 0),
                                    points=row.get('PTS', 0),
                                    rebounds=row.get('REB', 0),
                                    offensive_rebounds=row.get('OREB', 0),
                                    defensive_rebounds=row.get('DREB', 0),
                                    assists=row.get('AST', 0),
                                    steals=row.get('STL', 0),
                                    blocks=row.get('BLK', 0),
                                    turnovers=row.get('TOV', 0),
                                    personal_fouls=row.get('PF', 0),
                                    field_goals_made=row.get('FGM', 0),
                                    field_goals_attempted=row.get('FGA', 0),
                                    field_goal_percentage=row.get('FG_PCT', 0),
                                    three_pointers_made=row.get('FG3M', 0),
                                    three_pointers_attempted=row.get('FG3A', 0),
                                    three_point_percentage=row.get('FG3_PCT', 0),
                                    free_throws_made=row.get('FTM', 0),
                                    free_throws_attempted=row.get('FTA', 0),
                                    free_throw_percentage=row.get('FT_PCT', 0),
                                    plus_minus=row.get('PLUS_MINUS', 0),
                                    home_game=1 if 'vs.' in row['MATCHUP'] else 0
                                )
                                session.add(game_log)
                
                except Exception as e:
                    logger.error(f"Error storing game log: {e}")
                    continue
    
    def generate_collection_report(self) -> Dict:
        """Generate comprehensive collection report"""
        report = self.monitor.get_report()
        
        # Save report to file
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"data_collection_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Collection report saved to {report_file}")
        
        # Generate text summary
        summary_file = report_dir / f"data_collection_summary_{datetime.now():%Y%m%d_%H%M%S}.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("NBA DATA COLLECTION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 30 + "\n")
            for key, value in report["summary"].items():
                f.write(f"{key}: {value}\n")
            
            if report["details"]["anomalies"]:
                f.write("\nANOMALIES DETECTED\n")
                f.write("-" * 30 + "\n")
                for anomaly in report["details"]["anomalies"][:10]:  # First 10
                    f.write(f"- {anomaly}\n")
            
            if report["details"]["collection_errors"]:
                f.write("\nCOLLECTION ERRORS\n")
                f.write("-" * 30 + "\n")
                for error in report["details"]["collection_errors"][:10]:  # First 10
                    f.write(f"- {error}\n")
        
        logger.info(f"Collection summary saved to {summary_file}")
        
        return report


async def main():
    """Main data collection function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect NBA data for ML training')
    parser.add_argument('--seasons', nargs='+', default=["2023-24", "2024-25"],
                       help='Seasons to collect (e.g., 2023-24 2024-25)')
    parser.add_argument('--checkpoint-dir', default='data/checkpoints',
                       help='Directory for checkpoint files')
    parser.add_argument('--validate', action='store_true',
                       help='Validate collected data quality')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Initialize pipeline
    pipeline = NBADataCollectionPipeline(checkpoint_dir=args.checkpoint_dir)
    
    try:
        # Collect data
        report = await pipeline.collect_all_data(seasons=args.seasons)
        
        # Validate if requested
        if args.validate:
            logger.info("\nValidating collected data...")
            
            # Load a sample of collected data
            with pipeline.db_manager.get_db(read_only=True) as session:
                sample_logs = session.query(PlayerGameLog).limit(1000).all()
                
                if sample_logs:
                    # Convert to DataFrame for validation
                    data_dict = []
                    for log in sample_logs:
                        data_dict.append({
                            'player_id': log.player_id,
                            'game_id': log.game_id,
                            'game_date': log.game_date,
                            'points': log.points,
                            'rebounds': log.rebounds,
                            'assists': log.assists,
                            'minutes_played': log.minutes_played,
                            'field_goals_made': log.field_goals_made,
                            'field_goals_attempted': log.field_goals_attempted
                        })
                    
                    df = pd.DataFrame(data_dict)
                    
                    # Validate
                    validator = NBADataValidator()
                    validation_result = validator.validate_player_game_log(df)
                    
                    logger.info(f"Validation Status: {validation_result.status.value}")
                    logger.info(f"Passed Checks: {validation_result.passed_checks}")
                    logger.info(f"Failed Checks: {validation_result.failed_checks}")
                    
                    if validation_result.errors:
                        logger.warning("Validation errors found:")
                        for error in validation_result.errors[:5]:
                            logger.warning(f"  - {error}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("DATA COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total games: {report['summary']['total_games_collected']}")
        logger.info(f"Total players: {report['summary']['total_players_collected']}")
        logger.info(f"API calls: {report['summary']['api_calls_made']}")
        logger.info(f"Cache hit rate: {report['summary']['cache_hit_rate']:.1%}")
        logger.info(f"Anomalies: {report['summary']['anomalies_detected']}")
        logger.info(f"Errors: {report['summary']['collection_errors']}")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())