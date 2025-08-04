#!/usr/bin/env python3
"""
Historical NBA data collection script
"""
import asyncio
import argparse
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.data.collectors.nba_api_collector import NBADataCollector
from ml.data.processors.feature_engineer import NBAFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HistoricalDataCollector:
    """Collect and process historical NBA data"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.collector = NBADataCollector(use_cache=True)
        self.feature_engineer = NBAFeatureEngineer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def collect_player_data(self, player_ids: list, seasons: list):
        """Collect data for multiple players and seasons"""
        all_data = []
        
        for season in seasons:
            logger.info(f"Collecting data for season {season}")
            
            for player_id in player_ids:
                try:
                    logger.info(f"Processing player {player_id}")
                    
                    # Collect player stats
                    player_stats = await self.collector.collect_player_stats(
                        player_id, season
                    )
                    
                    if not player_stats.empty:
                        player_stats['PLAYER_ID'] = player_id
                        player_stats['SEASON'] = season
                        all_data.append(player_stats)
                        
                        logger.info(f"Collected {len(player_stats)} games for player {player_id}")
                    
                    # Rate limiting delay
                    await asyncio.sleep(0.6)  # 100 requests per minute
                    
                except Exception as e:
                    logger.error(f"Error collecting data for player {player_id}: {e}")
                    continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()
    
    def save_raw_data(self, data: pd.DataFrame, filename: str):
        """Save raw data to file"""
        filepath = self.output_dir / filename
        data.to_parquet(filepath, index=False)
        logger.info(f"Saved raw data to {filepath}")
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create feature-engineered dataset"""
        logger.info("Creating features...")
        
        # Group by player for feature engineering
        featured_data = []
        
        for player_id in data['PLAYER_ID'].unique():
            player_data = data[data['PLAYER_ID'] == player_id].copy()
            
            # Create features for this player
            try:
                features = self.feature_engineer.create_features(player_data)
                featured_data.append(features)
            except Exception as e:
                logger.error(f"Error creating features for player {player_id}: {e}")
                continue
        
        if featured_data:
            return pd.concat(featured_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def save_processed_data(self, data: pd.DataFrame, filename: str):
        """Save processed data to file"""
        filepath = self.output_dir / filename
        data.to_parquet(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
        
        # Also save feature names
        feature_cols = [col for col in data.columns if col not in 
                       ['GAME_DATE', 'GAME_ID', 'PLAYER_ID', 'TEAM_ID', 'SEASON']]
        
        feature_file = self.output_dir / 'feature_names.txt'
        with open(feature_file, 'w') as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        
        logger.info(f"Saved feature names to {feature_file}")

def get_top_players():
    """Get list of top NBA players for data collection"""
    # Top players by popularity/relevance for demo
    return [
        "203999",  # Nikola Jokic
        "2544",    # LeBron James
        "201939",  # Stephen Curry
        "1628369", # Jayson Tatum
        "1629029", # Jaylen Brown
        "203507",  # Giannis Antetokounmpo
        "201142",  # Kevin Durant
        "203081",  # Damian Lillard
        "1627732", # Trae Young
        "1628983", # Shai Gilgeous-Alexander
        "1630173", # Ja Morant
        "1629627", # Zion Williamson
        "1630162", # Anthony Edwards
        "203076",  # Anthony Davis
        "202681",  # Kyrie Irving
        "203954",  # Joel Embiid
        "1627783", # Pascal Siakam
        "203932",  # Aaron Gordon
        "203935",  # Marcus Smart
        "1628368"  # De'Aaron Fox
    ]

async def main():
    """Main data collection function"""
    parser = argparse.ArgumentParser(description='Collect historical NBA data')
    parser.add_argument('--seasons', nargs='+', default=['2023-24', '2024-25'],
                       help='Seasons to collect (e.g., 2023-24 2024-25)')
    parser.add_argument('--players', nargs='+', 
                       help='Specific player IDs to collect (default: top 20 players)')
    parser.add_argument('--output-dir', default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--features-only', action='store_true',
                       help='Only create features from existing raw data')
    
    args = parser.parse_args()
    
    collector = HistoricalDataCollector(args.output_dir)
    
    if args.features_only:
        # Load existing raw data and create features
        raw_file = Path(args.output_dir) / 'raw_player_data.parquet'
        if raw_file.exists():
            logger.info(f"Loading existing data from {raw_file}")
            raw_data = pd.read_parquet(raw_file)
            
            # Create features
            featured_data = collector.create_features(raw_data)
            
            if not featured_data.empty:
                collector.save_processed_data(featured_data, 'featured_player_data.parquet')
                logger.info(f"Feature engineering completed. Created {len(featured_data)} samples with {len(featured_data.columns)} features")
            else:
                logger.error("Feature engineering failed")
        else:
            logger.error(f"Raw data file not found: {raw_file}")
        return
    
    # Use specified players or default top players
    player_ids = args.players if args.players else get_top_players()
    
    logger.info(f"Starting data collection for {len(player_ids)} players and {len(args.seasons)} seasons")
    logger.info(f"Players: {player_ids}")
    logger.info(f"Seasons: {args.seasons}")
    
    try:
        # Collect raw data
        raw_data = await collector.collect_player_data(player_ids, args.seasons)
        
        if raw_data.empty:
            logger.error("No data collected")
            return
        
        logger.info(f"Collected {len(raw_data)} total game records")
        
        # Save raw data
        collector.save_raw_data(raw_data, 'raw_player_data.parquet')
        
        # Create features
        featured_data = collector.create_features(raw_data)
        
        if not featured_data.empty:
            # Save featured data
            collector.save_processed_data(featured_data, 'featured_player_data.parquet')
            
            logger.info(f"Data collection completed successfully!")
            logger.info(f"- Raw data: {len(raw_data)} samples")
            logger.info(f"- Featured data: {len(featured_data)} samples with {len(featured_data.columns)} features")
            logger.info(f"- Output directory: {args.output_dir}")
        else:
            logger.error("Feature engineering failed")
    
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())