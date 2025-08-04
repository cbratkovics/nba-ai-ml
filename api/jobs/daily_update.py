"""
Daily data update job for NBA statistics
"""
import logging
import asyncio
from datetime import datetime, timedelta
from api.data.pipeline import NBADataPipeline
from api.features.player_features import FeatureEngineer
from database.connection import get_db_session
from api.models.game_data import GameLog, Player
from sqlalchemy import select, and_

logger = logging.getLogger(__name__)


class DailyUpdateJob:
    """Daily job to update NBA data and features"""
    
    def __init__(self):
        self.pipeline = NBADataPipeline()
        self.feature_engineer = FeatureEngineer()
    
    async def run(self):
        """Execute daily update tasks"""
        start_time = datetime.now()
        logger.info(f"Starting daily update job at {start_time}")
        
        try:
            # Step 1: Update yesterday's game logs
            await self._update_yesterday_games()
            
            # Step 2: Update player features for active players
            await self._update_player_features()
            
            # Step 3: Check prediction accuracy for yesterday's games
            await self._check_prediction_accuracy()
            
            # Step 4: Clean up old cache entries
            await self._cleanup_cache()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Daily update job completed in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in daily update job: {e}")
            raise
    
    async def _update_yesterday_games(self):
        """Update game logs for games played yesterday"""
        logger.info("Updating yesterday's game logs...")
        
        # This would use the actual NBA API to fetch yesterday's games
        # For now, it's a placeholder
        await self.pipeline.update_daily_games()
    
    async def _update_player_features(self):
        """Recalculate features for players who played recently"""
        logger.info("Updating player features...")
        
        yesterday = datetime.now().date() - timedelta(days=1)
        
        async with get_db_session() as session:
            # Get players who played in the last 3 days
            result = await session.execute(
                select(GameLog.player_id).distinct()
                .where(GameLog.game_date >= yesterday - timedelta(days=2))
            )
            player_ids = [row[0] for row in result]
            
            logger.info(f"Updating features for {len(player_ids)} players")
            
            # Update features for each player
            for player_id in player_ids:
                try:
                    features = await self.feature_engineer.calculate_player_features(
                        player_id,
                        datetime.now().date(),
                        "UNK"  # Placeholder opponent
                    )
                    await self.feature_engineer.save_features(
                        player_id,
                        datetime.now().date(),
                        features
                    )
                except Exception as e:
                    logger.error(f"Error updating features for player {player_id}: {e}")
                    continue
    
    async def _check_prediction_accuracy(self):
        """Check accuracy of predictions made for yesterday's games"""
        logger.info("Checking prediction accuracy...")
        
        # This would compare actual results with predictions
        # Placeholder for now
        pass
    
    async def _cleanup_cache(self):
        """Clean up old cache entries"""
        logger.info("Cleaning up old cache entries...")
        
        # This would remove cache entries older than 7 days
        # Placeholder for now
        pass


async def run_daily_update():
    """Entry point for daily update job"""
    job = DailyUpdateJob()
    await job.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_daily_update())