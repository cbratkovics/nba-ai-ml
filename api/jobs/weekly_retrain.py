"""
Weekly model retraining job
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from api.ml.train_models import NBAModelTrainer
from database.connection import get_db_session
from api.models.game_data import GameLog
from sqlalchemy import select, func
import joblib
import os

logger = logging.getLogger(__name__)


class WeeklyRetrainJob:
    """Weekly job to retrain models with latest data"""
    
    def __init__(self, model_path: str = "./models"):
        self.trainer = NBAModelTrainer(model_path)
        self.model_path = model_path
        self.improvement_threshold = 0.02  # 2% improvement required
    
    async def run(self):
        """Execute weekly retraining"""
        start_time = datetime.now()
        logger.info(f"Starting weekly retrain job at {start_time}")
        
        try:
            # Step 1: Check if we have enough new data
            new_games = await self._count_new_games()
            if new_games < 100:
                logger.info(f"Only {new_games} new games. Skipping retraining.")
                return
            
            # Step 2: Load current model performance
            current_performance = await self._get_current_model_performance()
            
            # Step 3: Train new models
            logger.info("Training new models with latest data...")
            features_df, targets = await self.trainer.prepare_training_data(
                seasons=['2023-24', '2024-25'],  # Use recent seasons
                min_games_per_player=20
            )
            
            results = self.trainer.train_models(features_df, targets)
            
            # Step 4: Compare performance
            should_deploy = await self._compare_model_performance(
                current_performance, 
                results
            )
            
            # Step 5: Deploy if improved
            if should_deploy:
                logger.info("New models show improvement. Deploying...")
                await self._deploy_new_models(results)
            else:
                logger.info("New models do not show significant improvement. Keeping current models.")
            
            # Step 6: Log results
            await self._log_training_results(results, should_deploy)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Weekly retrain job completed in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in weekly retrain job: {e}")
            raise
    
    async def _count_new_games(self) -> int:
        """Count games added in the last week"""
        one_week_ago = datetime.now().date() - timedelta(days=7)
        
        async with get_db_session() as session:
            result = await session.execute(
                select(func.count(GameLog.id))
                .where(GameLog.game_date >= one_week_ago)
            )
            count = result.scalar()
            
        return count or 0
    
    async def _get_current_model_performance(self) -> Dict[str, float]:
        """Get performance metrics of current models"""
        # Load from model metadata if available
        metadata_path = os.path.join(self.model_path, 'model_metadata.json')
        
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                # Return stored performance metrics
                return metadata.get('performance', {
                    'points_mae': 5.0,
                    'rebounds_mae': 3.0,
                    'assists_mae': 2.5
                })
        
        # Default performance if no metadata
        return {
            'points_mae': 5.0,
            'rebounds_mae': 3.0,
            'assists_mae': 2.5
        }
    
    async def _compare_model_performance(
        self, 
        current_performance: Dict[str, float],
        new_results: Dict[str, Any]
    ) -> bool:
        """Compare new model performance with current"""
        improvements = []
        
        for target in ['points', 'rebounds', 'assists']:
            if target in new_results:
                current_mae = current_performance.get(f'{target}_mae', float('inf'))
                new_mae = new_results[target]['all_models'][new_results[target]['best_model_name']]['mae']
                
                improvement = (current_mae - new_mae) / current_mae
                improvements.append(improvement)
                
                logger.info(f"{target}: Current MAE={current_mae:.2f}, New MAE={new_mae:.2f}, "
                          f"Improvement={improvement:.2%}")
        
        # Deploy if average improvement exceeds threshold
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        return avg_improvement >= self.improvement_threshold
    
    async def _deploy_new_models(self, results: Dict[str, Any]):
        """Deploy new models by saving them"""
        # Backup current models
        backup_dir = os.path.join(self.model_path, f"backup_{datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(backup_dir, exist_ok=True)
        
        for file in os.listdir(self.model_path):
            if file.endswith('.pkl'):
                src = os.path.join(self.model_path, file)
                dst = os.path.join(backup_dir, file)
                import shutil
                shutil.copy2(src, dst)
        
        # Save new models
        self.trainer.save_models()
        
        # Update metadata with performance metrics
        metadata = {
            'version': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'timestamp': datetime.now().isoformat(),
            'performance': {}
        }
        
        for target, result in results.items():
            best_model_metrics = result['all_models'][result['best_model_name']]
            metadata['performance'][f'{target}_mae'] = best_model_metrics['mae']
            metadata['performance'][f'{target}_r2'] = best_model_metrics['r2']
        
        import json
        metadata_path = os.path.join(self.model_path, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("New models deployed successfully")
    
    async def _log_training_results(self, results: Dict[str, Any], deployed: bool):
        """Log training results to database"""
        # This would log to a model_performance table
        # For now, just log to file
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'deployed': deployed,
            'results': {}
        }
        
        for target, result in results.items():
            best_model = result['best_model_name']
            metrics = result['all_models'][best_model]
            log_entry['results'][target] = {
                'model': best_model,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2']
            }
        
        # Append to log file
        import json
        log_path = os.path.join(self.model_path, 'training_log.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


async def run_weekly_retrain():
    """Entry point for weekly retrain job"""
    job = WeeklyRetrainJob()
    await job.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_weekly_retrain())