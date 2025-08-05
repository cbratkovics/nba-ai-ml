#!/usr/bin/env python3
"""
Production setup script for NBA ML Platform
Automated setup with progress tracking and resume capability
"""
import asyncio
import logging
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text, create_engine
from sqlalchemy.exc import OperationalError
import pandas as pd
from tqdm import tqdm
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"setup_log_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class ProductionSetup:
    """Automated production setup with resume capability"""
    
    def __init__(self):
        self.state_file = Path(".setup_state.json")
        self.state = self._load_state()
        self.start_time = time.time()
        
    def _load_state(self) -> Dict[str, Any]:
        """Load setup state for resume capability"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "steps_completed": [],
            "current_step": None,
            "started_at": datetime.now().isoformat()
        }
    
    def _save_state(self):
        """Save current state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def _mark_step_complete(self, step: str):
        """Mark a step as completed"""
        if step not in self.state["steps_completed"]:
            self.state["steps_completed"].append(step)
        self.state["current_step"] = None
        self._save_state()
    
    def _is_step_complete(self, step: str) -> bool:
        """Check if a step is already completed"""
        return step in self.state["steps_completed"]
    
    def _print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{text:^60}")
        print(f"{'='*60}{Style.RESET_ALL}\n")
    
    def _print_success(self, text: str):
        """Print success message"""
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")
    
    def _print_error(self, text: str):
        """Print error message"""
        print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")
    
    def _print_warning(self, text: str):
        """Print warning message"""
        print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")
    
    async def run(self):
        """Run the complete setup process"""
        self._print_header("NBA ML Platform Production Setup")
        
        # Check for resume
        if self.state["steps_completed"]:
            self._print_warning(f"Resuming setup. Already completed: {', '.join(self.state['steps_completed'])}")
        
        try:
            # Step 1: Environment check
            if not self._is_step_complete("env_check"):
                await self._check_environment()
                self._mark_step_complete("env_check")
            
            # Step 2: Database setup
            if not self._is_step_complete("database_setup"):
                await self._setup_database()
                self._mark_step_complete("database_setup")
            
            # Step 3: Load NBA data
            if not self._is_step_complete("data_load"):
                await self._load_nba_data()
                self._mark_step_complete("data_load")
            
            # Step 4: Train models
            if not self._is_step_complete("model_training"):
                await self._train_models()
                self._mark_step_complete("model_training")
            
            # Step 5: Validate setup
            if not self._is_step_complete("validation"):
                await self._validate_setup()
                self._mark_step_complete("validation")
            
            # Step 6: Generate report
            await self._generate_report()
            
            # Cleanup
            self.state_file.unlink(missing_ok=True)
            
            elapsed = time.time() - self.start_time
            self._print_header("Setup Complete!")
            self._print_success(f"Total time: {elapsed/60:.1f} minutes")
            self._print_success(f"Log file: {log_file}")
            
        except Exception as e:
            self._print_error(f"Setup failed: {e}")
            logger.error(f"Setup failed: {e}", exc_info=True)
            self._print_warning("Run this script again to resume from the last successful step")
            sys.exit(1)
    
    async def _check_environment(self):
        """Check environment and dependencies"""
        self._print_header("Step 1: Environment Check")
        
        checks = {
            "Python version": sys.version.split()[0],
            "PostgreSQL": None,
            "Redis": None,
            "Required packages": None
        }
        
        # Check database connection
        try:
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL not set")
            
            engine = create_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                checks["PostgreSQL"] = result.scalar()
                self._print_success("Database connection successful")
        except Exception as e:
            self._print_error(f"Database connection failed: {e}")
            checks["PostgreSQL"] = f"Failed: {e}"
        
        # Check Redis connection
        try:
            import redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            r = redis.from_url(redis_url)
            r.ping()
            checks["Redis"] = "Connected"
            self._print_success("Redis connection successful")
        except Exception as e:
            self._print_warning(f"Redis connection failed: {e} (optional)")
            checks["Redis"] = f"Not available: {e}"
        
        # Check required packages
        required_packages = [
            "pandas", "numpy", "sklearn", "xgboost", "lightgbm",
            "fastapi", "sqlalchemy", "redis", "nba_api", "tqdm"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self._print_error(f"Missing packages: {', '.join(missing_packages)}")
            checks["Required packages"] = f"Missing: {', '.join(missing_packages)}"
        else:
            self._print_success("All required packages installed")
            checks["Required packages"] = "All installed"
        
        # Log environment details
        logger.info("Environment check results:")
        for key, value in checks.items():
            logger.info(f"  {key}: {value}")
    
    async def _setup_database(self):
        """Create database tables"""
        self._print_header("Step 2: Database Setup")
        
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not set")
        
        # Read SQL script
        sql_file = Path("scripts/create_nba_tables.sql")
        if not sql_file.exists():
            raise FileNotFoundError(f"SQL script not found: {sql_file}")
        
        with open(sql_file, 'r') as f:
            sql_script = f.read()
        
        # Execute SQL
        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Split by semicolons and execute each statement
            statements = [s.strip() for s in sql_script.split(';') if s.strip()]
            
            with tqdm(total=len(statements), desc="Creating tables") as pbar:
                for statement in statements:
                    try:
                        conn.execute(text(statement))
                        conn.commit()
                        pbar.update(1)
                    except Exception as e:
                        if "already exists" in str(e):
                            self._print_warning(f"Table already exists (skipping)")
                            pbar.update(1)
                        else:
                            raise
        
        self._print_success("Database tables created successfully")
    
    async def _load_nba_data(self):
        """Load NBA data with progress tracking"""
        self._print_header("Step 3: Load NBA Data")
        
        from api.data.pipeline import NBADataPipeline
        pipeline = NBADataPipeline()
        
        # Load teams
        self._print_success("Loading NBA teams...")
        teams_count = await pipeline.load_teams()
        self._print_success(f"Loaded {teams_count} teams")
        
        # Load players
        self._print_success("Loading active NBA players...")
        players_count = await pipeline.load_players(only_active=True)
        self._print_success(f"Loaded {players_count} players")
        
        # Load sample game data for demo
        self._print_success("Loading game history (sample for demo)...")
        await pipeline.load_historical_data(
            seasons=['2023-24', '2024-25'],
            sample_players=50  # Load data for 50 players for faster setup
        )
        self._print_success("Game history loaded")
    
    async def _train_models(self):
        """Train ML models"""
        self._print_header("Step 4: Train Models")
        
        from api.ml.train_models import NBAModelTrainer
        trainer = NBAModelTrainer()
        
        # Prepare training data
        self._print_success("Preparing training data...")
        features_df, targets = await trainer.prepare_training_data(
            seasons=['2023-24'],
            min_games_per_player=20
        )
        
        if len(features_df) < 100:
            self._print_warning(f"Limited training data: {len(features_df)} samples")
        
        # Train models
        self._print_success("Training models...")
        results = trainer.train_models(features_df, targets)
        
        # Save models
        trainer.save_models()
        
        # Print results
        print("\nModel Training Results:")
        for target, result in results.items():
            best_model = result['best_model_name']
            metrics = result['all_models'][best_model]
            print(f"\n{target.upper()}:")
            print(f"  Best Model: {best_model}")
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  R²: {metrics['r2']:.3f}")
        
        self._print_success("Models trained and saved")
    
    async def _validate_setup(self):
        """Validate the complete setup"""
        self._print_header("Step 5: Validation")
        
        validation_results = {
            "database": False,
            "models": False,
            "api": False,
            "predictions": False
        }
        
        # Validate database
        try:
            from database.connection import get_db_session
            from api.models.game_data import Player, GameLog
            from sqlalchemy import select, func
            
            async with get_db_session() as session:
                # Check player count
                result = await session.execute(select(func.count(Player.player_id)))
                player_count = result.scalar()
                
                # Check game log count
                result = await session.execute(select(func.count(GameLog.id)))
                game_count = result.scalar()
                
                if player_count > 0 and game_count > 0:
                    validation_results["database"] = True
                    self._print_success(f"Database validated: {player_count} players, {game_count} games")
                else:
                    self._print_error("Database validation failed: No data found")
        except Exception as e:
            self._print_error(f"Database validation error: {e}")
        
        # Validate models
        try:
            model_files = list(Path("models").glob("*.pkl"))
            if len(model_files) >= 3:  # At least 3 models (points, rebounds, assists)
                validation_results["models"] = True
                self._print_success(f"Models validated: {len(model_files)} model files found")
            else:
                self._print_error(f"Model validation failed: Only {len(model_files)} models found")
        except Exception as e:
            self._print_error(f"Model validation error: {e}")
        
        # Validate API
        try:
            # Test import
            from api.main import app
            validation_results["api"] = True
            self._print_success("API validated: Successfully imported")
        except Exception as e:
            self._print_error(f"API validation error: {e}")
        
        # Test prediction
        try:
            from ml.serving.predictor_v2 import PredictionService
            service = PredictionService()
            
            # Make a test prediction
            result = await service.predict(
                player_id="2544",  # LeBron James
                game_date="2025-01-20",
                opponent_team="LAL"
            )
            
            if result and "predictions" in result:
                validation_results["predictions"] = True
                self._print_success(f"Prediction validated: {result['predictions']}")
            else:
                self._print_error("Prediction validation failed: Invalid result")
        except Exception as e:
            self._print_error(f"Prediction validation error: {e}")
        
        # Summary
        all_valid = all(validation_results.values())
        if all_valid:
            self._print_success("All validations passed!")
        else:
            failed = [k for k, v in validation_results.items() if not v]
            self._print_warning(f"Validation incomplete. Failed: {', '.join(failed)}")
        
        return all_valid
    
    async def _generate_report(self):
        """Generate setup report"""
        self._print_header("Setup Report")
        
        report = {
            "setup_completed": datetime.now().isoformat(),
            "duration_minutes": (time.time() - self.start_time) / 60,
            "log_file": str(log_file),
            "database_status": "Connected",
            "models_trained": [],
            "data_loaded": {},
            "next_steps": []
        }
        
        # Check models
        model_files = list(Path("models").glob("*.pkl"))
        report["models_trained"] = [f.name for f in model_files]
        
        # Get data stats
        try:
            from database.connection import get_db_session
            from api.models.game_data import Player, GameLog, Team
            from sqlalchemy import select, func
            
            async with get_db_session() as session:
                # Count records
                players = await session.execute(select(func.count(Player.player_id)))
                games = await session.execute(select(func.count(GameLog.id)))
                teams = await session.execute(select(func.count(Team.team_id)))
                
                report["data_loaded"] = {
                    "players": players.scalar(),
                    "games": games.scalar(),
                    "teams": teams.scalar()
                }
        except Exception as e:
            logger.error(f"Error getting data stats: {e}")
        
        # Next steps
        report["next_steps"] = [
            "Start the API: uvicorn api.main:app --reload",
            "Access API docs: http://localhost:8000/docs",
            "Load more historical data: python scripts/load_initial_nba_data.py",
            "Run weekly model training: python api/jobs/weekly_retrain.py",
            "Deploy to Railway: git push"
        ]
        
        # Save report
        report_file = Path("setup_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\nSetup Summary:")
        print(f"  Duration: {report['duration_minutes']:.1f} minutes")
        print(f"  Models trained: {len(report['models_trained'])}")
        print(f"  Data loaded: {report['data_loaded']}")
        print(f"\nReport saved to: {report_file}")
        print("\nNext steps:")
        for i, step in enumerate(report["next_steps"], 1):
            print(f"  {i}. {step}")


async def main():
    """Main entry point"""
    setup = ProductionSetup()
    await setup.run()


if __name__ == "__main__":
    asyncio.run(main())