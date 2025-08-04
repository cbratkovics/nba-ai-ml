#!/usr/bin/env python3
"""
Comprehensive Test Suite for NBA ML System
Validates all components: data pipeline, models, API, and infrastructure
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import requests
import psutil
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import DatabaseManager, init_db
from database.models import Player, Team, Game, PlayerGameLog, MLModel, ModelVersion
from ml.data.processors.data_validator import NBADataValidator
from api.models import PredictionRequest, BatchPredictionRequest
import joblib
from sqlalchemy import func, and_
import subprocess
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/test_suite_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestResult:
    """Represents a test result"""
    
    def __init__(self, test_name: str, category: str):
        self.test_name = test_name
        self.category = category
        self.status = "PENDING"
        self.message = ""
        self.duration = 0
        self.details = {}
    
    def passed(self, message: str = "", details: Dict = None):
        self.status = "PASSED"
        self.message = message
        self.details = details or {}
    
    def failed(self, message: str = "", details: Dict = None):
        self.status = "FAILED"
        self.message = message
        self.details = details or {}
    
    def skipped(self, message: str = ""):
        self.status = "SKIPPED"
        self.message = message


class ComprehensiveTestSuite:
    """Comprehensive test suite for NBA ML system"""
    
    def __init__(self):
        """Initialize test suite"""
        self.db_manager = None
        self.validator = NBADataValidator()
        self.test_results = []
        self.api_process = None
        self.api_url = "http://localhost:8000"
        
        # Test categories
        self.categories = {
            "database": "Database Tests",
            "data": "Data Quality Tests",
            "models": "Model Tests",
            "api": "API Tests",
            "performance": "Performance Tests",
            "integration": "Integration Tests"
        }
        
        logger.info("Comprehensive Test Suite initialized")
    
    async def test_database_connection(self) -> TestResult:
        """Test database connection and tables"""
        result = TestResult("Database Connection", "database")
        start_time = time.time()
        
        try:
            if not self.db_manager:
                self.db_manager = init_db()
            
            with self.db_manager.get_db(read_only=True) as session:
                # Check tables exist
                player_count = session.query(func.count(Player.id)).scalar()
                team_count = session.query(func.count(Team.id)).scalar()
                game_count = session.query(func.count(Game.id)).scalar()
                log_count = session.query(func.count(PlayerGameLog.id)).scalar()
                
                details = {
                    "players": player_count,
                    "teams": team_count,
                    "games": game_count,
                    "game_logs": log_count
                }
                
                if player_count > 0 and team_count > 0:
                    result.passed(f"Database connected. {log_count} game logs found", details)
                else:
                    result.failed("Database empty or missing data", details)
                    
        except Exception as e:
            result.failed(f"Database connection failed: {e}")
        
        result.duration = time.time() - start_time
        return result
    
    async def test_data_quality(self) -> TestResult:
        """Test data quality and validation"""
        result = TestResult("Data Quality Validation", "data")
        start_time = time.time()
        
        try:
            with self.db_manager.get_db(read_only=True) as session:
                # Get sample data
                sample_logs = session.query(PlayerGameLog).limit(1000).all()
                
                if not sample_logs:
                    result.skipped("No data to validate")
                    return result
                
                # Convert to DataFrame
                data = []
                for log in sample_logs:
                    data.append({
                        'player_id': log.player_id,
                        'game_date': log.game_date,
                        'points': log.points,
                        'rebounds': log.rebounds,
                        'assists': log.assists,
                        'minutes_played': log.minutes_played,
                        'field_goals_made': log.field_goals_made,
                        'field_goals_attempted': log.field_goals_attempted
                    })
                
                df = pd.DataFrame(data)
                
                # Validate
                validation_result = self.validator.validate_player_game_log(df)
                
                details = {
                    "status": validation_result.status.value,
                    "passed_checks": validation_result.passed_checks,
                    "failed_checks": validation_result.failed_checks,
                    "warning_checks": validation_result.warning_checks,
                    "sample_size": len(df)
                }
                
                if validation_result.status.value == "PASSED":
                    result.passed("Data quality validation passed", details)
                elif validation_result.status.value == "WARNING":
                    result.passed(f"Data validation passed with {len(validation_result.warnings)} warnings", details)
                else:
                    result.failed(f"Data validation failed: {validation_result.errors[0] if validation_result.errors else 'Unknown'}", details)
                    
        except Exception as e:
            result.failed(f"Data quality test failed: {e}")
        
        result.duration = time.time() - start_time
        return result
    
    async def test_model_loading(self) -> List[TestResult]:
        """Test model loading and predictions"""
        results = []
        
        models_dir = Path("api/ml_models")
        
        for target in ['pts', 'reb', 'ast']:
            result = TestResult(f"{target.upper()} Model Loading", "models")
            start_time = time.time()
            
            try:
                model_path = models_dir / target / "model.pkl"
                scaler_path = models_dir / target / "scaler.pkl"
                metadata_path = models_dir / target / "metadata.json"
                
                if not model_path.exists():
                    result.skipped(f"No {target.upper()} model found")
                    results.append(result)
                    continue
                
                # Load model
                model = joblib.load(model_path)
                
                # Load scaler if exists
                scaler = None
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)
                
                # Load metadata
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                # Test prediction
                n_features = metadata.get("n_features", 100)
                test_input = np.random.randn(1, n_features)
                
                if scaler:
                    test_input = scaler.transform(test_input)
                
                prediction = model.predict(test_input)
                
                details = {
                    "model_type": metadata.get("model_type", "unknown"),
                    "n_features": n_features,
                    "prediction_shape": prediction.shape,
                    "sample_prediction": float(prediction[0])
                }
                
                result.passed(f"{target.upper()} model loaded and predicted successfully", details)
                
            except Exception as e:
                result.failed(f"{target.upper()} model test failed: {e}")
            
            result.duration = time.time() - start_time
            results.append(result)
        
        return results
    
    async def start_api_server(self) -> bool:
        """Start the API server for testing"""
        try:
            logger.info("Starting API server...")
            
            # Start API server in background
            self.api_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if sys.platform != "win32" else None
            )
            
            # Wait for server to start
            max_attempts = 30
            for i in range(max_attempts):
                try:
                    response = requests.get(f"{self.api_url}/health")
                    if response.status_code == 200:
                        logger.info("API server started successfully")
                        return True
                except:
                    pass
                
                await asyncio.sleep(1)
            
            logger.error("API server failed to start")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    async def stop_api_server(self):
        """Stop the API server"""
        if self.api_process:
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(self.api_process.pid), signal.SIGTERM)
                else:
                    self.api_process.terminate()
                self.api_process.wait(timeout=5)
                logger.info("API server stopped")
            except:
                self.api_process.kill()
    
    async def test_api_endpoints(self) -> List[TestResult]:
        """Test API endpoints"""
        results = []
        
        # Test health endpoint
        result = TestResult("API Health Check", "api")
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.api_url}/health")
            if response.status_code == 200:
                data = response.json()
                result.passed("API health check passed", data)
            else:
                result.failed(f"Health check returned {response.status_code}")
        except Exception as e:
            result.failed(f"Health check failed: {e}")
        
        result.duration = time.time() - start_time
        results.append(result)
        
        # Test prediction endpoints
        for target in ['points', 'rebounds', 'assists']:
            result = TestResult(f"{target.title()} Prediction Endpoint", "api")
            start_time = time.time()
            
            try:
                # Create test request
                request_data = {
                    "player_id": "203999",  # Nikola Jokic
                    "opponent_team": "LAL",
                    "is_home": True,
                    "days_rest": 2,
                    "season_avg_points": 25.5,
                    "season_avg_rebounds": 11.2,
                    "season_avg_assists": 8.1,
                    "last_5_games_avg_points": 28.0,
                    "last_5_games_avg_rebounds": 12.0,
                    "last_5_games_avg_assists": 9.0
                }
                
                response = requests.post(
                    f"{self.api_url}/api/v1/predict/{target}",
                    json=request_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    details = {
                        "prediction": data.get("prediction"),
                        "confidence": data.get("confidence"),
                        "model_version": data.get("model_version")
                    }
                    result.passed(f"{target.title()} prediction successful", details)
                else:
                    result.failed(f"Prediction returned {response.status_code}: {response.text}")
                    
            except Exception as e:
                result.failed(f"Prediction test failed: {e}")
            
            result.duration = time.time() - start_time
            results.append(result)
        
        return results
    
    async def test_performance(self) -> TestResult:
        """Test system performance"""
        result = TestResult("Performance Benchmarks", "performance")
        start_time = time.time()
        
        try:
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Database query performance
            db_start = time.time()
            with self.db_manager.get_db(read_only=True) as session:
                count = session.query(func.count(PlayerGameLog.id)).scalar()
            db_time = time.time() - db_start
            
            # Model prediction performance (if API is running)
            api_times = []
            if self.api_process:
                for _ in range(10):
                    api_start = time.time()
                    try:
                        response = requests.get(f"{self.api_url}/health")
                        api_times.append(time.time() - api_start)
                    except:
                        pass
            
            details = {
                "memory_mb": round(memory_mb, 2),
                "cpu_percent": round(cpu_percent, 2),
                "db_query_time": round(db_time, 3),
                "avg_api_response_time": round(np.mean(api_times), 3) if api_times else None,
                "total_game_logs": count
            }
            
            # Check thresholds
            if memory_mb < 2000 and cpu_percent < 80 and db_time < 1.0:
                result.passed("Performance within acceptable limits", details)
            else:
                result.failed("Performance issues detected", details)
                
        except Exception as e:
            result.failed(f"Performance test failed: {e}")
        
        result.duration = time.time() - start_time
        return result
    
    async def test_integration(self) -> TestResult:
        """Test end-to-end integration"""
        result = TestResult("End-to-End Integration", "integration")
        start_time = time.time()
        
        try:
            # Test data flow: DB -> Model -> API -> Prediction
            with self.db_manager.get_db(read_only=True) as session:
                # Get a real player
                player = session.query(Player).first()
                
                if not player:
                    result.skipped("No players in database")
                    return result
                
                # Get their recent stats
                recent_logs = session.query(PlayerGameLog).filter_by(
                    player_id=player.id
                ).order_by(PlayerGameLog.game_date.desc()).limit(10).all()
                
                if len(recent_logs) < 5:
                    result.skipped("Insufficient player data")
                    return result
                
                # Calculate averages
                pts_avg = np.mean([log.points for log in recent_logs[:5]])
                reb_avg = np.mean([log.rebounds for log in recent_logs[:5]])
                ast_avg = np.mean([log.assists for log in recent_logs[:5]])
            
            # If API is running, test prediction
            if self.api_process:
                request_data = {
                    "player_id": player.nba_player_id,
                    "opponent_team": "LAL",
                    "is_home": True,
                    "days_rest": 2,
                    "season_avg_points": float(pts_avg),
                    "season_avg_rebounds": float(reb_avg),
                    "season_avg_assists": float(ast_avg),
                    "last_5_games_avg_points": float(pts_avg),
                    "last_5_games_avg_rebounds": float(reb_avg),
                    "last_5_games_avg_assists": float(ast_avg)
                }
                
                response = requests.post(
                    f"{self.api_url}/api/v1/predict/points",
                    json=request_data
                )
                
                if response.status_code == 200:
                    prediction = response.json()
                    details = {
                        "player": player.full_name,
                        "prediction": prediction.get("prediction"),
                        "confidence": prediction.get("confidence"),
                        "recent_avg": float(pts_avg)
                    }
                    result.passed("End-to-end integration test passed", details)
                else:
                    result.failed(f"API prediction failed: {response.status_code}")
            else:
                # Test without API
                details = {
                    "player": player.full_name,
                    "recent_pts_avg": float(pts_avg),
                    "recent_reb_avg": float(reb_avg),
                    "recent_ast_avg": float(ast_avg)
                }
                result.passed("Data integration test passed (API not tested)", details)
                
        except Exception as e:
            result.failed(f"Integration test failed: {e}")
        
        result.duration = time.time() - start_time
        return result
    
    async def run_all_tests(self, include_api: bool = True) -> Dict[str, Any]:
        """
        Run all tests
        
        Args:
            include_api: Whether to include API tests
            
        Returns:
            Dictionary with test results
        """
        logger.info("="*60)
        logger.info("COMPREHENSIVE TEST SUITE")
        logger.info("="*60)
        
        all_results = []
        
        # Initialize database
        if not self.db_manager:
            self.db_manager = init_db()
        
        # Database tests
        logger.info("\n[1/6] Running database tests...")
        all_results.append(await self.test_database_connection())
        
        # Data quality tests
        logger.info("\n[2/6] Running data quality tests...")
        all_results.append(await self.test_data_quality())
        
        # Model tests
        logger.info("\n[3/6] Running model tests...")
        model_results = await self.test_model_loading()
        all_results.extend(model_results)
        
        # API tests
        if include_api:
            logger.info("\n[4/6] Starting API server...")
            api_started = await self.start_api_server()
            
            if api_started:
                logger.info("\n[5/6] Running API tests...")
                api_results = await self.test_api_endpoints()
                all_results.extend(api_results)
            else:
                logger.warning("API tests skipped - server failed to start")
        
        # Performance tests
        logger.info("\n[6/6] Running performance tests...")
        all_results.append(await self.test_performance())
        
        # Integration tests
        logger.info("\n[7/7] Running integration tests...")
        all_results.append(await self.test_integration())
        
        # Stop API server
        if include_api and self.api_process:
            await self.stop_api_server()
        
        # Generate report
        report = self.generate_test_report(all_results)
        
        return report
    
    def generate_test_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        # Calculate statistics
        total_tests = len(results)
        passed = sum(1 for r in results if r.status == "PASSED")
        failed = sum(1 for r in results if r.status == "FAILED")
        skipped = sum(1 for r in results if r.status == "SKIPPED")
        
        # Group by category
        by_category = {}
        for category_key, category_name in self.categories.items():
            category_results = [r for r in results if r.category == category_key]
            if category_results:
                by_category[category_name] = {
                    "total": len(category_results),
                    "passed": sum(1 for r in category_results if r.status == "PASSED"),
                    "failed": sum(1 for r in category_results if r.status == "FAILED"),
                    "skipped": sum(1 for r in category_results if r.status == "SKIPPED")
                }
        
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "pass_rate": round(passed / total_tests * 100, 1) if total_tests > 0 else 0
            },
            "by_category": by_category,
            "test_results": []
        }
        
        # Add individual test results
        for result in results:
            report["test_results"].append({
                "name": result.test_name,
                "category": result.category,
                "status": result.status,
                "message": result.message,
                "duration": round(result.duration, 3),
                "details": result.details
            })
        
        # Save reports
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_path = report_dir / f"test_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Text report
        text_path = report_dir / f"test_report_{timestamp}.txt"
        with open(text_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("NBA ML SYSTEM TEST REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-"*30 + "\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Passed: {passed} ({report['summary']['pass_rate']}%)\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Skipped: {skipped}\n\n")
            
            f.write("BY CATEGORY\n")
            f.write("-"*30 + "\n")
            for category, stats in by_category.items():
                f.write(f"{category}: {stats['passed']}/{stats['total']} passed\n")
            f.write("\n")
            
            f.write("TEST RESULTS\n")
            f.write("-"*30 + "\n")
            for result in results:
                status_symbol = "✓" if result.status == "PASSED" else "✗" if result.status == "FAILED" else "⊘"
                f.write(f"{status_symbol} {result.test_name}: {result.status}\n")
                if result.message:
                    f.write(f"  {result.message}\n")
                if result.duration > 0:
                    f.write(f"  Duration: {result.duration:.3f}s\n")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUITE COMPLETE")
        logger.info("="*60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed} ({report['summary']['pass_rate']}%)")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped: {skipped}")
        
        if failed == 0:
            logger.info("\n✓ All tests passed successfully!")
        else:
            logger.warning(f"\n⚠ {failed} test(s) failed")
            for result in results:
                if result.status == "FAILED":
                    logger.warning(f"  - {result.test_name}: {result.message}")
        
        logger.info(f"\nReports saved to:")
        logger.info(f"  - {json_path}")
        logger.info(f"  - {text_path}")
        
        return report


async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Run comprehensive test suite')
    parser.add_argument('--skip-api', action='store_true',
                       help='Skip API tests')
    parser.add_argument('--category', choices=['database', 'data', 'models', 'api', 'performance', 'integration'],
                       help='Run only specific category of tests')
    
    args = parser.parse_args()
    
    # Create necessary directories
    for dir_name in ['logs', 'reports']:
        os.makedirs(dir_name, exist_ok=True)
    
    # Initialize test suite
    test_suite = ComprehensiveTestSuite()
    
    # Run tests
    report = await test_suite.run_all_tests(include_api=not args.skip_api)
    
    # Exit with appropriate code
    if report['summary']['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())