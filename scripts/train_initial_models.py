#!/usr/bin/env python3
"""
Train initial NBA ML models for all targets
Production-ready training script with comprehensive logging
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.training.pipeline import ProductionTrainingPipeline
from ml.data.processors.data_validator import NBADataValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/training_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_model_report(results: dict, output_path: str = "reports/model_performance.json"):
    """
    Generate comprehensive model performance report
    
    Args:
        results: Dictionary with training results for all models
        output_path: Path to save report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "models": {},
        "summary": {
            "total_models": len(results),
            "successful": 0,
            "failed": 0,
            "meets_targets": 0
        }
    }
    
    for target, result in results.items():
        if result and "metrics" in result:
            model_info = {
                "target": target,
                "metrics": result["metrics"],
                "model_id": result.get("model_info", {}).get("model_id"),
                "model_path": result.get("model_info", {}).get("model_path"),
                "feature_count": len(result.get("feature_columns", [])),
                "validation_status": result.get("validation", {}).get("status"),
                "meets_performance_targets": result["metrics"].get("test_r2", 0) >= 0.94
            }
            
            report["models"][target] = model_info
            report["summary"]["successful"] += 1
            
            if model_info["meets_performance_targets"]:
                report["summary"]["meets_targets"] += 1
        else:
            report["models"][target] = {"status": "failed"}
            report["summary"]["failed"] += 1
    
    # Add best model
    if report["models"]:
        best_model = max(
            [(k, v) for k, v in report["models"].items() if "metrics" in v],
            key=lambda x: x[1]["metrics"].get("test_r2", 0),
            default=(None, None)
        )
        if best_model[0]:
            report["summary"]["best_model"] = {
                "target": best_model[0],
                "r2_score": best_model[1]["metrics"]["test_r2"]
            }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also save human-readable report
    text_report_path = output_path.replace('.json', '.txt')
    with open(text_report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("NBA ML MODEL TRAINING REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {report['generated_at']}\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Models: {report['summary']['total_models']}\n")
        f.write(f"Successful: {report['summary']['successful']}\n")
        f.write(f"Failed: {report['summary']['failed']}\n")
        f.write(f"Meet Targets: {report['summary']['meets_targets']}\n")
        
        if "best_model" in report["summary"]:
            f.write(f"\nBest Model: {report['summary']['best_model']['target']} ")
            f.write(f"(R² = {report['summary']['best_model']['r2_score']:.4f})\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("INDIVIDUAL MODEL RESULTS\n")
        f.write("=" * 60 + "\n")
        
        for target, model_info in report["models"].items():
            f.write(f"\n{target} PREDICTION MODEL\n")
            f.write("-" * 30 + "\n")
            
            if "metrics" in model_info:
                f.write(f"R² Score: {model_info['metrics'].get('test_r2', 0):.4f}\n")
                f.write(f"MAE: {model_info['metrics'].get('test_mae', 0):.2f}\n")
                f.write(f"RMSE: {model_info['metrics'].get('test_rmse', 0):.2f}\n")
                f.write(f"Features: {model_info.get('feature_count', 0)}\n")
                f.write(f"Meets Targets: {'Yes' if model_info['meets_performance_targets'] else 'No'}\n")
            else:
                f.write("Status: Failed\n")
    
    logger.info(f"Model report saved to {output_path} and {text_report_path}")
    
    return report


async def train_single_model(pipeline: ProductionTrainingPipeline,
                           target: str,
                           seasons: list,
                           experiment_prefix: str = "initial") -> dict:
    """
    Train a single model with error handling
    
    Args:
        pipeline: Training pipeline instance
        target: Target variable to predict
        seasons: List of seasons to use
        experiment_prefix: Prefix for experiment name
        
    Returns:
        Training results dictionary
    """
    try:
        logger.info(f"Training model for {target}...")
        logger.info(f"Using seasons: {seasons}")
        
        result = await pipeline.train(
            target=target,
            seasons=seasons,
            experiment_name=f"{experiment_prefix}_{target.lower()}_model"
        )
        
        logger.info(f"✓ {target} model training completed successfully")
        logger.info(f"  R² Score: {result['metrics']['test_r2']:.4f}")
        logger.info(f"  MAE: {result['metrics']['test_mae']:.2f}")
        logger.info(f"  RMSE: {result['metrics']['test_rmse']:.2f}")
        
        return result
        
    except Exception as e:
        logger.error(f"✗ {target} model training failed: {str(e)}")
        return None


async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train NBA ML models')
    parser.add_argument('--targets', nargs='+', default=["PTS", "REB", "AST"],
                       help='Target variables to train models for')
    parser.add_argument('--seasons', nargs='+', default=["2021-22", "2022-23", "2023-24"],
                       help='Seasons to use for training')
    parser.add_argument('--config', default='config/training.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--experiment-prefix', default='initial',
                       help='Prefix for MLflow experiment names')
    parser.add_argument('--parallel', action='store_true',
                       help='Train models in parallel')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run data validation without training')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("NBA ML MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Targets: {args.targets}")
    logger.info(f"Seasons: {args.seasons}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Parallel: {args.parallel}")
    
    # Initialize pipeline
    logger.info("\nInitializing training pipeline...")
    pipeline = ProductionTrainingPipeline(config_path=args.config)
    
    # Validate data first if requested
    if args.validate_only:
        logger.info("\nRunning data validation only...")
        validator = NBADataValidator()
        
        # Load sample data for validation
        sample_data = await pipeline.load_training_data(args.seasons[:1])
        validation_result = validator.validate_player_game_log(sample_data)
        
        logger.info(f"Validation Status: {validation_result.status.value}")
        logger.info(f"Passed Checks: {validation_result.passed_checks}")
        logger.info(f"Failed Checks: {validation_result.failed_checks}")
        logger.info(f"Warnings: {validation_result.warning_checks}")
        
        if validation_result.errors:
            logger.error("Validation Errors:")
            for error in validation_result.errors:
                logger.error(f"  - {error}")
        
        if validation_result.warnings:
            logger.warning("Validation Warnings:")
            for warning in validation_result.warnings:
                logger.warning(f"  - {warning}")
        
        return
    
    # Train models
    results = {}
    
    if args.parallel:
        # Train models in parallel
        logger.info("\nTraining models in parallel...")
        tasks = [
            train_single_model(pipeline, target, args.seasons, args.experiment_prefix)
            for target in args.targets
        ]
        
        model_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for target, result in zip(args.targets, model_results):
            if isinstance(result, Exception):
                logger.error(f"Error training {target}: {result}")
                results[target] = None
            else:
                results[target] = result
    else:
        # Train models sequentially
        logger.info("\nTraining models sequentially...")
        for target in args.targets:
            results[target] = await train_single_model(
                pipeline, target, args.seasons, args.experiment_prefix
            )
    
    # Generate performance report
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING PERFORMANCE REPORT")
    logger.info("=" * 60)
    
    report = generate_model_report(results)
    
    # Print summary
    logger.info("\nTRAINING SUMMARY")
    logger.info("-" * 30)
    logger.info(f"Total Models Attempted: {len(args.targets)}")
    logger.info(f"Successful: {report['summary']['successful']}")
    logger.info(f"Failed: {report['summary']['failed']}")
    logger.info(f"Meet Performance Targets: {report['summary']['meets_targets']}")
    
    if "best_model" in report["summary"]:
        logger.info(f"\nBest Model: {report['summary']['best_model']['target']} "
                   f"(R² = {report['summary']['best_model']['r2_score']:.4f})")
    
    # Check if any models met targets
    if report['summary']['meets_targets'] > 0:
        logger.info("\n✓ Training completed successfully!")
        logger.info(f"  {report['summary']['meets_targets']} model(s) meet performance targets")
    else:
        logger.warning("\n⚠ No models met performance targets")
        logger.warning("  Consider adjusting hyperparameters or collecting more data")
    
    logger.info(f"\nReports saved to reports/model_performance.*")
    logger.info(f"Models saved to models/")
    logger.info(f"Logs saved to logs/")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())