#!/usr/bin/env python3
"""
Create a dummy model for quick deployment testing
This creates a simple RandomForestRegressor with dummy data
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_model():
    """Create and save a dummy model for deployment testing"""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    logger.info("Creating dummy training data...")
    
    # Create dummy features (simulating NBA player stats)
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Feature names that match what the API expects
    feature_names = [
        'PTS_MA5', 'PTS_MA10', 'PTS_MA20',
        'REB_MA5', 'REB_MA10', 'REB_MA20', 
        'AST_MA5', 'AST_MA10', 'AST_MA20',
        'FG_PCT', 'FT_PCT', 'FG3_PCT',
        'MIN', 'GAMES_PLAYED', 'AGE',
        'HOME_GAME', 'REST_DAYS', 'BACK_TO_BACK',
        'MATCHUP_DIFFICULTY', 'SEASON_GAME_NUM'
    ]
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Add some realistic constraints
    X[:, 9] = np.clip(X[:, 9], 0.3, 0.6)  # FG_PCT between 30-60%
    X[:, 10] = np.clip(X[:, 10], 0.6, 0.95)  # FT_PCT between 60-95%
    X[:, 11] = np.clip(X[:, 11], 0.2, 0.45)  # FG3_PCT between 20-45%
    X[:, 12] = np.clip(np.abs(X[:, 12]) * 10 + 25, 15, 40)  # MIN between 15-40
    X[:, 13] = np.clip(np.abs(X[:, 13]) * 20 + 40, 10, 82)  # GAMES_PLAYED 10-82
    X[:, 14] = np.clip(np.abs(X[:, 14]) * 5 + 25, 19, 40)  # AGE between 19-40
    X[:, 15] = np.random.randint(0, 2, n_samples)  # HOME_GAME binary
    X[:, 16] = np.random.randint(0, 4, n_samples)  # REST_DAYS 0-3
    X[:, 17] = np.random.randint(0, 2, n_samples)  # BACK_TO_BACK binary
    X[:, 18] = np.random.uniform(0.5, 1.5, n_samples)  # MATCHUP_DIFFICULTY
    X[:, 19] = np.random.randint(1, 83, n_samples)  # SEASON_GAME_NUM 1-82
    
    # Create target variable (points) with some correlation to features
    y = (
        X[:, 0] * 5 +  # Strong correlation with recent average
        X[:, 12] * 0.3 +  # Minutes played affects points
        X[:, 15] * 2 +  # Home game advantage
        X[:, 16] * 1 +  # Rest days help
        np.random.randn(n_samples) * 3  # Some noise
    ) + 20  # Base points around 20
    
    # Ensure points are realistic (0-60 range)
    y = np.clip(y, 0, 60)
    
    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Train models for different targets
    targets = ['points', 'rebounds', 'assists']
    
    for target in targets:
        logger.info(f"Training {target} model...")
        
        # Adjust target values for different stats
        if target == 'rebounds':
            y_target = np.clip(y * 0.3 + np.random.randn(n_samples) * 2, 0, 20)
        elif target == 'assists':
            y_target = np.clip(y * 0.2 + np.random.randn(n_samples) * 1.5, 0, 15)
        else:
            y_target = y
        
        # Create and train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X, y_target)
        
        # Calculate simple metrics
        train_score = model.score(X, y_target)
        logger.info(f"{target} model R² score: {train_score:.3f}")
        
        # Save model
        model_path = models_dir / f"rf_{target}_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved {target} model to {model_path}")
        
        # Also save a scaler for the model
        scaler = StandardScaler()
        scaler.fit(X)
        scaler_path = models_dir / f"scaler_{target}.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved {target} scaler to {scaler_path}")
    
    # Save feature names
    features_path = models_dir / "features.json"
    with open(features_path, 'w') as f:
        json.dump({
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "targets": targets,
            "model_type": "RandomForestRegressor",
            "created": "2024-01-01"
        }, f, indent=2)
    logger.info(f"Saved feature configuration to {features_path}")
    
    # Create a simple ensemble model as well
    logger.info("Creating ensemble model...")
    ensemble_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    ensemble_model.fit(X, y)
    
    ensemble_path = models_dir / "rf_model.pkl"
    joblib.dump(ensemble_model, ensemble_path)
    logger.info(f"Saved ensemble model to {ensemble_path}")
    
    logger.info("✅ Successfully created dummy models for deployment!")
    
    return True

if __name__ == "__main__":
    create_dummy_model()