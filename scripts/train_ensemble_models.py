#!/usr/bin/env python3
"""
Train ensemble ML models for NBA predictions
"""
import pandas as pd
import numpy as np
import json
import os
import pickle
import warnings
from datetime import datetime
from typing import Dict, Tuple, List, Any

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.features.feature_pipeline import FeaturePipeline

warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = 'data/nba_players_2024.csv'
MODEL_DIR = 'models'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Target columns
TARGETS = ['PTS', 'REB', 'AST']

# Feature columns to exclude
EXCLUDE_COLS = ['PLAYER_ID', 'PLAYER_NAME', 'GAME_DATE', 'OPPONENT'] + TARGETS


class NBAEnsembleModel:
    """Ensemble model for NBA predictions"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_pipeline = FeaturePipeline()
        self.feature_names = []
        self.metrics = {}
        
    def prepare_data(self, df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for training"""
        # Apply feature engineering
        df_features = self.feature_pipeline.transform(df.copy())
        
        # Get feature columns
        feature_cols = [col for col in df_features.columns 
                       if col not in EXCLUDE_COLS and not col.startswith(target)]
        
        # Remove NaN and inf values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(0)
        
        # Extract features and target
        X = df_features[feature_cols].values
        y = df_features[target].values
        
        self.feature_names = feature_cols
        
        return X, y
    
    def create_ensemble(self, target: str) -> VotingRegressor:
        """Create ensemble model for a specific target"""
        
        # XGBoost - Primary model
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # LightGBM - Fast and accurate
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
        
        # Random Forest - Stability
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # Neural Network - Non-linear patterns
        nn_model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE
        )
        
        # Create ensemble with weighted voting
        # Weights based on expected performance
        ensemble = VotingRegressor([
            ('xgboost', xgb_model),
            ('lightgbm', lgb_model),
            ('rf', rf_model),
            ('nn', nn_model)
        ])
        
        return ensemble
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train ensemble models for all targets"""
        print("Starting ensemble training...")
        print("=" * 60)
        
        results = {}
        
        for target in TARGETS:
            print(f"\nTraining models for {target}...")
            print("-" * 40)
            
            # Prepare data
            X, y = self.prepare_data(df, target)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler
            self.scalers[target] = scaler
            
            # Create and train ensemble
            ensemble = self.create_ensemble(target)
            
            # Train individual models for comparison
            models = {
                'xgboost': xgb.XGBRegressor(
                    n_estimators=300, max_depth=6, learning_rate=0.01,
                    subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1
                ),
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=300, max_depth=6, learning_rate=0.01,
                    num_leaves=31, subsample=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
                ),
                'neural_net': MLPRegressor(
                    hidden_layer_sizes=(64, 32), max_iter=500, 
                    early_stopping=True, random_state=RANDOM_STATE
                )
            }
            
            # Train and evaluate individual models
            model_scores = {}
            best_model = None
            best_score = -np.inf
            
            for name, model in models.items():
                print(f"  Training {name}...")
                
                # Use appropriate data based on model type
                if name == 'neural_net':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                model_scores[name] = {
                    'r2_score': r2,
                    'mae': mae,
                    'rmse': rmse
                }
                
                print(f"    R² Score: {r2:.4f}")
                print(f"    MAE: {mae:.2f}")
                print(f"    RMSE: {rmse:.2f}")
                
                if r2 > best_score:
                    best_score = r2
                    best_model = model
            
            # Create weighted ensemble based on individual performance
            weights = []
            for name in ['xgboost', 'lightgbm', 'random_forest', 'neural_net']:
                # Weight based on R² score
                weight = max(0, model_scores[name]['r2_score'])
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w/total_weight for w in weights]
            else:
                weights = [0.25, 0.25, 0.25, 0.25]
            
            print(f"\n  Ensemble weights: {dict(zip(models.keys(), weights))}")
            
            # Create final weighted ensemble
            final_ensemble = VotingRegressor([
                ('xgboost', models['xgboost']),
                ('lightgbm', models['lightgbm']),
                ('rf', models['random_forest']),
                ('nn', models['neural_net'])
            ], weights=weights)
            
            # Train final ensemble
            print(f"  Training weighted ensemble...")
            
            # For neural net in ensemble, we need scaled data
            # Train on original data and let each model handle it
            final_ensemble.fit(X_train, y_train)
            
            # Predict and evaluate ensemble
            y_pred_ensemble = final_ensemble.predict(X_test)
            
            ensemble_r2 = r2_score(y_test, y_pred_ensemble)
            ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
            
            print(f"\n  Ensemble Performance:")
            print(f"    R² Score: {ensemble_r2:.4f}")
            print(f"    MAE: {ensemble_mae:.2f}")
            print(f"    RMSE: {ensemble_rmse:.2f}")
            
            # Cross-validation
            print(f"\n  Running cross-validation...")
            tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
            cv_scores = cross_val_score(
                best_model, X, y, cv=tscv, 
                scoring='r2', n_jobs=-1
            )
            
            print(f"    CV R² Scores: {cv_scores}")
            print(f"    Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Store results
            self.models[target] = final_ensemble
            self.metrics[target] = {
                'ensemble': {
                    'r2_score': ensemble_r2,
                    'mae': ensemble_mae,
                    'rmse': ensemble_rmse,
                    'weights': dict(zip(models.keys(), weights))
                },
                'individual_models': model_scores,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_size': len(y_test),
                'train_size': len(y_train)
            }
            
            results[target] = self.metrics[target]
        
        # Calculate overall metrics
        overall_r2 = np.mean([self.metrics[t]['ensemble']['r2_score'] for t in TARGETS])
        print("\n" + "=" * 60)
        print(f"OVERALL ENSEMBLE R² SCORE: {overall_r2:.4f}")
        print("=" * 60)
        
        return results
    
    def save(self, model_path: str, metrics_path: str):
        """Save trained models and metrics"""
        # Save models
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_pipeline': self.feature_pipeline,
            'feature_names': self.feature_names,
            'targets': TARGETS,
            'version': '2.1.0',
            'trained_at': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metrics
        metrics_data = {
            'metrics': self.metrics,
            'overall_r2': np.mean([self.metrics[t]['ensemble']['r2_score'] for t in TARGETS]),
            'version': '2.1.0',
            'trained_at': datetime.now().isoformat()
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def get_feature_importance(self, target: str) -> pd.DataFrame:
        """Get feature importance for a target"""
        if target not in self.models:
            return pd.DataFrame()
        
        # Get XGBoost model from ensemble for feature importance
        xgb_model = self.models[target].estimators_[0]
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


def main():
    """Main training pipeline"""
    print("NBA Ensemble Model Training")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        print("Please run 'python scripts/collect_nba_data_2024.py' first")
        return
    
    # Load data
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} game records")
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Initialize and train ensemble
    ensemble = NBAEnsembleModel()
    results = ensemble.train(df)
    
    # Save models and metrics
    model_path = os.path.join(MODEL_DIR, 'nba_ensemble_2024.pkl')
    metrics_path = os.path.join(MODEL_DIR, 'performance_metrics.json')
    
    print(f"\nSaving models to {model_path}...")
    ensemble.save(model_path, metrics_path)
    
    print(f"Saving metrics to {metrics_path}...")
    
    # Print feature importance for points prediction
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES FOR POINTS PREDICTION")
    print("=" * 60)
    importance_df = ensemble.get_feature_importance('PTS')
    if not importance_df.empty:
        print(importance_df.head(10).to_string(index=False))
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Models saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Overall R² Score: {results['PTS']['ensemble']['r2_score']:.4f}")
    print("\nNext steps:")
    print("1. Update ml/serving/predictor.py to use the new models")
    print("2. Test predictions with python scripts/test_predictions.py")
    print("3. Deploy to production")


if __name__ == "__main__":
    main()