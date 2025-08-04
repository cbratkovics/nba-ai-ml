"""
SQLAlchemy models for NBA ML prediction system
Production-ready schema with partitioning and optimization
"""
from datetime import datetime
from decimal import Decimal
from typing import Optional
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text, 
    ForeignKey, UniqueConstraint, Index, CheckConstraint, 
    DECIMAL, JSON, TIMESTAMP, Date, SmallInteger, BigInteger
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class UserTier(PyEnum):
    """API user tiers"""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class ModelStatus(PyEnum):
    """Model deployment status"""
    TRAINING = "training"
    VALIDATING = "validating"
    STAGED = "staged"
    PRODUCTION = "production"
    RETIRED = "retired"


class ExperimentStatus(PyEnum):
    """A/B test experiment status"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


# ============ Core Data Tables ============

class Player(Base):
    """Player master data"""
    __tablename__ = 'players'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    nba_player_id = Column(String(20), unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    full_name = Column(String(200), nullable=False, index=True)
    position = Column(String(10))
    height_inches = Column(SmallInteger)
    weight_pounds = Column(SmallInteger)
    birth_date = Column(Date)
    country = Column(String(100))
    draft_year = Column(SmallInteger)
    draft_round = Column(SmallInteger)
    draft_number = Column(SmallInteger)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSONB, default={})
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
    
    # Relationships
    game_logs = relationship("PlayerGameLog", back_populates="player", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="player", cascade="all, delete-orphan")
    features = relationship("PlayerFeature", back_populates="player", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_player_full_name', 'full_name'),
        Index('idx_player_active', 'is_active'),
    )


class Team(Base):
    """Team master data"""
    __tablename__ = 'teams'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    nba_team_id = Column(String(20), unique=True, nullable=False, index=True)
    abbreviation = Column(String(5), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    full_name = Column(String(200), nullable=False)
    city = Column(String(100))
    state = Column(String(50))
    conference = Column(String(20))
    division = Column(String(50))
    venue = Column(String(200))
    year_founded = Column(SmallInteger)
    metadata = Column(JSONB, default={})
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
    
    # Relationships
    home_games = relationship("Game", foreign_keys="Game.home_team_id", back_populates="home_team")
    away_games = relationship("Game", foreign_keys="Game.away_team_id", back_populates="away_team")
    
    __table_args__ = (
        Index('idx_team_abbreviation', 'abbreviation'),
        Index('idx_team_conference_division', 'conference', 'division'),
    )


class Game(Base):
    """Game schedule and results"""
    __tablename__ = 'games'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    nba_game_id = Column(String(30), unique=True, nullable=False, index=True)
    game_date = Column(Date, nullable=False, index=True)
    game_time = Column(TIMESTAMP(timezone=True))
    season = Column(String(10), nullable=False, index=True)
    season_type = Column(String(20), default='Regular Season')
    
    home_team_id = Column(UUID(as_uuid=True), ForeignKey('teams.id'), nullable=False)
    away_team_id = Column(UUID(as_uuid=True), ForeignKey('teams.id'), nullable=False)
    
    home_team_score = Column(SmallInteger)
    away_team_score = Column(SmallInteger)
    
    game_status = Column(String(20))  # scheduled, in_progress, final
    attendance = Column(Integer)
    game_duration_minutes = Column(SmallInteger)
    
    metadata = Column(JSONB, default={})  # Store additional game context
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_games")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_games")
    player_game_logs = relationship("PlayerGameLog", back_populates="game", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_game_date', 'game_date'),
        Index('idx_game_season', 'season'),
        Index('idx_game_teams', 'home_team_id', 'away_team_id'),
        CheckConstraint('home_team_id != away_team_id', name='different_teams'),
    )


class PlayerGameLog(Base):
    """Player performance by game - partitioned by date"""
    __tablename__ = 'player_game_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), ForeignKey('players.id'), nullable=False)
    game_id = Column(UUID(as_uuid=True), ForeignKey('games.id'), nullable=False)
    team_id = Column(UUID(as_uuid=True), ForeignKey('teams.id'), nullable=False)
    
    game_date = Column(Date, nullable=False, index=True)  # Denormalized for partitioning
    
    # Playing time
    minutes_played = Column(DECIMAL(5, 2))
    started = Column(Boolean, default=False)
    
    # Basic stats
    points = Column(SmallInteger, default=0)
    rebounds = Column(SmallInteger, default=0)
    offensive_rebounds = Column(SmallInteger, default=0)
    defensive_rebounds = Column(SmallInteger, default=0)
    assists = Column(SmallInteger, default=0)
    steals = Column(SmallInteger, default=0)
    blocks = Column(SmallInteger, default=0)
    turnovers = Column(SmallInteger, default=0)
    personal_fouls = Column(SmallInteger, default=0)
    
    # Shooting stats
    field_goals_made = Column(SmallInteger, default=0)
    field_goals_attempted = Column(SmallInteger, default=0)
    field_goal_percentage = Column(DECIMAL(4, 3))
    three_pointers_made = Column(SmallInteger, default=0)
    three_pointers_attempted = Column(SmallInteger, default=0)
    three_point_percentage = Column(DECIMAL(4, 3))
    free_throws_made = Column(SmallInteger, default=0)
    free_throws_attempted = Column(SmallInteger, default=0)
    free_throw_percentage = Column(DECIMAL(4, 3))
    
    # Advanced stats
    plus_minus = Column(SmallInteger)
    true_shooting_percentage = Column(DECIMAL(4, 3))
    effective_field_goal_percentage = Column(DECIMAL(4, 3))
    usage_rate = Column(DECIMAL(4, 3))
    game_score = Column(DECIMAL(5, 2))
    
    # Context
    home_game = Column(Boolean)
    rest_days = Column(SmallInteger)
    back_to_back = Column(Boolean, default=False)
    
    metadata = Column(JSONB, default={})
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="game_logs")
    game = relationship("Game", back_populates="player_game_logs")
    
    __table_args__ = (
        UniqueConstraint('player_id', 'game_id', name='unique_player_game'),
        Index('idx_player_game_log_player_date', 'player_id', 'game_date'),
        Index('idx_player_game_log_date', 'game_date'),
        Index('idx_player_game_log_player', 'player_id'),
        {
            'postgresql_partition_by': 'RANGE (game_date)',
            'postgresql_tablespace': 'pg_default'
        }
    )


# ============ Feature Store ============

class PlayerFeature(Base):
    """Engineered features for ML - partitioned by date"""
    __tablename__ = 'player_features'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    player_id = Column(UUID(as_uuid=True), ForeignKey('players.id'), nullable=False)
    game_id = Column(UUID(as_uuid=True), ForeignKey('games.id'), nullable=False)
    game_date = Column(Date, nullable=False, index=True)
    
    # Feature version for tracking
    feature_version = Column(String(20), nullable=False, default='v1.0')
    
    # Rolling averages (different windows)
    points_ma5 = Column(DECIMAL(5, 2))
    points_ma10 = Column(DECIMAL(5, 2))
    points_ma20 = Column(DECIMAL(5, 2))
    rebounds_ma5 = Column(DECIMAL(5, 2))
    rebounds_ma10 = Column(DECIMAL(5, 2))
    assists_ma5 = Column(DECIMAL(5, 2))
    assists_ma10 = Column(DECIMAL(5, 2))
    
    # Trends
    points_trend = Column(DECIMAL(5, 2))
    rebounds_trend = Column(DECIMAL(5, 2))
    assists_trend = Column(DECIMAL(5, 2))
    hot_streak = Column(Boolean)
    
    # Matchup features
    opponent_defensive_rating = Column(DECIMAL(5, 2))
    matchup_difficulty = Column(DECIMAL(3, 2))
    pace_factor = Column(DECIMAL(5, 2))
    
    # All features as JSONB for flexibility
    features = Column(JSONB, nullable=False)
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    ttl = Column(TIMESTAMP(timezone=True))  # Feature expiration
    
    # Relationships
    player = relationship("Player", back_populates="features")
    
    __table_args__ = (
        UniqueConstraint('player_id', 'game_id', 'feature_version', name='unique_player_game_features'),
        Index('idx_player_features_date', 'game_date'),
        Index('idx_player_features_player_date', 'player_id', 'game_date'),
        Index('idx_player_features_version', 'feature_version'),
        {
            'postgresql_partition_by': 'RANGE (game_date)',
        }
    )


# ============ Model Registry ============

class Model(Base):
    """ML model registry"""
    __tablename__ = 'models'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False)
    target = Column(String(50), nullable=False)  # points, rebounds, assists, etc.
    
    model_type = Column(String(100))  # ensemble, xgboost, neural_network
    framework = Column(String(50))  # sklearn, pytorch, tensorflow
    
    # Storage
    artifact_uri = Column(Text, nullable=False)  # S3 or filesystem path
    model_size_mb = Column(DECIMAL(10, 2))
    
    # Performance metrics
    train_r2_score = Column(DECIMAL(5, 4))
    val_r2_score = Column(DECIMAL(5, 4))
    test_r2_score = Column(DECIMAL(5, 4))
    train_mae = Column(DECIMAL(10, 4))
    val_mae = Column(DECIMAL(10, 4))
    test_mae = Column(DECIMAL(10, 4))
    train_rmse = Column(DECIMAL(10, 4))
    val_rmse = Column(DECIMAL(10, 4))
    test_rmse = Column(DECIMAL(10, 4))
    
    # Training metadata
    training_duration_seconds = Column(Integer)
    training_samples = Column(Integer)
    feature_count = Column(Integer)
    hyperparameters = Column(JSONB)
    feature_importance = Column(JSONB)
    
    # Deployment
    status = Column(String(20), default='training')
    deployed_at = Column(TIMESTAMP(timezone=True))
    retired_at = Column(TIMESTAMP(timezone=True))
    
    metadata = Column(JSONB, default={})
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    created_by = Column(String(100))
    
    # Relationships
    predictions = relationship("Prediction", back_populates="model")
    experiments = relationship("ExperimentModel", back_populates="model")
    
    __table_args__ = (
        UniqueConstraint('name', 'version', name='unique_model_version'),
        Index('idx_model_status', 'status'),
        Index('idx_model_target', 'target'),
        CheckConstraint('val_r2_score >= 0 AND val_r2_score <= 1', name='valid_r2_score'),
    )


# ============ Predictions ============

class Prediction(Base):
    """Prediction history - partitioned by date"""
    __tablename__ = 'predictions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(String(100), unique=True, nullable=False, index=True)
    
    player_id = Column(UUID(as_uuid=True), ForeignKey('players.id'), nullable=False)
    game_id = Column(UUID(as_uuid=True), ForeignKey('games.id'))
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    game_date = Column(Date, nullable=False, index=True)
    
    # Predictions
    predicted_points = Column(DECIMAL(5, 2))
    predicted_rebounds = Column(DECIMAL(5, 2))
    predicted_assists = Column(DECIMAL(5, 2))
    predicted_steals = Column(DECIMAL(5, 2))
    predicted_blocks = Column(DECIMAL(5, 2))
    
    # Confidence intervals
    points_lower_bound = Column(DECIMAL(5, 2))
    points_upper_bound = Column(DECIMAL(5, 2))
    rebounds_lower_bound = Column(DECIMAL(5, 2))
    rebounds_upper_bound = Column(DECIMAL(5, 2))
    assists_lower_bound = Column(DECIMAL(5, 2))
    assists_upper_bound = Column(DECIMAL(5, 2))
    
    # Confidence and metadata
    confidence_score = Column(DECIMAL(3, 2))
    prediction_metadata = Column(JSONB, default={})
    
    # API tracking
    api_key_id = Column(UUID(as_uuid=True), ForeignKey('api_keys.id'))
    request_id = Column(String(100))
    response_time_ms = Column(Integer)
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="predictions")
    model = relationship("Model", back_populates="predictions")
    feedback = relationship("PredictionFeedback", back_populates="prediction", uselist=False)
    
    __table_args__ = (
        Index('idx_prediction_date', 'game_date'),
        Index('idx_prediction_player_date', 'player_id', 'game_date'),
        Index('idx_prediction_created', 'created_at'),
        {
            'postgresql_partition_by': 'RANGE (game_date)',
        }
    )


class PredictionFeedback(Base):
    """Actual results for predictions"""
    __tablename__ = 'prediction_feedback'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey('predictions.id'), unique=True, nullable=False)
    
    # Actual results
    actual_points = Column(SmallInteger)
    actual_rebounds = Column(SmallInteger)
    actual_assists = Column(SmallInteger)
    actual_steals = Column(SmallInteger)
    actual_blocks = Column(SmallInteger)
    
    # Calculated errors
    points_error = Column(DECIMAL(5, 2))
    rebounds_error = Column(DECIMAL(5, 2))
    assists_error = Column(DECIMAL(5, 2))
    
    points_absolute_error = Column(DECIMAL(5, 2))
    rebounds_absolute_error = Column(DECIMAL(5, 2))
    assists_absolute_error = Column(DECIMAL(5, 2))
    
    # Accuracy flags
    within_confidence_interval = Column(Boolean)
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Relationships
    prediction = relationship("Prediction", back_populates="feedback")
    
    __table_args__ = (
        Index('idx_feedback_created', 'created_at'),
    )


# ============ User Management ============

class APIKey(Base):
    """API key management"""
    __tablename__ = 'api_keys'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash = Column(String(256), unique=True, nullable=False, index=True)
    key_prefix = Column(String(20), nullable=False)  # For identification
    
    user_email = Column(String(255), nullable=False, index=True)
    organization = Column(String(200))
    
    tier = Column(String(20), nullable=False, default='free')
    
    # Rate limits
    requests_per_hour = Column(Integer, default=1000)
    requests_per_day = Column(Integer, default=10000)
    
    # Usage tracking
    total_requests = Column(BigInteger, default=0)
    total_predictions = Column(BigInteger, default=0)
    last_used_at = Column(TIMESTAMP(timezone=True))
    
    # Status
    is_active = Column(Boolean, default=True)
    expires_at = Column(TIMESTAMP(timezone=True))
    
    metadata = Column(JSONB, default={})
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_api_key_email', 'user_email'),
        Index('idx_api_key_tier', 'tier'),
        Index('idx_api_key_active', 'is_active'),
    )


# ============ A/B Testing ============

class Experiment(Base):
    """A/B testing experiments"""
    __tablename__ = 'experiments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), unique=True, nullable=False)
    description = Column(Text)
    
    status = Column(String(20), default='draft')
    
    # Traffic allocation
    traffic_percentage = Column(DECIMAL(5, 2), default=50.0)
    
    # Timing
    start_date = Column(TIMESTAMP(timezone=True))
    end_date = Column(TIMESTAMP(timezone=True))
    
    # Success metrics
    primary_metric = Column(String(100))  # r2_score, mae, rmse
    success_criteria = Column(JSONB)
    
    # Results
    control_metrics = Column(JSONB)
    treatment_metrics = Column(JSONB)
    statistical_significance = Column(DECIMAL(5, 4))
    winner = Column(String(20))  # control, treatment, none
    
    metadata = Column(JSONB, default={})
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    created_by = Column(String(100))
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
    
    # Relationships
    models = relationship("ExperimentModel", back_populates="experiment")
    
    __table_args__ = (
        Index('idx_experiment_status', 'status'),
        Index('idx_experiment_dates', 'start_date', 'end_date'),
        CheckConstraint('traffic_percentage >= 0 AND traffic_percentage <= 100', name='valid_traffic_percentage'),
    )


class ExperimentModel(Base):
    """Models in experiments"""
    __tablename__ = 'experiment_models'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey('experiments.id'), nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    
    variant = Column(String(20), nullable=False)  # control, treatment
    
    # Tracking
    request_count = Column(BigInteger, default=0)
    total_predictions = Column(BigInteger, default=0)
    
    # Performance
    cumulative_mae = Column(DECIMAL(10, 4))
    cumulative_rmse = Column(DECIMAL(10, 4))
    cumulative_r2 = Column(DECIMAL(5, 4))
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Relationships
    experiment = relationship("Experiment", back_populates="models")
    model = relationship("Model", back_populates="experiments")
    
    __table_args__ = (
        UniqueConstraint('experiment_id', 'variant', name='unique_experiment_variant'),
        Index('idx_experiment_model_variant', 'variant'),
    )


# ============ Audit & Monitoring ============

class AuditLog(Base):
    """Audit trail for compliance"""
    __tablename__ = 'audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    entity_type = Column(String(50), nullable=False)  # model, prediction, user, etc.
    entity_id = Column(String(100))
    action = Column(String(50), nullable=False)  # create, update, delete, access
    
    user_id = Column(String(100))
    api_key_id = Column(UUID(as_uuid=True))
    
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)
    
    __table_args__ = (
        Index('idx_audit_entity', 'entity_type', 'entity_id'),
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_created', 'created_at'),
    )


# ============ Data Source Tracking ============

class DataSource(Base):
    """Track data sources and their usage"""
    __tablename__ = 'data_sources'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    source_type = Column(String(50))  # api, scraper, manual
    
    base_url = Column(Text)
    requires_auth = Column(Boolean, default=False)
    
    # Rate limits
    rate_limit_requests = Column(Integer)
    rate_limit_window_seconds = Column(Integer)
    
    # Cost tracking
    cost_per_request = Column(DECIMAL(10, 6))
    monthly_cost = Column(DECIMAL(10, 2))
    
    # Usage
    total_requests = Column(BigInteger, default=0)
    failed_requests = Column(BigInteger, default=0)
    last_accessed_at = Column(TIMESTAMP(timezone=True))
    
    # Status
    is_active = Column(Boolean, default=True)
    health_status = Column(String(20))  # healthy, degraded, down
    
    metadata = Column(JSONB, default={})
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_data_source_active', 'is_active'),
        Index('idx_data_source_type', 'source_type'),
    )