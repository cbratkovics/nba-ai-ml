"""
Game data models for NBA player and team statistics
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Date, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Optional

Base = declarative_base()


class Player(Base):
    """NBA Player model"""
    __tablename__ = "players"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    team = Column(String)
    position = Column(String)
    height = Column(String)
    weight = Column(Integer)
    birth_date = Column(Date)
    experience = Column(Integer)
    jersey_number = Column(String)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    game_logs = relationship("GameLog", back_populates="player")
    features = relationship("PlayerFeatures", back_populates="player", uselist=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Team(Base):
    """NBA Team model"""
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(String, unique=True, index=True, nullable=False)
    abbreviation = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    city = Column(String)
    conference = Column(String)
    division = Column(String)
    
    # Team statistics
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    win_percentage = Column(Float, default=0.0)
    pace = Column(Float)
    offensive_rating = Column(Float)
    defensive_rating = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GameLog(Base):
    """Player game statistics"""
    __tablename__ = "game_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(String, ForeignKey("players.player_id"), index=True)
    game_id = Column(String, index=True)
    game_date = Column(Date, index=True)
    team = Column(String)
    opponent = Column(String)
    is_home = Column(Boolean)
    
    # Game statistics
    minutes_played = Column(Float)
    points = Column(Integer)
    rebounds = Column(Integer)
    assists = Column(Integer)
    steals = Column(Integer)
    blocks = Column(Integer)
    turnovers = Column(Integer)
    personal_fouls = Column(Integer)
    
    # Shooting statistics
    field_goals_made = Column(Integer)
    field_goals_attempted = Column(Integer)
    field_goal_percentage = Column(Float)
    three_pointers_made = Column(Integer)
    three_pointers_attempted = Column(Integer)
    three_point_percentage = Column(Float)
    free_throws_made = Column(Integer)
    free_throws_attempted = Column(Integer)
    free_throw_percentage = Column(Float)
    
    # Advanced statistics
    plus_minus = Column(Integer)
    game_score = Column(Float)
    true_shooting_percentage = Column(Float)
    effective_field_goal_percentage = Column(Float)
    
    # Relationships
    player = relationship("Player", back_populates="game_logs")
    
    created_at = Column(DateTime, default=datetime.utcnow)


class Schedule(Base):
    """NBA game schedule"""
    __tablename__ = "schedule"
    
    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(String, unique=True, index=True)
    game_date = Column(Date, index=True)
    home_team = Column(String)
    away_team = Column(String)
    season = Column(String)
    season_type = Column(String)  # Regular, Playoffs, etc.
    
    # Game results (populated after game)
    home_score = Column(Integer)
    away_score = Column(Integer)
    overtime_periods = Column(Integer, default=0)
    game_status = Column(String, default="scheduled")  # scheduled, in_progress, final
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PlayerFeatures(Base):
    """Pre-computed player features for ML models"""
    __tablename__ = "player_features"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(String, ForeignKey("players.player_id"), unique=True, index=True)
    
    # Rolling averages
    points_last_5 = Column(Float)
    points_last_10 = Column(Float)
    points_last_20 = Column(Float)
    points_season = Column(Float)
    
    rebounds_last_5 = Column(Float)
    rebounds_last_10 = Column(Float)
    rebounds_last_20 = Column(Float)
    rebounds_season = Column(Float)
    
    assists_last_5 = Column(Float)
    assists_last_10 = Column(Float)
    assists_last_20 = Column(Float)
    assists_season = Column(Float)
    
    # Performance metrics
    field_goal_pct_last_10 = Column(Float)
    three_point_pct_last_10 = Column(Float)
    free_throw_pct_last_10 = Column(Float)
    
    # Advanced metrics
    player_efficiency_rating = Column(Float)
    true_shooting_percentage = Column(Float)
    usage_rate = Column(Float)
    
    # Trend indicators
    points_trend = Column(Float)  # Positive = improving, negative = declining
    rebounds_trend = Column(Float)
    assists_trend = Column(Float)
    
    # Consistency metrics
    points_std_dev = Column(Float)
    rebounds_std_dev = Column(Float)
    assists_std_dev = Column(Float)
    
    # Rest and fatigue
    avg_rest_days = Column(Float)
    back_to_back_performance = Column(JSON)  # JSON with b2b stats
    
    # Matchup history
    opponent_performance = Column(JSON)  # JSON with performance vs each team
    
    # Relationships
    player = relationship("Player", back_populates="features")
    
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    
class PlayerGameData(Base):
    """Historical game data for a player (simplified for API responses)"""
    __tablename__ = "player_game_data"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(String, index=True)
    player_name = Column(String)
    game_date = Column(DateTime)
    opponent = Column(String)
    points = Column(Float)
    rebounds = Column(Float)
    assists = Column(Float)
    minutes_played = Column(Float)
    field_goal_percentage = Column(Float)
    three_point_percentage = Column(Float)
    free_throw_percentage = Column(Float)
    

class TeamGameData(Base):
    """Team performance data"""
    __tablename__ = "team_game_data"
    
    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(String, index=True)
    team_name = Column(String)
    game_date = Column(DateTime)
    opponent = Column(String)
    points_scored = Column(Integer)
    points_allowed = Column(Integer)
    pace = Column(Float)
    offensive_rating = Column(Float)
    defensive_rating = Column(Float)