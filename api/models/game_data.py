"""
Database models for NBA game data
"""
from sqlalchemy import Column, String, Integer, Float, Boolean, Date, DateTime, UniqueConstraint, Index
from sqlalchemy.sql import func
from database.connection import Base


class Player(Base):
    """NBA Player model"""
    __tablename__ = 'players'
    
    player_id = Column(String(50), primary_key=True)
    player_name = Column(String(255), nullable=False)
    team_id = Column(String(50))
    team_abbreviation = Column(String(10))
    position = Column(String(20))
    jersey_number = Column(String(10))
    height = Column(String(20))
    weight = Column(String(20))
    birth_date = Column(String(50))
    country = Column(String(100))
    school = Column(String(255))
    draft_year = Column(Integer)
    draft_round = Column(Integer)
    draft_number = Column(Integer)
    is_active = Column(Boolean, default=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def to_dict(self):
        return {
            'player_id': self.player_id,
            'player_name': self.player_name,
            'team_id': self.team_id,
            'team_abbreviation': self.team_abbreviation,
            'position': self.position,
            'is_active': self.is_active,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class GameLog(Base):
    """Player game statistics log"""
    __tablename__ = 'game_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(50), nullable=False)
    game_id = Column(String(50), nullable=False)
    game_date = Column(Date, nullable=False)
    season = Column(String(20))
    team_id = Column(String(50))
    opponent_id = Column(String(50))
    is_home = Column(Boolean)
    minutes_played = Column(Float)
    points = Column(Integer)
    rebounds = Column(Integer)
    assists = Column(Integer)
    steals = Column(Integer)
    blocks = Column(Integer)
    turnovers = Column(Integer)
    field_goals_made = Column(Integer)
    field_goals_attempted = Column(Integer)
    field_goal_pct = Column(Float)
    three_pointers_made = Column(Integer)
    three_pointers_attempted = Column(Integer)
    three_point_pct = Column(Float)
    free_throws_made = Column(Integer)
    free_throws_attempted = Column(Integer)
    free_throw_pct = Column(Float)
    offensive_rebounds = Column(Integer)
    defensive_rebounds = Column(Integer)
    personal_fouls = Column(Integer)
    plus_minus = Column(Integer)
    game_score = Column(Float)  # John Hollinger's game score metric
    
    __table_args__ = (
        UniqueConstraint('player_id', 'game_id', name='uq_player_game'),
        Index('idx_game_logs_player_date', 'player_id', 'game_date'),
        Index('idx_game_logs_game_date', 'game_date'),
        Index('idx_game_logs_season', 'season'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'player_id': self.player_id,
            'game_id': self.game_id,
            'game_date': self.game_date.isoformat() if self.game_date else None,
            'season': self.season,
            'points': self.points,
            'rebounds': self.rebounds,
            'assists': self.assists,
            'minutes_played': self.minutes_played,
            'field_goal_pct': self.field_goal_pct,
            'is_home': self.is_home
        }


class Team(Base):
    """NBA Team model"""
    __tablename__ = 'teams'
    
    team_id = Column(String(50), primary_key=True)
    team_name = Column(String(255), nullable=False)
    team_abbreviation = Column(String(10), nullable=False)
    team_city = Column(String(100))
    team_state = Column(String(50))
    conference = Column(String(20))
    division = Column(String(50))
    year_founded = Column(Integer)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def to_dict(self):
        return {
            'team_id': self.team_id,
            'team_name': self.team_name,
            'team_abbreviation': self.team_abbreviation,
            'conference': self.conference,
            'division': self.division
        }


class PlayerFeatures(Base):
    """Calculated features for player predictions"""
    __tablename__ = 'player_features'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(50), nullable=False)
    calculation_date = Column(Date, nullable=False)
    
    # Rolling averages
    pts_last_5 = Column(Float)
    pts_last_10 = Column(Float)
    pts_last_20 = Column(Float)
    pts_season = Column(Float)
    
    reb_last_5 = Column(Float)
    reb_last_10 = Column(Float)
    reb_last_20 = Column(Float)
    reb_season = Column(Float)
    
    ast_last_5 = Column(Float)
    ast_last_10 = Column(Float)
    ast_last_20 = Column(Float)
    ast_season = Column(Float)
    
    # Shooting percentages
    fg_pct_last_10 = Column(Float)
    ft_pct_last_10 = Column(Float)
    fg3_pct_last_10 = Column(Float)
    
    # Other features
    minutes_last_10 = Column(Float)
    games_played_season = Column(Integer)
    days_since_last_game = Column(Integer)
    back_to_backs_last_10 = Column(Integer)
    home_ppg = Column(Float)
    away_ppg = Column(Float)
    
    # Opponent-specific features (stored as JSON in a real implementation)
    vs_opponent_avg_pts = Column(Float)
    vs_opponent_avg_reb = Column(Float)
    vs_opponent_avg_ast = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('player_id', 'calculation_date', name='uq_player_date_features'),
        Index('idx_player_features_date', 'calculation_date'),
    )


class Schedule(Base):
    """NBA game schedule"""
    __tablename__ = 'schedule'
    
    game_id = Column(String(50), primary_key=True)
    game_date = Column(Date, nullable=False)
    home_team_id = Column(String(50), nullable=False)
    away_team_id = Column(String(50), nullable=False)
    home_team_score = Column(Integer)
    away_team_score = Column(Integer)
    is_completed = Column(Boolean, default=False)
    season = Column(String(20))
    
    __table_args__ = (
        Index('idx_schedule_date', 'game_date'),
        Index('idx_schedule_teams', 'home_team_id', 'away_team_id'),
    )