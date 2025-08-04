from sqlalchemy import Column, String, Float, DateTime, Integer, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255))
    player_id = Column(String(50))
    player_name = Column(String(255))
    prediction_date = Column(DateTime)
    game_date = Column(DateTime)
    opponent_team = Column(String(10))
    points_predicted = Column(Float)
    rebounds_predicted = Column(Float)
    assists_predicted = Column(Float)
    points_actual = Column(Float, nullable=True)
    rebounds_actual = Column(Float, nullable=True)
    assists_actual = Column(Float, nullable=True)
    confidence = Column(Float)
    model_version = Column(String(50))
    created_at = Column(DateTime, server_default=func.now())

class APIKey(Base):
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    key_hash = Column(String(255), unique=True)
    user_id = Column(String(255))
    clerk_user_id = Column(String(255))
    tier = Column(String(50), default='free')
    monthly_limit = Column(Integer, default=1000)
    requests_this_month = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    last_used_at = Column(DateTime)

class ModelPerformance(Base):
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_version = Column(String(50))
    target = Column(String(20))  # 'points', 'rebounds', 'assists'
    accuracy = Column(Float)
    mae = Column(Float)
    rmse = Column(Float)
    predictions_count = Column(Integer)
    evaluated_at = Column(DateTime, server_default=func.now())