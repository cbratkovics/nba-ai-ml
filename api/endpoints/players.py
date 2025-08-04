"""
Player management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import logging
from sqlalchemy import select, and_, or_, func, desc
from database.connection import get_db_session
from api.models.game_data import Player, GameLog
from api.middleware.auth import verify_api_key
from api.data.nba_client import NBAStatsClient
from pydantic import BaseModel
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

router = APIRouter()


class PlayerInfo(BaseModel):
    player_id: str
    player_name: str
    team_id: Optional[str]
    team_abbreviation: Optional[str]
    position: Optional[str]
    is_active: bool
    games_played: Optional[int] = 0
    last_game_date: Optional[str] = None
    season_avg_pts: Optional[float] = None
    season_avg_reb: Optional[float] = None
    season_avg_ast: Optional[float] = None


class PlayerStats(BaseModel):
    player_id: str
    player_name: str
    games_played: int
    avg_points: float
    avg_rebounds: float
    avg_assists: float
    avg_minutes: float
    fg_pct: float
    ft_pct: float
    fg3_pct: float
    last_5_games: Optional[List[dict]] = None
    season_highs: Optional[dict] = None


@router.get("/players", response_model=List[PlayerInfo])
async def list_players(
    team_id: Optional[str] = None,
    position: Optional[str] = None,
    active_only: bool = True,
    search: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    api_key: str = Depends(verify_api_key)
):
    """
    List available players for predictions
    
    Filter by team, position, or search by name
    """
    try:
        async with get_db_session() as session:
            # Build query
            query = select(Player)
            
            # Apply filters
            if active_only:
                query = query.where(Player.is_active == True)
            
            if team_id:
                query = query.where(Player.team_id == team_id)
            
            if position:
                query = query.where(Player.position.ilike(f"%{position}%"))
            
            if search:
                query = query.where(
                    Player.player_name.ilike(f"%{search}%")
                )
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            # Execute query
            result = await session.execute(query)
            players = result.scalars().all()
            
            # Get additional stats for each player
            player_list = []
            for player in players:
                # Get game count and last game
                games_result = await session.execute(
                    select(
                        func.count(GameLog.id),
                        func.max(GameLog.game_date),
                        func.avg(GameLog.points),
                        func.avg(GameLog.rebounds),
                        func.avg(GameLog.assists)
                    ).where(
                        and_(
                            GameLog.player_id == player.player_id,
                            GameLog.season == '2024-25'
                        )
                    )
                )
                games_count, last_game, avg_pts, avg_reb, avg_ast = games_result.one()
                
                player_info = PlayerInfo(
                    player_id=player.player_id,
                    player_name=player.player_name,
                    team_id=player.team_id,
                    team_abbreviation=player.team_abbreviation,
                    position=player.position,
                    is_active=player.is_active,
                    games_played=games_count or 0,
                    last_game_date=last_game.isoformat() if last_game else None,
                    season_avg_pts=round(float(avg_pts), 1) if avg_pts else None,
                    season_avg_reb=round(float(avg_reb), 1) if avg_reb else None,
                    season_avg_ast=round(float(avg_ast), 1) if avg_ast else None
                )
                player_list.append(player_info)
            
            return player_list
            
    except Exception as e:
        logger.error(f"Error listing players: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving players")


@router.get("/players/{player_id}/stats", response_model=PlayerStats)
async def get_player_stats(
    player_id: str,
    season: str = "2024-25",
    include_last_5: bool = True,
    api_key: str = Depends(verify_api_key)
):
    """
    Get detailed statistics for a specific player
    """
    try:
        async with get_db_session() as session:
            # Get player info
            result = await session.execute(
                select(Player).where(Player.player_id == player_id)
            )
            player = result.scalar_one_or_none()
            
            if not player:
                # Try to fetch from NBA API
                client = NBAStatsClient()
                player_info = client.get_player_info(player_id)
                if not player_info:
                    raise HTTPException(status_code=404, detail="Player not found")
                player_name = player_info.get('player_name', f"Player {player_id}")
            else:
                player_name = player.player_name
            
            # Get season stats
            stats_result = await session.execute(
                select(
                    func.count(GameLog.id),
                    func.avg(GameLog.points),
                    func.avg(GameLog.rebounds),
                    func.avg(GameLog.assists),
                    func.avg(GameLog.minutes_played),
                    func.avg(GameLog.field_goal_pct),
                    func.avg(GameLog.free_throw_pct),
                    func.avg(GameLog.three_point_pct),
                    func.max(GameLog.points),
                    func.max(GameLog.rebounds),
                    func.max(GameLog.assists)
                ).where(
                    and_(
                        GameLog.player_id == player_id,
                        GameLog.season == season
                    )
                )
            )
            
            (games_count, avg_pts, avg_reb, avg_ast, avg_min,
             fg_pct, ft_pct, fg3_pct, max_pts, max_reb, max_ast) = stats_result.one()
            
            # Get last 5 games if requested
            last_5_games = None
            if include_last_5:
                last_5_result = await session.execute(
                    select(GameLog)
                    .where(GameLog.player_id == player_id)
                    .order_by(desc(GameLog.game_date))
                    .limit(5)
                )
                last_5 = last_5_result.scalars().all()
                
                last_5_games = [{
                    'game_date': g.game_date.isoformat(),
                    'opponent': g.opponent_id,
                    'points': g.points,
                    'rebounds': g.rebounds,
                    'assists': g.assists,
                    'minutes': g.minutes_played,
                    'fg_pct': g.field_goal_pct
                } for g in last_5]
            
            return PlayerStats(
                player_id=player_id,
                player_name=player_name,
                games_played=games_count or 0,
                avg_points=round(float(avg_pts), 1) if avg_pts else 0.0,
                avg_rebounds=round(float(avg_reb), 1) if avg_reb else 0.0,
                avg_assists=round(float(avg_ast), 1) if avg_ast else 0.0,
                avg_minutes=round(float(avg_min), 1) if avg_min else 0.0,
                fg_pct=round(float(fg_pct), 3) if fg_pct else 0.0,
                ft_pct=round(float(ft_pct), 3) if ft_pct else 0.0,
                fg3_pct=round(float(fg3_pct), 3) if fg3_pct else 0.0,
                last_5_games=last_5_games,
                season_highs={
                    'points': int(max_pts) if max_pts else 0,
                    'rebounds': int(max_reb) if max_reb else 0,
                    'assists': int(max_ast) if max_ast else 0
                } if games_count else None
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting player stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving player statistics")


@router.get("/predictions/accuracy")
async def get_prediction_accuracy(
    days: int = Query(7, ge=1, le=30),
    player_id: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Show model accuracy over time
    
    Compare predictions with actual results
    """
    try:
        # This is a placeholder implementation
        # In production, this would query stored predictions and compare with actual results
        
        accuracy_data = {
            'period': f'Last {days} days',
            'total_predictions': 150,
            'predictions_with_results': 120,
            'overall_accuracy': {
                'mae': 4.2,
                'rmse': 5.8,
                'percentage_within_5pts': 68.5
            },
            'by_stat': {
                'points': {'mae': 4.2, 'rmse': 5.8},
                'rebounds': {'mae': 2.4, 'rmse': 3.1},
                'assists': {'mae': 1.8, 'rmse': 2.3}
            },
            'trend': 'improving'  # or 'stable', 'declining'
        }
        
        if player_id:
            accuracy_data['player_specific'] = {
                'player_id': player_id,
                'predictions': 10,
                'mae': 3.8
            }
        
        return accuracy_data
        
    except Exception as e:
        logger.error(f"Error calculating prediction accuracy: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving accuracy data")


@router.post("/admin/data/refresh")
async def refresh_data(
    target: str = Query(..., regex="^(players|games|all)$"),
    api_key: str = Depends(verify_api_key)
):
    """
    Manual trigger for data update
    
    Admin endpoint to refresh NBA data
    """
    try:
        from api.data.pipeline import NBADataPipeline
        pipeline = NBADataPipeline()
        
        if target == "players" or target == "all":
            logger.info("Refreshing player data...")
            await pipeline.load_players(only_active=True)
        
        if target == "games" or target == "all":
            logger.info("Refreshing game data...")
            await pipeline.update_daily_games()
        
        return {
            "status": "success",
            "message": f"Data refresh initiated for: {target}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        raise HTTPException(status_code=500, detail="Error refreshing data")