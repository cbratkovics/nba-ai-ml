"""
Real-time NBA game monitoring with WebSocket updates
"""
import asyncio
import json
import logging
import os
from datetime import datetime, date
from typing import Dict, Set, List, Any, Optional
import aiohttp
from fastapi import WebSocket, WebSocketDisconnect
from ml.serving.predictor_v2 import PredictionService
from api.data.nba_client import NBAStatsClient
from database.connection import get_db_session
from api.models.game_data import GameLog, Schedule
from sqlalchemy import select, and_
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class GameMonitor:
    """Monitor live NBA games and update predictions"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.monitoring_task: Optional[asyncio.Task] = None
        self.prediction_service = PredictionService()
        self.nba_client = NBAStatsClient()
        self.redis_client = None
        self.current_games: Dict[str, Dict] = {}
        self.game_predictions: Dict[str, Dict] = {}
        
        # Initialize Redis for caching
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        # Send initial data
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to NBA Game Monitor",
            "timestamp": datetime.now().isoformat()
        })
        
        # Send current games if any
        if self.current_games:
            await websocket.send_json({
                "type": "current_games",
                "games": list(self.current_games.values()),
                "timestamp": datetime.now().isoformat()
            })
    
    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected.add(connection)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        self.active_connections -= disconnected
    
    async def start_monitoring(self):
        """Start monitoring live games"""
        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Game monitoring already running")
            return
        
        self.monitoring_task = asyncio.create_task(self._monitor_games())
        logger.info("Game monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring games"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Game monitoring stopped")
    
    async def _monitor_games(self):
        """Main monitoring loop"""
        while True:
            try:
                # Get today's games
                todays_games = await self._get_todays_games()
                
                # Update current games
                for game in todays_games:
                    game_id = game['game_id']
                    
                    # Check if game is live
                    if self._is_game_live(game):
                        # Update game status
                        await self._update_game_status(game)
                        
                        # Update predictions
                        await self._update_game_predictions(game)
                
                # Broadcast updates
                await self._broadcast_game_updates()
                
                # Sleep for 30 seconds before next update
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in game monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _get_todays_games(self) -> List[Dict[str, Any]]:
        """Get today's NBA games"""
        # This would use a real NBA API endpoint
        # For demo, we'll simulate some games
        today = date.today()
        
        games = [
            {
                "game_id": f"GAME_{today}_LAL_BOS",
                "home_team": "LAL",
                "away_team": "BOS",
                "status": "live",
                "period": 3,
                "time_remaining": "5:32",
                "home_score": 78,
                "away_score": 82,
                "start_time": "19:30"
            },
            {
                "game_id": f"GAME_{today}_GSW_MIA",
                "home_team": "GSW",
                "away_team": "MIA",
                "status": "scheduled",
                "period": 0,
                "time_remaining": "",
                "home_score": 0,
                "away_score": 0,
                "start_time": "22:00"
            }
        ]
        
        return games
    
    def _is_game_live(self, game: Dict[str, Any]) -> bool:
        """Check if game is currently live"""
        return game.get("status") == "live"
    
    async def _update_game_status(self, game: Dict[str, Any]):
        """Update game status and scores"""
        game_id = game['game_id']
        
        # Check if this is a new update
        if game_id not in self.current_games or self.current_games[game_id] != game:
            self.current_games[game_id] = game
            
            # Broadcast score update
            await self.broadcast({
                "type": "score_update",
                "game_id": game_id,
                "game": game,
                "timestamp": datetime.now().isoformat()
            })
    
    async def _update_game_predictions(self, game: Dict[str, Any]):
        """Update predictions for players in live game"""
        game_id = game['game_id']
        
        # Get key players for both teams
        key_players = await self._get_key_players(game['home_team'], game['away_team'])
        
        predictions = {}
        for player_id, player_info in key_players.items():
            try:
                # Make updated prediction based on game progress
                prediction = await self._make_live_prediction(
                    player_id, 
                    player_info,
                    game
                )
                predictions[player_id] = prediction
                
            except Exception as e:
                logger.error(f"Error predicting for player {player_id}: {e}")
        
        # Store predictions
        self.game_predictions[game_id] = predictions
        
        # Broadcast predictions
        await self.broadcast({
            "type": "prediction_update",
            "game_id": game_id,
            "predictions": predictions,
            "game_progress": {
                "period": game['period'],
                "time_remaining": game['time_remaining']
            },
            "timestamp": datetime.now().isoformat()
        })
    
    async def _get_key_players(self, home_team: str, away_team: str) -> Dict[str, Dict]:
        """Get key players for teams"""
        # For demo, return some known players
        # In production, this would query the database
        key_players = {
            "2544": {"name": "LeBron James", "team": "LAL"},
            "203999": {"name": "Nikola Jokic", "team": "DEN"},
            "201939": {"name": "Stephen Curry", "team": "GSW"},
            "1628369": {"name": "Jayson Tatum", "team": "BOS"}
        }
        
        # Filter by teams
        return {
            pid: info for pid, info in key_players.items()
            if info['team'] in [home_team, away_team]
        }
    
    async def _make_live_prediction(self, 
                                   player_id: str, 
                                   player_info: Dict,
                                   game: Dict) -> Dict[str, Any]:
        """Make prediction adjusted for game progress"""
        # Get base prediction
        base_prediction = await self.prediction_service.predict(
            player_id=player_id,
            game_date=date.today(),
            opponent_team=game['away_team'] if player_info['team'] == game['home_team'] else game['home_team']
        )
        
        # Adjust based on game progress
        period = game['period']
        periods_played = min(period, 4)  # Regular periods
        periods_remaining = max(4 - periods_played, 0)
        
        if periods_remaining > 0:
            # Scale predictions based on remaining time
            scale_factor = periods_remaining / 4
            
            adjusted_predictions = {}
            for stat, value in base_prediction['predictions'].items():
                # Current pace
                if periods_played > 0:
                    current_pace = self._get_current_pace(player_id, stat, game)
                    # Weighted average of pace and prediction
                    adjusted = (current_pace * periods_played + value * scale_factor) / 4
                else:
                    adjusted = value
                
                adjusted_predictions[stat] = round(adjusted, 1)
            
            base_prediction['predictions'] = adjusted_predictions
            base_prediction['adjusted_for_live'] = True
            base_prediction['game_progress'] = f"Q{period}"
        
        return base_prediction
    
    def _get_current_pace(self, player_id: str, stat: str, game: Dict) -> float:
        """Get player's current pace in the game"""
        # This would get real-time stats from the game
        # For demo, simulate based on typical performance
        pace_multipliers = {
            "points": 1.1,
            "rebounds": 0.95,
            "assists": 1.05
        }
        
        # Simulate some variance
        import random
        base_pace = 25.0 if stat == "points" else 8.0
        return base_pace * pace_multipliers.get(stat, 1.0) * random.uniform(0.8, 1.2)
    
    async def _broadcast_game_updates(self):
        """Broadcast all game updates"""
        if not self.current_games:
            return
        
        # Calculate accuracy for completed quarters
        accuracy_updates = []
        for game_id, game in self.current_games.items():
            if game['period'] > 1:  # At least one quarter completed
                accuracy = await self._calculate_quarter_accuracy(game_id, game)
                if accuracy:
                    accuracy_updates.append(accuracy)
        
        if accuracy_updates:
            await self.broadcast({
                "type": "accuracy_update",
                "accuracies": accuracy_updates,
                "timestamp": datetime.now().isoformat()
            })
    
    async def _calculate_quarter_accuracy(self, game_id: str, game: Dict) -> Optional[Dict]:
        """Calculate prediction accuracy for completed quarters"""
        # This would compare predictions with actual performance
        # For demo, return simulated accuracy
        return {
            "game_id": game_id,
            "period": game['period'] - 1,
            "accuracy": {
                "points": {"mae": 2.3, "within_3": 0.75},
                "rebounds": {"mae": 1.8, "within_2": 0.68},
                "assists": {"mae": 1.2, "within_2": 0.82}
            }
        }
    
    async def handle_client_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle messages from client"""
        msg_type = message.get("type")
        
        if msg_type == "subscribe_game":
            game_id = message.get("game_id")
            # Send current predictions for this game
            if game_id in self.game_predictions:
                await websocket.send_json({
                    "type": "game_predictions",
                    "game_id": game_id,
                    "predictions": self.game_predictions[game_id],
                    "timestamp": datetime.now().isoformat()
                })
        
        elif msg_type == "get_live_games":
            # Send all current live games
            live_games = [g for g in self.current_games.values() if self._is_game_live(g)]
            await websocket.send_json({
                "type": "live_games",
                "games": live_games,
                "timestamp": datetime.now().isoformat()
            })


class LivePredictionTracker:
    """Track and store live predictions for analysis"""
    
    def __init__(self):
        self.predictions: Dict[str, List[Dict]] = {}  # game_id -> predictions
        self.actual_results: Dict[str, Dict] = {}  # game_id -> results
    
    async def store_prediction(self, 
                             game_id: str, 
                             player_id: str, 
                             prediction: Dict[str, Any],
                             game_state: Dict[str, Any]):
        """Store a prediction with game state"""
        if game_id not in self.predictions:
            self.predictions[game_id] = []
        
        self.predictions[game_id].append({
            "player_id": player_id,
            "prediction": prediction,
            "game_state": game_state,
            "timestamp": datetime.now().isoformat()
        })
        
        # Store in database for historical analysis
        await self._store_to_database(game_id, player_id, prediction, game_state)
    
    async def update_actual_results(self, game_id: str, results: Dict[str, Any]):
        """Update with actual game results"""
        self.actual_results[game_id] = results
        
        # Calculate accuracy
        if game_id in self.predictions:
            accuracy = await self._calculate_accuracy(game_id)
            logger.info(f"Game {game_id} accuracy: {accuracy}")
    
    async def _store_to_database(self, 
                               game_id: str,
                               player_id: str,
                               prediction: Dict[str, Any],
                               game_state: Dict[str, Any]):
        """Store prediction in database"""
        try:
            async with get_db_session() as session:
                # This would store in a live_predictions table
                # For now, just log
                logger.info(f"Storing live prediction for {player_id} in game {game_id}")
        except Exception as e:
            logger.error(f"Error storing live prediction: {e}")
    
    async def _calculate_accuracy(self, game_id: str) -> Dict[str, float]:
        """Calculate prediction accuracy for a game"""
        predictions = self.predictions.get(game_id, [])
        results = self.actual_results.get(game_id, {})
        
        if not predictions or not results:
            return {}
        
        # Calculate MAE for each stat
        accuracies = {}
        for stat in ['points', 'rebounds', 'assists']:
            errors = []
            for pred in predictions:
                player_id = pred['player_id']
                if player_id in results:
                    predicted = pred['prediction']['predictions'].get(stat, 0)
                    actual = results[player_id].get(stat, 0)
                    errors.append(abs(predicted - actual))
            
            if errors:
                accuracies[f"{stat}_mae"] = sum(errors) / len(errors)
        
        return accuracies


# WebSocket endpoint handler
async def websocket_endpoint(websocket: WebSocket, monitor: GameMonitor):
    """Handle WebSocket connections for live updates"""
    await monitor.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            await monitor.handle_client_message(websocket, data)
            
    except WebSocketDisconnect:
        await monitor.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await monitor.disconnect(websocket)