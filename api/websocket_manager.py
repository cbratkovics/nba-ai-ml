"""
WebSocket Manager for Real-time NBA Predictions
Handles live game predictions and streaming updates
"""
import json
import asyncio
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    PREDICTION = "prediction"
    GAME_UPDATE = "game_update"
    PLAYER_STATS = "player_stats"
    LIVE_ODDS = "live_odds"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"
    SUBSCRIPTION = "subscription"
    ERROR = "error"


@dataclass
class PredictionUpdate:
    """Real-time prediction update"""
    prediction_id: str
    player_id: str
    player_name: str
    game_id: str
    target: str  # PTS, REB, AST
    prediction: float
    confidence: float
    confidence_interval: tuple
    timestamp: datetime
    model_version: str
    features_used: Dict[str, float]
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['confidence_interval'] = list(self.confidence_interval)
        return data


@dataclass
class GameUpdate:
    """Live game status update"""
    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    quarter: int
    time_remaining: str
    player_stats: List[Dict]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ConnectionManager:
    """Manages WebSocket connections and subscriptions"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        self.user_metadata: Dict[str, Dict] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
    async def initialize(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis connection for pub/sub"""
        self.redis_client = await redis.from_url(redis_url)
        logger.info("WebSocket manager initialized with Redis")
        
    async def connect(self, websocket: WebSocket, client_id: str = None) -> str:
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        if not client_id:
            client_id = str(uuid.uuid4())
        
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        self.user_metadata[client_id] = {
            "connected_at": datetime.now(),
            "message_count": 0,
            "last_activity": datetime.now()
        }
        
        # Send welcome message
        await self.send_personal_message(
            client_id,
            {
                "type": MessageType.SUBSCRIPTION.value,
                "message": "Connected to NBA ML real-time predictions",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Client {client_id} connected")
        return client_id
    
    async def disconnect(self, client_id: str):
        """Handle client disconnection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.subscriptions[client_id]
            del self.user_metadata[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def subscribe(self, client_id: str, channels: List[str]):
        """Subscribe client to specific channels"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].update(channels)
            
            await self.send_personal_message(
                client_id,
                {
                    "type": MessageType.SUBSCRIPTION.value,
                    "subscribed_channels": list(self.subscriptions[client_id]),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Client {client_id} subscribed to {channels}")
    
    async def unsubscribe(self, client_id: str, channels: List[str]):
        """Unsubscribe client from channels"""
        if client_id in self.subscriptions:
            for channel in channels:
                self.subscriptions[client_id].discard(channel)
            
            await self.send_personal_message(
                client_id,
                {
                    "type": MessageType.SUBSCRIPTION.value,
                    "unsubscribed_channels": channels,
                    "remaining_channels": list(self.subscriptions[client_id]),
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def send_personal_message(self, client_id: str, message: Dict):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
                self.user_metadata[client_id]["message_count"] += 1
                self.user_metadata[client_id]["last_activity"] = datetime.now()
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                await self.disconnect(client_id)
    
    async def broadcast_to_channel(self, channel: str, message: Dict):
        """Broadcast message to all clients subscribed to channel"""
        disconnected_clients = []
        
        for client_id, channels in self.subscriptions.items():
            if channel in channels:
                try:
                    await self.send_personal_message(client_id, message)
                except WebSocketDisconnect:
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
    
    async def broadcast_prediction(self, prediction: PredictionUpdate):
        """Broadcast prediction update to relevant channels"""
        message = {
            "type": MessageType.PREDICTION.value,
            "data": prediction.to_dict()
        }
        
        # Broadcast to player channel
        await self.broadcast_to_channel(f"player:{prediction.player_id}", message)
        
        # Broadcast to game channel
        await self.broadcast_to_channel(f"game:{prediction.game_id}", message)
        
        # Broadcast to target channel (PTS, REB, AST)
        await self.broadcast_to_channel(f"target:{prediction.target}", message)
        
        # Store in Redis for persistence
        if self.redis_client:
            await self.redis_client.setex(
                f"prediction:{prediction.prediction_id}",
                3600,  # 1 hour TTL
                json.dumps(prediction.to_dict())
            )
    
    async def broadcast_game_update(self, game_update: GameUpdate):
        """Broadcast live game update"""
        message = {
            "type": MessageType.GAME_UPDATE.value,
            "data": game_update.to_dict()
        }
        
        # Broadcast to game channel
        await self.broadcast_to_channel(f"game:{game_update.game_id}", message)
        
        # Broadcast to team channels
        await self.broadcast_to_channel(f"team:{game_update.home_team}", message)
        await self.broadcast_to_channel(f"team:{game_update.away_team}", message)
    
    async def send_alert(self, alert_type: str, message: str, severity: str = "info"):
        """Send system-wide alert"""
        alert_message = {
            "type": MessageType.ALERT.value,
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        
        # Broadcast to all connected clients
        for client_id in list(self.active_connections.keys()):
            await self.send_personal_message(client_id, alert_message)
    
    async def heartbeat(self):
        """Send heartbeat to all clients"""
        while True:
            await asyncio.sleep(30)  # Every 30 seconds
            
            heartbeat_message = {
                "type": MessageType.HEARTBEAT.value,
                "timestamp": datetime.now().isoformat()
            }
            
            disconnected = []
            for client_id in list(self.active_connections.keys()):
                try:
                    await self.send_personal_message(client_id, heartbeat_message)
                except:
                    disconnected.append(client_id)
            
            for client_id in disconnected:
                await self.disconnect(client_id)
    
    def get_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "total_subscriptions": sum(len(s) for s in self.subscriptions.values()),
            "clients": {
                client_id: {
                    "subscriptions": list(channels),
                    "metadata": self.user_metadata.get(client_id, {})
                }
                for client_id, channels in self.subscriptions.items()
            }
        }


class PredictionStreamer:
    """Streams real-time predictions for live games"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
        self.active_streams: Dict[str, asyncio.Task] = {}
        
    async def start_game_stream(self, game_id: str, player_ids: List[str]):
        """Start streaming predictions for a live game"""
        
        async def stream_predictions():
            """Generate and stream predictions"""
            while game_id in self.active_streams:
                for player_id in player_ids:
                    # Generate mock prediction (would use actual model in production)
                    for target in ["PTS", "REB", "AST"]:
                        prediction = PredictionUpdate(
                            prediction_id=str(uuid.uuid4()),
                            player_id=player_id,
                            player_name=f"Player {player_id}",  # Would fetch actual name
                            game_id=game_id,
                            target=target,
                            prediction=np.random.uniform(10, 35) if target == "PTS" else np.random.uniform(3, 15),
                            confidence=np.random.uniform(0.85, 0.95),
                            confidence_interval=(
                                np.random.uniform(8, 30),
                                np.random.uniform(12, 40)
                            ),
                            timestamp=datetime.now(),
                            model_version="v1.0.0",
                            features_used={
                                "season_avg": np.random.uniform(15, 25),
                                "last_5_avg": np.random.uniform(18, 28),
                                "rest_days": np.random.randint(1, 3)
                            }
                        )
                        
                        await self.manager.broadcast_prediction(prediction)
                
                # Stream every 30 seconds during live game
                await asyncio.sleep(30)
        
        # Start streaming task
        if game_id not in self.active_streams:
            task = asyncio.create_task(stream_predictions())
            self.active_streams[game_id] = task
            logger.info(f"Started prediction stream for game {game_id}")
    
    async def stop_game_stream(self, game_id: str):
        """Stop streaming predictions for a game"""
        if game_id in self.active_streams:
            self.active_streams[game_id].cancel()
            del self.active_streams[game_id]
            logger.info(f"Stopped prediction stream for game {game_id}")
    
    async def update_live_odds(self, game_id: str, odds_data: Dict):
        """Stream live betting odds updates"""
        message = {
            "type": MessageType.LIVE_ODDS.value,
            "game_id": game_id,
            "odds": odds_data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.manager.broadcast_to_channel(f"odds:{game_id}", message)


class WebSocketHandler:
    """Handles WebSocket message processing"""
    
    def __init__(self, manager: ConnectionManager, streamer: PredictionStreamer):
        self.manager = manager
        self.streamer = streamer
        
    async def handle_message(self, client_id: str, message: Dict) -> Dict:
        """Process incoming WebSocket message"""
        
        message_type = message.get("type")
        
        if message_type == "subscribe":
            channels = message.get("channels", [])
            await self.manager.subscribe(client_id, channels)
            return {"status": "subscribed", "channels": channels}
            
        elif message_type == "unsubscribe":
            channels = message.get("channels", [])
            await self.manager.unsubscribe(client_id, channels)
            return {"status": "unsubscribed", "channels": channels}
            
        elif message_type == "start_stream":
            game_id = message.get("game_id")
            player_ids = message.get("player_ids", [])
            await self.streamer.start_game_stream(game_id, player_ids)
            return {"status": "stream_started", "game_id": game_id}
            
        elif message_type == "stop_stream":
            game_id = message.get("game_id")
            await self.streamer.stop_game_stream(game_id)
            return {"status": "stream_stopped", "game_id": game_id}
            
        elif message_type == "get_prediction":
            # On-demand prediction request
            player_id = message.get("player_id")
            target = message.get("target", "PTS")
            
            # Generate prediction (mock for now)
            prediction = PredictionUpdate(
                prediction_id=str(uuid.uuid4()),
                player_id=player_id,
                player_name=f"Player {player_id}",
                game_id=message.get("game_id", "unknown"),
                target=target,
                prediction=np.random.uniform(10, 35),
                confidence=np.random.uniform(0.85, 0.95),
                confidence_interval=(np.random.uniform(8, 30), np.random.uniform(12, 40)),
                timestamp=datetime.now(),
                model_version="v1.0.0",
                features_used={}
            )
            
            await self.manager.send_personal_message(
                client_id,
                {
                    "type": MessageType.PREDICTION.value,
                    "data": prediction.to_dict()
                }
            )
            
            return {"status": "prediction_sent"}
            
        else:
            return {"status": "error", "message": f"Unknown message type: {message_type}"}


# Global instances
connection_manager = ConnectionManager()
prediction_streamer = PredictionStreamer(connection_manager)
websocket_handler = WebSocketHandler(connection_manager, prediction_streamer)


async def websocket_endpoint(websocket: WebSocket, client_id: Optional[str] = None):
    """WebSocket endpoint for real-time predictions"""
    
    client_id = await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Process message
            response = await websocket_handler.handle_message(client_id, data)
            
            # Send response
            await connection_manager.send_personal_message(
                client_id,
                {
                    "type": "response",
                    "data": response,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    except WebSocketDisconnect:
        await connection_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await connection_manager.disconnect(client_id)