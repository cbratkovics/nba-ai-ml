"""
NBA ML Python SDK
Client library for NBA ML prediction API
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import websockets
from urllib.parse import urljoin
import hashlib
import hmac
import time

__version__ = "1.0.0"
__author__ = "NBA ML Team"

logger = logging.getLogger(__name__)


class PredictionTarget(Enum):
    """Prediction targets"""
    POINTS = "points"
    REBOUNDS = "rebounds"
    ASSISTS = "assists"
    ALL = "all"


@dataclass
class PredictionRequest:
    """Prediction request model"""
    player_id: str
    opponent_team: str
    is_home: bool
    days_rest: int = 2
    season_avg_points: Optional[float] = None
    season_avg_rebounds: Optional[float] = None
    season_avg_assists: Optional[float] = None
    last_5_games_avg_points: Optional[float] = None
    last_5_games_avg_rebounds: Optional[float] = None
    last_5_games_avg_assists: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PredictionResponse:
    """Prediction response model"""
    prediction: float
    confidence: float
    confidence_interval: tuple
    model_version: str
    features_used: Dict[str, float]
    timestamp: datetime
    cache_hit: bool
    response_time_ms: float
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PredictionResponse':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['confidence_interval'] = tuple(data['confidence_interval'])
        return cls(**data)


class NBAMLClient:
    """Main client for NBA ML API"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.nba-predictions.com",
        timeout: int = 30,
        max_retries: int = 3,
        enable_caching: bool = True
    ):
        """
        Initialize NBA ML client
        
        Args:
            api_key: API authentication key
            base_url: Base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            enable_caching: Enable client-side caching
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_caching = enable_caching
        self._cache: Dict[str, Any] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self._rate_limit_remaining = 100
        self._rate_limit_reset = time.time()
        
        logger.info(f"NBA ML Client initialized (v{__version__})")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if not self._session:
            self._session = aiohttp.ClientSession(
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
    
    async def close(self):
        """Close client session"""
        if self._session:
            await self._session.close()
            self._session = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        timestamp = str(int(time.time()))
        signature = self._generate_signature(timestamp)
        
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Timestamp": timestamp,
            "X-Signature": signature,
            "Content-Type": "application/json",
            "User-Agent": f"NBA-ML-SDK-Python/{__version__}"
        }
    
    def _generate_signature(self, timestamp: str) -> str:
        """Generate HMAC signature for request"""
        message = f"{timestamp}:{self.api_key}"
        signature = hmac.new(
            self.api_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key for request"""
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{endpoint}:{params_str}".encode()).hexdigest()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict:
        """Make HTTP request with retries"""
        await self._ensure_session()
        
        url = urljoin(self.base_url, endpoint)
        
        # Check cache
        if self.enable_caching and method == "GET":
            cache_key = self._get_cache_key(endpoint, kwargs.get("params", {}))
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                if cached['expires'] > time.time():
                    logger.debug(f"Cache hit for {endpoint}")
                    return cached['data']
        
        # Check rate limit
        if time.time() < self._rate_limit_reset and self._rate_limit_remaining <= 0:
            wait_time = self._rate_limit_reset - time.time()
            logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        # Make request with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with self._session.request(method, url, **kwargs) as response:
                    # Update rate limit info
                    self._rate_limit_remaining = int(
                        response.headers.get('X-RateLimit-Remaining', 100)
                    )
                    self._rate_limit_reset = float(
                        response.headers.get('X-RateLimit-Reset', time.time() + 60)
                    )
                    
                    # Handle response
                    if response.status == 200:
                        data = await response.json()
                        
                        # Cache successful GET requests
                        if self.enable_caching and method == "GET":
                            self._cache[cache_key] = {
                                'data': data,
                                'expires': time.time() + 300  # 5 minute cache
                            }
                        
                        return data
                    elif response.status == 429:
                        # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited, retry after {retry_after}s")
                        await asyncio.sleep(retry_after)
                    else:
                        error_data = await response.text()
                        raise Exception(f"API error {response.status}: {error_data}")
                        
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
        
        raise Exception(f"Request failed after {self.max_retries} attempts: {last_error}")
    
    async def predict(
        self,
        request: Union[PredictionRequest, Dict],
        target: PredictionTarget = PredictionTarget.POINTS
    ) -> PredictionResponse:
        """
        Get prediction for player performance
        
        Args:
            request: Prediction request data
            target: Target to predict (points, rebounds, assists)
            
        Returns:
            PredictionResponse object
        """
        if isinstance(request, PredictionRequest):
            request_data = request.to_dict()
        else:
            request_data = request
        
        endpoint = f"/api/v1/predict/{target.value}"
        
        start_time = time.time()
        response = await self._request("POST", endpoint, json=request_data)
        response_time = (time.time() - start_time) * 1000
        
        response['response_time_ms'] = response_time
        
        return PredictionResponse.from_dict(response)
    
    async def batch_predict(
        self,
        requests: List[Union[PredictionRequest, Dict]],
        target: PredictionTarget = PredictionTarget.POINTS
    ) -> List[PredictionResponse]:
        """
        Get batch predictions
        
        Args:
            requests: List of prediction requests
            target: Target to predict
            
        Returns:
            List of PredictionResponse objects
        """
        batch_data = []
        for req in requests:
            if isinstance(req, PredictionRequest):
                batch_data.append(req.to_dict())
            else:
                batch_data.append(req)
        
        endpoint = f"/api/v1/predict/batch/{target.value}"
        
        response = await self._request("POST", endpoint, json={"predictions": batch_data})
        
        return [PredictionResponse.from_dict(r) for r in response['results']]
    
    async def get_player_stats(self, player_id: str) -> Dict:
        """Get player statistics"""
        endpoint = f"/api/v1/players/{player_id}/stats"
        return await self._request("GET", endpoint)
    
    async def get_model_info(self, model_name: str = None) -> Dict:
        """Get model information"""
        endpoint = "/api/v1/models"
        if model_name:
            endpoint += f"/{model_name}"
        return await self._request("GET", endpoint)
    
    async def get_live_games(self) -> List[Dict]:
        """Get list of live games"""
        endpoint = "/api/v1/games/live"
        return await self._request("GET", endpoint)
    
    async def health_check(self) -> Dict:
        """Check API health status"""
        endpoint = "/health"
        return await self._request("GET", endpoint)


class NBAMLWebSocket:
    """WebSocket client for real-time predictions"""
    
    def __init__(
        self,
        api_key: str,
        ws_url: str = "wss://api.nba-predictions.com/ws",
        auto_reconnect: bool = True
    ):
        """
        Initialize WebSocket client
        
        Args:
            api_key: API authentication key
            ws_url: WebSocket URL
            auto_reconnect: Automatically reconnect on disconnect
        """
        self.api_key = api_key
        self.ws_url = ws_url
        self.auto_reconnect = auto_reconnect
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._handlers: Dict[str, List[Callable]] = {}
        self._subscriptions: Set[str] = set()
        self._running = False
    
    async def connect(self):
        """Connect to WebSocket server"""
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        self._ws = await websockets.connect(self.ws_url, extra_headers=headers)
        self._running = True
        
        # Start message handler
        asyncio.create_task(self._handle_messages())
        
        logger.info("WebSocket connected")
    
    async def disconnect(self):
        """Disconnect from WebSocket server"""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("WebSocket disconnected")
    
    async def _handle_messages(self):
        """Handle incoming WebSocket messages"""
        while self._running and self._ws:
            try:
                message = await self._ws.recv()
                data = json.loads(message)
                
                message_type = data.get('type')
                if message_type in self._handlers:
                    for handler in self._handlers[message_type]:
                        asyncio.create_task(handler(data))
                        
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                if self.auto_reconnect:
                    await asyncio.sleep(5)
                    await self.connect()
                else:
                    self._running = False
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
    
    async def subscribe(self, channels: List[str]):
        """Subscribe to channels"""
        if not self._ws:
            raise Exception("WebSocket not connected")
        
        await self._ws.send(json.dumps({
            "type": "subscribe",
            "channels": channels
        }))
        
        self._subscriptions.update(channels)
        logger.info(f"Subscribed to channels: {channels}")
    
    async def unsubscribe(self, channels: List[str]):
        """Unsubscribe from channels"""
        if not self._ws:
            raise Exception("WebSocket not connected")
        
        await self._ws.send(json.dumps({
            "type": "unsubscribe",
            "channels": channels
        }))
        
        for channel in channels:
            self._subscriptions.discard(channel)
        
        logger.info(f"Unsubscribed from channels: {channels}")
    
    def on(self, event: str, handler: Callable):
        """Register event handler"""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
    
    def off(self, event: str, handler: Callable):
        """Remove event handler"""
        if event in self._handlers:
            self._handlers[event].remove(handler)
    
    async def request_prediction(self, player_id: str, target: str = "PTS"):
        """Request real-time prediction"""
        if not self._ws:
            raise Exception("WebSocket not connected")
        
        await self._ws.send(json.dumps({
            "type": "get_prediction",
            "player_id": player_id,
            "target": target
        }))


# Convenience functions
async def quick_predict(
    api_key: str,
    player_id: str,
    opponent_team: str,
    **kwargs
) -> PredictionResponse:
    """Quick prediction without managing client"""
    async with NBAMLClient(api_key) as client:
        request = PredictionRequest(
            player_id=player_id,
            opponent_team=opponent_team,
            is_home=kwargs.get('is_home', True),
            **kwargs
        )
        return await client.predict(request)


# Example usage
async def example_usage():
    """Example usage of NBA ML SDK"""
    
    # Initialize client
    client = NBAMLClient(api_key="your-api-key")
    
    # Single prediction
    request = PredictionRequest(
        player_id="203999",  # Nikola Jokic
        opponent_team="LAL",
        is_home=True,
        days_rest=2,
        season_avg_points=25.5,
        season_avg_rebounds=11.2,
        season_avg_assists=8.1
    )
    
    prediction = await client.predict(request, PredictionTarget.POINTS)
    print(f"Predicted points: {prediction.prediction:.1f} (±{prediction.confidence:.1f})")
    
    # Batch predictions
    requests = [
        PredictionRequest(player_id="203999", opponent_team="LAL", is_home=True),
        PredictionRequest(player_id="201939", opponent_team="BOS", is_home=False),  # Stephen Curry
    ]
    
    predictions = await client.batch_predict(requests, PredictionTarget.POINTS)
    for pred in predictions:
        print(f"Prediction: {pred.prediction:.1f}")
    
    # WebSocket streaming
    ws_client = NBAMLWebSocket(api_key="your-api-key")
    
    # Register handlers
    async def on_prediction(data):
        print(f"New prediction: {data}")
    
    ws_client.on('prediction', on_prediction)
    
    # Connect and subscribe
    await ws_client.connect()
    await ws_client.subscribe(['player:203999', 'game:12345'])
    
    # Keep running
    await asyncio.sleep(60)
    
    # Cleanup
    await ws_client.disconnect()
    await client.close()


if __name__ == "__main__":
    asyncio.run(example_usage())