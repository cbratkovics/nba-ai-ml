"""
LLM-powered insights generation for NBA players
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from openai import AsyncOpenAI
import tiktoken
import redis
import hashlib

from llmops.prompt_templates import (
    PLAYER_ANALYSIS_PROMPT,
    GAME_PREDICTION_PROMPT,
    TREND_ANALYSIS_PROMPT,
    COMPARISON_PROMPT
)

logger = logging.getLogger(__name__)


class InsightCache:
    """Cache for LLM-generated insights to reduce costs"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/2"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.cache_ttl = 21600  # 6 hours
        
    def get_cache_key(self, prompt: str, data_hash: str) -> str:
        """Generate cache key from prompt and data"""
        key_string = f"{prompt}_{data_hash}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, prompt: str, data_hash: str) -> Optional[str]:
        """Get cached insight"""
        try:
            cache_key = self.get_cache_key(prompt, data_hash)
            return self.redis_client.get(cache_key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, prompt: str, data_hash: str, insight: str):
        """Cache insight"""
        try:
            cache_key = self.get_cache_key(prompt, data_hash)
            self.redis_client.setex(cache_key, self.cache_ttl, insight)
        except Exception as e:
            logger.error(f"Cache set error: {e}")


class TokenManager:
    """Manage token usage and costs"""
    
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.daily_limit = 100000  # tokens per day
        self.cost_per_1k_tokens = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate API call cost"""
        costs = self.cost_per_1k_tokens.get(self.model, {"input": 0.01, "output": 0.03})
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost


class NBAInsightsEngine:
    """LLM-powered analysis engine for NBA insights"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.cache = InsightCache()
        self.token_manager = TokenManager(model)
        self.max_retries = 3
        
    async def generate_player_insights(self,
                                     player_data: Dict[str, Any],
                                     recent_stats: pd.DataFrame,
                                     predictions: Optional[Dict[str, float]] = None) -> str:
        """
        Generate comprehensive player insights
        
        Args:
            player_data: Player information and season stats
            recent_stats: Recent game performance data
            predictions: Future game predictions
            
        Returns:
            Natural language insights about the player
        """
        # Create data hash for caching
        data_dict = {
            "player": player_data,
            "recent": recent_stats.to_dict('records')[-10:],  # Last 10 games
            "predictions": predictions
        }
        data_hash = hashlib.md5(json.dumps(data_dict, sort_keys=True).encode()).hexdigest()
        
        # Check cache
        cached_insight = self.cache.get("player_analysis", data_hash)
        if cached_insight:
            logger.info("Using cached player insight")
            return cached_insight
        
        # Prepare context data
        context = self._prepare_player_context(player_data, recent_stats, predictions)
        
        # Generate insight
        prompt = PLAYER_ANALYSIS_PROMPT.format(**context)
        
        try:
            insight = await self._call_llm(prompt, max_tokens=1000)
            
            # Cache the result
            self.cache.set("player_analysis", data_hash, insight)
            
            return insight
            
        except Exception as e:
            logger.error(f"Error generating player insights: {e}")
            return self._generate_fallback_insight(player_data)
    
    async def generate_game_prediction_narrative(self,
                                               player_name: str,
                                               opponent: str,
                                               predictions: Dict[str, float],
                                               factors: List[Dict[str, Any]],
                                               confidence: float) -> str:
        """Generate narrative explanation for game prediction"""
        
        context = {
            "player_name": player_name,
            "opponent": opponent,
            "predicted_points": predictions.get("points", 0),
            "predicted_rebounds": predictions.get("rebounds", 0),
            "predicted_assists": predictions.get("assists", 0),
            "confidence": confidence * 100,
            "key_factors": self._format_factors(factors)
        }
        
        prompt = GAME_PREDICTION_PROMPT.format(**context)
        
        try:
            return await self._call_llm(prompt, max_tokens=500)
        except Exception as e:
            logger.error(f"Error generating prediction narrative: {e}")
            return f"{player_name} is expected to score {predictions.get('points', 0):.1f} points against {opponent}."
    
    async def analyze_performance_trends(self,
                                       player_name: str,
                                       stats_df: pd.DataFrame,
                                       period: str = "season") -> Dict[str, Any]:
        """
        Analyze player performance trends over time
        
        Args:
            player_name: Player name
            stats_df: Performance data over time
            period: Analysis period ("recent", "season", "career")
            
        Returns:
            Dictionary with trend analysis and insights
        """
        # Calculate trends
        trends = self._calculate_trends(stats_df)
        
        # Prepare context
        context = {
            "player_name": player_name,
            "period": period,
            "games_analyzed": len(stats_df),
            "trends_summary": self._format_trends(trends),
            "recent_performance": self._summarize_recent_performance(stats_df)
        }
        
        prompt = TREND_ANALYSIS_PROMPT.format(**context)
        
        try:
            trend_insight = await self._call_llm(prompt, max_tokens=800)
            
            return {
                "narrative": trend_insight,
                "trends": trends,
                "key_metrics": self._extract_key_metrics(stats_df),
                "recommendations": self._generate_recommendations(trends)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {"narrative": "Trend analysis unavailable", "trends": trends}
    
    async def compare_players(self,
                            player1_data: Dict[str, Any],
                            player2_data: Dict[str, Any],
                            metrics: List[str] = ["points", "rebounds", "assists"]) -> str:
        """Generate player comparison analysis"""
        
        context = {
            "player1_name": player1_data["name"],
            "player2_name": player2_data["name"],
            "player1_stats": self._format_player_stats(player1_data, metrics),
            "player2_stats": self._format_player_stats(player2_data, metrics),
            "comparison_metrics": ", ".join(metrics)
        }
        
        prompt = COMPARISON_PROMPT.format(**context)
        
        try:
            return await self._call_llm(prompt, max_tokens=600)
        except Exception as e:
            logger.error(f"Error generating comparison: {e}")
            return "Player comparison unavailable."
    
    async def _call_llm(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Make API call to LLM with retries and error handling"""
        
        input_tokens = self.token_manager.count_tokens(prompt)
        estimated_cost = self.token_manager.estimate_cost(input_tokens, max_tokens)
        
        logger.info(f"LLM call - Input tokens: {input_tokens}, "
                   f"Max output: {max_tokens}, Estimated cost: ${estimated_cost:.4f}")
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert NBA analyst providing insights based on statistical data. "
                                     "Be precise, informative, and engaging in your analysis. "
                                     "Focus on actionable insights and avoid speculation."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9
                )
                
                output_tokens = response.usage.completion_tokens
                actual_cost = self.token_manager.estimate_cost(input_tokens, output_tokens)
                
                logger.info(f"LLM response - Output tokens: {output_tokens}, "
                           f"Actual cost: ${actual_cost:.4f}")
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _prepare_player_context(self,
                              player_data: Dict[str, Any],
                              recent_stats: pd.DataFrame,
                              predictions: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Prepare context data for player analysis"""
        
        # Calculate recent averages
        recent_games = recent_stats.head(10)
        recent_averages = {
            "points": recent_games["PTS"].mean(),
            "rebounds": recent_games["REB"].mean(),
            "assists": recent_games["AST"].mean(),
            "fg_pct": recent_games["FG_PCT"].mean(),
            "minutes": recent_games["MIN"].mean()
        }
        
        # Season averages
        season_averages = {
            "points": player_data.get("ppg", 0),
            "rebounds": player_data.get("rpg", 0),
            "assists": player_data.get("apg", 0),
            "fg_pct": player_data.get("fg_pct", 0)
        }
        
        return {
            "player_name": player_data.get("name", "Player"),
            "position": player_data.get("position", "Unknown"),
            "team": player_data.get("team", "Unknown"),
            "season_ppg": season_averages["points"],
            "season_rpg": season_averages["rebounds"],
            "season_apg": season_averages["assists"],
            "recent_ppg": recent_averages["points"],
            "recent_rpg": recent_averages["rebounds"],
            "recent_apg": recent_averages["assists"],
            "recent_fg_pct": recent_averages["fg_pct"] * 100,
            "games_played": len(recent_stats),
            "predictions": json.dumps(predictions) if predictions else "None"
        }
    
    def _calculate_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance trends"""
        trends = {}
        
        for stat in ["PTS", "REB", "AST", "FG_PCT"]:
            if stat in df.columns:
                values = df[stat].dropna()
                if len(values) >= 5:
                    # Linear trend
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    
                    trends[stat.lower()] = {
                        "direction": "up" if slope > 0.1 else "down" if slope < -0.1 else "stable",
                        "magnitude": abs(slope),
                        "recent_avg": values.head(5).mean(),
                        "season_avg": values.mean()
                    }
        
        return trends
    
    def _format_trends(self, trends: Dict[str, Any]) -> str:
        """Format trends for prompt"""
        trend_strings = []
        for stat, trend in trends.items():
            direction = trend["direction"]
            recent = trend["recent_avg"]
            season = trend["season_avg"]
            trend_strings.append(f"{stat}: {direction} (recent: {recent:.1f}, season: {season:.1f})")
        
        return "; ".join(trend_strings)
    
    def _format_factors(self, factors: List[Dict[str, Any]]) -> str:
        """Format prediction factors for prompt"""
        factor_strings = []
        for factor in factors[:5]:  # Top 5 factors
            factor_strings.append(f"{factor['factor']}: {factor['value']} ({factor['impact']})")
        
        return "; ".join(factor_strings)
    
    def _generate_fallback_insight(self, player_data: Dict[str, Any]) -> str:
        """Generate simple fallback insight when LLM fails"""
        name = player_data.get("name", "Player")
        ppg = player_data.get("ppg", 0)
        rpg = player_data.get("rpg", 0)
        apg = player_data.get("apg", 0)
        
        return (f"{name} is averaging {ppg:.1f} points, {rpg:.1f} rebounds, "
                f"and {apg:.1f} assists per game this season. "
                f"Analysis shows consistent performance with room for growth in key areas.")
    
    def _summarize_recent_performance(self, df: pd.DataFrame) -> str:
        """Summarize recent performance"""
        recent = df.head(5)
        avg_pts = recent["PTS"].mean()
        avg_reb = recent["REB"].mean()
        avg_ast = recent["AST"].mean()
        
        return f"Recent 5-game averages: {avg_pts:.1f} pts, {avg_reb:.1f} reb, {avg_ast:.1f} ast"
    
    def _extract_key_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract key performance metrics"""
        return {
            "consistency": 1 - (df["PTS"].std() / df["PTS"].mean()),
            "efficiency": df["FG_PCT"].mean(),
            "usage": df["MIN"].mean(),
            "impact": df["PLUS_MINUS"].mean() if "PLUS_MINUS" in df.columns else 0
        }
    
    def _generate_recommendations(self, trends: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        for stat, trend in trends.items():
            if trend["direction"] == "down":
                recommendations.append(f"Focus on improving {stat} performance")
            elif trend["direction"] == "up":
                recommendations.append(f"Maintain current {stat} momentum")
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _format_player_stats(self, player_data: Dict[str, Any], metrics: List[str]) -> str:
        """Format player stats for comparison"""
        stats = []
        for metric in metrics:
            value = player_data.get(metric, 0)
            stats.append(f"{metric}: {value:.1f}")
        return ", ".join(stats)