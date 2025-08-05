"""
Natural language narrative generator for prediction explanations
"""
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class NarrativeGenerator:
    """Generate human-readable explanations for NBA predictions"""
    
    def __init__(self):
        # Feature category mappings
        self.feature_categories = {
            "recent_performance": ["pts_last_10", "pts_last_5", "pts_last_3", 
                                 "reb_last_10", "ast_last_10"],
            "season_stats": ["pts_season_avg", "reb_season_avg", "ast_season_avg",
                           "min_season_avg", "fg_pct_season"],
            "matchup": ["vs_opponent_avg", "vs_opponent_last_3", "team_vs_team"],
            "game_context": ["home_away", "days_rest", "back_to_back", "game_time"],
            "team_factors": ["team_pace", "opponent_def_rating", "team_off_rating"],
            "usage": ["usage_rate", "touches_per_game", "time_of_possession"]
        }
        
        # Impact descriptors
        self.impact_descriptors = {
            "very_high": ["significantly", "substantially", "considerably"],
            "high": ["notably", "meaningfully", "importantly"],
            "medium": ["moderately", "somewhat", "partially"],
            "low": ["slightly", "marginally", "minimally"]
        }
    
    def generate_narrative(self, 
                         player_name: str,
                         prediction: Dict[str, float],
                         shap_values: Dict[str, float],
                         features: Dict[str, float],
                         confidence: float = 0.0) -> str:
        """Create comprehensive natural language explanation"""
        
        # Start with prediction summary
        narrative = self._generate_prediction_summary(player_name, prediction, confidence)
        
        # Add key factors
        narrative += " " + self._explain_key_factors(shap_values, features)
        
        # Add contextual insights
        narrative += " " + self._add_contextual_insights(features, shap_values)
        
        # Add confidence explanation if provided
        if confidence > 0:
            narrative += " " + self._explain_confidence(confidence, features)
        
        return narrative
    
    def generate_comparison_narrative(self,
                                    player_name: str,
                                    current_prediction: Dict[str, float],
                                    baseline: Dict[str, float],
                                    key_differences: Dict[str, float]) -> str:
        """Generate narrative comparing prediction to baseline"""
        
        points_diff = current_prediction.get("points", 0) - baseline.get("points", 0)
        
        if abs(points_diff) < 1:
            narrative = f"{player_name} is expected to perform at their typical level with {current_prediction['points']:.1f} points."
        elif points_diff > 0:
            narrative = f"{player_name} is projected to exceed their baseline by {points_diff:.1f} points, scoring {current_prediction['points']:.1f}."
        else:
            narrative = f"{player_name} may underperform by {abs(points_diff):.1f} points, projected at {current_prediction['points']:.1f}."
        
        # Explain why
        narrative += " " + self._explain_differences(key_differences)
        
        return narrative
    
    def _generate_prediction_summary(self, 
                                   player_name: str,
                                   prediction: Dict[str, float],
                                   confidence: float) -> str:
        """Generate opening summary of prediction"""
        points = prediction.get("points", 0)
        rebounds = prediction.get("rebounds", 0)
        assists = prediction.get("assists", 0)
        
        # Determine performance level
        if points >= 30:
            performance = "an exceptional"
        elif points >= 25:
            performance = "a strong"
        elif points >= 20:
            performance = "a solid"
        elif points >= 15:
            performance = "a moderate"
        else:
            performance = "a below-average"
        
        summary = f"{player_name} is projected to have {performance} game with "
        summary += f"{points:.1f} points, {rebounds:.1f} rebounds, and {assists:.1f} assists."
        
        return summary
    
    def _explain_key_factors(self, 
                           shap_values: Dict[str, float],
                           features: Dict[str, float]) -> str:
        """Explain the most important factors"""
        # Sort by absolute SHAP value
        sorted_factors = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        if not sorted_factors:
            return "Multiple factors contribute to this prediction."
        
        # Group factors by category
        categorized_factors = self._categorize_factors(sorted_factors)
        
        explanations = []
        
        for category, factors in categorized_factors.items():
            if category == "recent_performance":
                explanations.append(self._explain_recent_performance(factors, features))
            elif category == "matchup":
                explanations.append(self._explain_matchup(factors, features))
            elif category == "game_context":
                explanations.append(self._explain_game_context(factors, features))
            elif category == "season_stats":
                explanations.append(self._explain_season_stats(factors, features))
        
        if len(explanations) > 1:
            return "The key factors are: " + "; ".join(explanations) + "."
        elif explanations:
            return explanations[0]
        else:
            return "Multiple statistical factors contribute to this prediction."
    
    def _categorize_factors(self, 
                          factors: List[Tuple[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
        """Group factors by category"""
        categorized = {}
        
        for feature, value in factors:
            for category, features in self.feature_categories.items():
                if feature in features:
                    if category not in categorized:
                        categorized[category] = []
                    categorized[category].append((feature, value))
                    break
        
        return categorized
    
    def _explain_recent_performance(self, 
                                  factors: List[Tuple[str, float]],
                                  features: Dict[str, float]) -> str:
        """Explain recent performance factors"""
        explanations = []
        
        for feature, impact in factors:
            feature_value = features.get(feature, 0)
            impact_desc = self._get_impact_descriptor(impact)
            
            if "pts_last_10" in feature:
                explanations.append(f"recent 10-game average of {feature_value:.1f} points {impact_desc} influences the projection")
            elif "pts_last_5" in feature:
                explanations.append(f"last 5 games ({feature_value:.1f} pts avg) shows {'improving' if impact > 0 else 'declining'} form")
            elif "pts_last_3" in feature:
                explanations.append(f"very recent performance ({feature_value:.1f} pts in last 3) is {'encouraging' if impact > 0 else 'concerning'}")
        
        if explanations:
            return "recent performance " + " and ".join(explanations)
        return ""
    
    def _explain_matchup(self, 
                       factors: List[Tuple[str, float]],
                       features: Dict[str, float]) -> str:
        """Explain matchup-specific factors"""
        explanations = []
        
        for feature, impact in factors:
            feature_value = features.get(feature, 0)
            
            if "vs_opponent_avg" in feature:
                if feature_value > 20:
                    explanations.append(f"historically strong performance against this opponent ({feature_value:.1f} pts career avg)")
                else:
                    explanations.append(f"typically struggles against this opponent ({feature_value:.1f} pts career avg)")
            elif "opponent_def_rating" in feature:
                if feature_value > 110:
                    explanations.append("facing a weak defensive team")
                else:
                    explanations.append("facing a strong defensive unit")
        
        if explanations:
            return "matchup factors include " + " and ".join(explanations)
        return ""
    
    def _explain_game_context(self, 
                            factors: List[Tuple[str, float]],
                            features: Dict[str, float]) -> str:
        """Explain game context factors"""
        explanations = []
        
        for feature, impact in factors:
            feature_value = features.get(feature, 0)
            
            if feature == "days_rest":
                if feature_value == 0:
                    explanations.append("playing on the second night of a back-to-back")
                elif feature_value >= 3:
                    explanations.append(f"well-rested with {int(feature_value)} days off")
                else:
                    explanations.append(f"{int(feature_value)} day{'s' if feature_value > 1 else ''} rest")
            
            elif feature == "home_away":
                location = "at home" if feature_value == 1 else "on the road"
                advantage = "advantage" if (feature_value == 1 and impact > 0) else "challenge"
                explanations.append(f"playing {location} provides a {advantage}")
        
        if explanations:
            return "game situation includes " + " and ".join(explanations)
        return ""
    
    def _explain_season_stats(self, 
                            factors: List[Tuple[str, float]],
                            features: Dict[str, float]) -> str:
        """Explain season-long statistics"""
        explanations = []
        
        for feature, impact in factors:
            feature_value = features.get(feature, 0)
            
            if "season_avg" in feature:
                stat_type = "points" if "pts" in feature else "performance"
                if impact > 0:
                    explanations.append(f"strong season average ({feature_value:.1f} {stat_type}) supports higher projection")
                else:
                    explanations.append(f"season average ({feature_value:.1f} {stat_type}) suggests moderated expectations")
        
        if explanations:
            return explanations[0]
        return ""
    
    def _add_contextual_insights(self, 
                               features: Dict[str, float],
                               shap_values: Dict[str, float]) -> str:
        """Add additional context and insights"""
        insights = []
        
        # Check for extreme rest situations
        days_rest = features.get("days_rest", 1)
        if days_rest == 0:
            insights.append("Fatigue from back-to-back games may impact performance.")
        elif days_rest > 4:
            insights.append("Extended rest could lead to rust or fresh legs.")
        
        # Check for revenge game narrative
        if features.get("vs_opponent_last_game_diff", 0) < -15:
            insights.append("Potential revenge game after previous poor showing.")
        
        # Usage rate insights
        usage = features.get("usage_rate", 25)
        if usage > 30:
            insights.append("High usage rate suggests heavy offensive load.")
        elif usage < 20:
            insights.append("Lower usage may limit scoring opportunities.")
        
        if insights:
            return " ".join(insights)
        return ""
    
    def _explain_confidence(self, confidence: float, features: Dict[str, float]) -> str:
        """Explain prediction confidence level"""
        if confidence >= 0.9:
            return "This prediction has very high confidence based on consistent patterns."
        elif confidence >= 0.8:
            return "The model shows strong confidence in this projection."
        elif confidence >= 0.7:
            return "This is a moderately confident prediction with some uncertainty."
        else:
            return "Lower confidence suggests higher variability possible in actual performance."
    
    def _explain_differences(self, differences: Dict[str, float]) -> str:
        """Explain what changed between predictions"""
        explanations = []
        
        sorted_diffs = sorted(
            differences.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        for feature, diff in sorted_diffs:
            if abs(diff) > 0.1:
                direction = "improved" if diff > 0 else "declined"
                explanations.append(f"{self._humanize_feature(feature)} has {direction}")
        
        if explanations:
            return "This is primarily due to: " + ", ".join(explanations) + "."
        return ""
    
    def _get_impact_descriptor(self, impact: float) -> str:
        """Get appropriate descriptor for impact magnitude"""
        abs_impact = abs(impact)
        
        if abs_impact > 5:
            category = "very_high"
        elif abs_impact > 3:
            category = "high"
        elif abs_impact > 1:
            category = "medium"
        else:
            category = "low"
        
        descriptors = self.impact_descriptors.get(category, ["slightly"])
        return descriptors[0]  # Could randomize for variety
    
    def _humanize_feature(self, feature_name: str) -> str:
        """Convert feature name to human-readable format"""
        replacements = {
            "pts": "points",
            "reb": "rebounds", 
            "ast": "assists",
            "min": "minutes",
            "avg": "average",
            "pct": "percentage",
            "_": " ",
            "last": "in last",
            "vs": "versus"
        }
        
        result = feature_name.lower()
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        # Clean up
        result = " ".join(result.split())
        
        return result