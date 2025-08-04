"""
Optimized prompt templates for NBA insights generation
"""

PLAYER_ANALYSIS_PROMPT = """
Analyze the performance of NBA player {player_name} based on the following statistical data:

## Player Information
- Name: {player_name}
- Position: {position}
- Team: {team}

## Season Statistics
- Points per game: {season_ppg:.1f}
- Rebounds per game: {season_rpg:.1f}
- Assists per game: {season_apg:.1f}

## Recent Performance (Last 10 games)
- Points per game: {recent_ppg:.1f}
- Rebounds per game: {recent_rpg:.1f}
- Assists per game: {recent_apg:.1f}
- Field goal percentage: {recent_fg_pct:.1f}%
- Games analyzed: {games_played}

## Future Predictions
{predictions}

Please provide a comprehensive analysis that includes:
1. Current performance assessment
2. Comparison between recent and season averages
3. Key strengths and areas for improvement
4. Performance trends and patterns
5. Outlook for upcoming games

Keep the analysis concise, data-driven, and insightful. Focus on actionable observations that would be valuable for fantasy players, coaches, or analysts.
"""

GAME_PREDICTION_PROMPT = """
Generate a compelling narrative explanation for the following NBA game prediction:

## Player: {player_name}
## Opponent: {opponent}
## Predictions:
- Points: {predicted_points:.1f}
- Rebounds: {predicted_rebounds:.1f}
- Assists: {predicted_assists:.1f}
- Confidence: {confidence:.0f}%

## Key Factors Influencing Prediction:
{key_factors}

Create a 2-3 paragraph explanation that:
1. Explains why these specific numbers are predicted
2. Highlights the most important factors affecting performance
3. Discusses the confidence level and what drives it
4. Provides context about the matchup and game situation

Write in an engaging, analytical tone suitable for sports analysts and fantasy players. Be specific about the data points that support the prediction.
"""

TREND_ANALYSIS_PROMPT = """
Analyze the performance trends for {player_name} over the {period} period:

## Analysis Period: {period}
## Games Analyzed: {games_analyzed}

## Statistical Trends:
{trends_summary}

## Recent Performance Summary:
{recent_performance}

Provide a detailed trend analysis that covers:
1. Performance trajectory over the analyzed period
2. Identification of hot and cold streaks
3. Consistency and volatility patterns
4. Factors that may be driving performance changes
5. Predictions for continued trends
6. Recommendations for fantasy/betting consideration

Structure your analysis with clear sections and use specific data points to support your observations. Highlight any significant changes or inflection points in performance.
"""

COMPARISON_PROMPT = """
Compare the performance of two NBA players across key statistical categories:

## Player 1: {player1_name}
{player1_stats}

## Player 2: {player2_name}
{player2_stats}

## Comparison Metrics: {comparison_metrics}

Provide a comprehensive comparison that includes:
1. Head-to-head statistical comparison
2. Strengths and weaknesses of each player
3. Different playing styles and roles
4. Situational advantages for each player
5. Overall assessment of who performs better in specific contexts

Be objective and data-driven in your analysis. Highlight nuanced differences that go beyond basic statistics and consider factors like consistency, clutch performance, and team context.
"""

INJURY_IMPACT_PROMPT = """
Analyze the potential impact of injury concerns on {player_name}'s performance:

## Player: {player_name}
## Injury Status: {injury_status}
## Recent Performance: {recent_stats}
## Historical Context: {injury_history}

Provide analysis covering:
1. How this type of injury typically affects performance
2. Expected timeline for recovery/adjustment
3. Performance patterns during injury recovery
4. Risk factors for re-injury or compensation injuries
5. Recommendations for fantasy/betting consideration

Be cautious and evidence-based in your assessment. Avoid speculation beyond what the data and medical knowledge support.
"""

MATCHUP_ANALYSIS_PROMPT = """
Analyze the matchup dynamics for {player_name} against {opponent_team}:

## Player: {player_name} ({player_team})
## Opponent: {opponent_team}
## Historical Matchup Data: {historical_performance}
## Opponent Defensive Rankings: {defensive_stats}
## Game Context: {game_context}

Provide matchup analysis including:
1. Historical performance against this opponent
2. How opponent's defensive style affects the player
3. Pace and style considerations
4. Key individual matchups to watch
5. Expected game script and its impact
6. Specific advantages or challenges for the player

Focus on tactical and strategic elements that influence individual performance in this specific matchup.
"""

FANTASY_RECOMMENDATION_PROMPT = """
Generate fantasy basketball recommendations for {player_name}:

## Player Stats: {player_stats}
## Upcoming Schedule: {schedule}
## Current Form: {recent_form}
## Salary/Price: {fantasy_price}
## Ownership Percentage: {ownership}

Provide fantasy advice covering:
1. Overall recommendation (start/sit/consider)
2. Ceiling and floor projections
3. Value assessment at current price
4. Ownership considerations for tournaments
5. Correlation plays and stack opportunities
6. Risk factors to monitor

Tailor recommendations for different contest types (cash games vs. tournaments) and provide clear reasoning for each recommendation.
"""

TEAM_IMPACT_PROMPT = """
Analyze how team dynamics and lineup changes affect {player_name}'s performance:

## Player: {player_name}
## Team Context: {team_stats}
## Lineup Changes: {lineup_changes}
## Usage Patterns: {usage_data}
## Team Performance: {team_record}

Analyze the following aspects:
1. How player's role has evolved with team changes
2. Impact of key teammates being in/out of lineup
3. Usage rate trends and opportunity changes
4. Team performance correlation with individual stats
5. Coaching tendencies and their effect on player usage
6. Predictions for future role based on team direction

Focus on actionable insights about how team context drives individual performance.
"""