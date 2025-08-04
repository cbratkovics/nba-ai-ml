#!/usr/bin/env python3
"""
Collect NBA player data for 2023-24 season
"""
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, playergamelog, leaguedashteamstats
from nba_api.stats.static import players, teams
from datetime import datetime, timedelta
import time
import os
import json
from typing import List, Dict, Any

# Configuration
SEASON = "2023-24"
OUTPUT_DIR = "data"
MIN_GAMES = 20
TOP_PLAYERS = 100

def get_top_players(season: str = SEASON, top_n: int = TOP_PLAYERS) -> List[Dict]:
    """Get top players by minutes played in the season"""
    print(f"Fetching top {top_n} players for {season} season...")
    
    from nba_api.stats.endpoints import leagueleaders
    
    # Get season leaders by minutes
    leaders = leagueleaders.LeagueLeaders(
        season=season,
        season_type_all_star="Regular Season",
        stat_category_abbreviation="MIN"
    )
    
    df = leaders.get_data_frames()[0]
    
    # Get top players by total minutes
    top_players_df = df.nlargest(top_n, 'MIN')
    
    player_list = []
    for _, row in top_players_df.iterrows():
        player_list.append({
            'id': row['PLAYER_ID'],
            'name': row['PLAYER'],
            'team': row['TEAM'],
            'games': row['GP'],
            'minutes': row['MIN']
        })
    
    print(f"Found {len(player_list)} top players")
    return player_list


def calculate_rest_days(game_dates: pd.Series) -> List[int]:
    """Calculate days of rest between games"""
    rest_days = []
    for i in range(len(game_dates)):
        if i == 0:
            rest_days.append(2)  # Assume 2+ days rest for first game
        else:
            days_diff = (game_dates.iloc[i] - game_dates.iloc[i-1]).days - 1
            rest_days.append(min(days_diff, 3))  # Cap at 3+ days
    return rest_days


def get_team_defensive_ratings() -> Dict[str, float]:
    """Get defensive ratings for all teams"""
    print("Fetching team defensive ratings...")
    
    team_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=SEASON,
        season_type_all_star="Regular Season"
    )
    
    df = team_stats.get_data_frames()[0]
    
    # Create mapping of team abbreviation to defensive rating
    def_ratings = {}
    for _, row in df.iterrows():
        # Use opponent points per game as proxy for defensive rating
        def_ratings[row['TEAM_ABBREVIATION']] = row['OPP_PTS']
    
    return def_ratings


def collect_player_games(player_id: str, player_name: str, last_n_games: int = 100) -> pd.DataFrame:
    """Collect game logs for a single player"""
    print(f"  Collecting data for {player_name}...")
    
    try:
        # Get player game logs
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=SEASON,
            season_type_all_star="Regular Season"
        )
        
        df = gamelog.get_data_frames()[0]
        
        if df.empty:
            print(f"    No data found for {player_name}")
            return pd.DataFrame()
        
        # Take last N games
        df = df.head(last_n_games)
        
        # Parse game date
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        # Extract opponent team
        df['OPPONENT'] = df['MATCHUP'].apply(lambda x: x.split()[-1])
        
        # Determine home/away
        df['HOME'] = df['MATCHUP'].apply(lambda x: 'vs.' in x).astype(int)
        
        # Calculate rest days
        df = df.sort_values('GAME_DATE')
        df['REST_DAYS'] = calculate_rest_days(df['GAME_DATE'])
        
        # Calculate back-to-back
        df['BACK_TO_BACK'] = (df['REST_DAYS'] == 0).astype(int)
        
        # Select relevant columns
        columns_to_keep = [
            'PLAYER_ID', 'GAME_DATE', 'OPPONENT', 'HOME', 'REST_DAYS', 'BACK_TO_BACK',
            'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
            'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
            'FTM', 'FTA', 'FT_PCT', 'PLUS_MINUS'
        ]
        
        df = df[columns_to_keep]
        df['PLAYER_NAME'] = player_name
        
        return df
        
    except Exception as e:
        print(f"    Error collecting data for {player_name}: {e}")
        return pd.DataFrame()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer advanced features from raw data"""
    print("Engineering features...")
    
    if df.empty:
        return df
    
    # Sort by player and date
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
    
    # Calculate rolling averages for each player
    rolling_cols = ['PTS', 'REB', 'AST', 'MIN', 'FG_PCT', 'PLUS_MINUS']
    
    for col in rolling_cols:
        if col in df.columns:
            # 3-game rolling average
            df[f'{col}_MA3'] = df.groupby('PLAYER_ID')[col].transform(
                lambda x: x.rolling(3, min_periods=1).mean().shift(1)
            )
            
            # 5-game rolling average
            df[f'{col}_MA5'] = df.groupby('PLAYER_ID')[col].transform(
                lambda x: x.rolling(5, min_periods=1).mean().shift(1)
            )
            
            # 10-game rolling average
            df[f'{col}_MA10'] = df.groupby('PLAYER_ID')[col].transform(
                lambda x: x.rolling(10, min_periods=1).mean().shift(1)
            )
    
    # Calculate usage rate proxy (FGA + 0.44*FTA + TOV) / MIN
    df['USAGE_RATE'] = ((df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['MIN']).fillna(0)
    df['USAGE_RATE'] = df['USAGE_RATE'].replace([np.inf, -np.inf], 0)
    
    # Calculate true shooting percentage
    df['TS_PCT'] = (df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))).fillna(0)
    df['TS_PCT'] = df['TS_PCT'].replace([np.inf, -np.inf], 0)
    
    # Calculate streak (consecutive games above/below average)
    df['PTS_ABOVE_AVG'] = df.groupby('PLAYER_ID').apply(
        lambda x: (x['PTS'] > x['PTS'].mean()).astype(int).values
    ).explode().values
    
    # Calculate hot/cold streak
    df['HOT_STREAK'] = df.groupby('PLAYER_ID')['PTS_ABOVE_AVG'].transform(
        lambda x: x.rolling(3, min_periods=1).sum().shift(1)
    )
    
    # Minutes trend (increasing/decreasing)
    df['MIN_TREND'] = df.groupby('PLAYER_ID')['MIN'].transform(
        lambda x: x.diff(periods=3).shift(1)
    )
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df


def main():
    """Main data collection pipeline"""
    print(f"Starting NBA data collection for {SEASON} season")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get top players
    top_players = get_top_players(season=SEASON, top_n=TOP_PLAYERS)
    
    # Collect game logs for each player
    all_games = []
    
    for i, player in enumerate(top_players, 1):
        print(f"\n[{i}/{len(top_players)}] Processing {player['name']}...")
        
        # Collect player games
        player_df = collect_player_games(
            player_id=player['id'],
            player_name=player['name'],
            last_n_games=100
        )
        
        if not player_df.empty:
            all_games.append(player_df)
        
        # Rate limiting
        time.sleep(0.5)
    
    # Combine all player data
    if all_games:
        print("\nCombining all player data...")
        combined_df = pd.concat(all_games, ignore_index=True)
        
        # Engineer features
        final_df = engineer_features(combined_df)
        
        # Get team defensive ratings
        def_ratings = get_team_defensive_ratings()
        
        # Add opponent defensive rating
        final_df['OPP_DEF_RATING'] = final_df['OPPONENT'].map(def_ratings).fillna(110)
        
        # Save to CSV
        output_file = os.path.join(OUTPUT_DIR, 'nba_players_2024.csv')
        final_df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        
        # Save metadata
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'season': SEASON,
            'num_players': len(top_players),
            'num_games': len(final_df),
            'features': list(final_df.columns),
            'players': [p['name'] for p in top_players]
        }
        
        metadata_file = os.path.join(OUTPUT_DIR, 'collection_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_file}")
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Total players collected: {len(top_players)}")
        print(f"Total games collected: {len(final_df)}")
        print(f"Date range: {final_df['GAME_DATE'].min()} to {final_df['GAME_DATE'].max()}")
        print(f"Features generated: {len(final_df.columns)}")
        print("\nTop 5 players by games:")
        player_games = final_df.groupby('PLAYER_NAME').size().sort_values(ascending=False).head()
        for player, count in player_games.items():
            print(f"  - {player}: {count} games")
        
    else:
        print("No data collected!")


if __name__ == "__main__":
    main()