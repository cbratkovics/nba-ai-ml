-- Create NBA data tables for Supabase

-- Players table
CREATE TABLE IF NOT EXISTS players (
    player_id VARCHAR(50) PRIMARY KEY,
    player_name VARCHAR(255) NOT NULL,
    team_id VARCHAR(50),
    team_abbreviation VARCHAR(10),
    position VARCHAR(20),
    jersey_number VARCHAR(10),
    height VARCHAR(20),
    weight VARCHAR(20),
    birth_date VARCHAR(50),
    country VARCHAR(100),
    school VARCHAR(255),
    draft_year INTEGER,
    draft_round INTEGER,
    draft_number INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Game logs table
CREATE TABLE IF NOT EXISTS game_logs (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    game_id VARCHAR(50) NOT NULL,
    game_date DATE NOT NULL,
    season VARCHAR(20),
    team_id VARCHAR(50),
    opponent_id VARCHAR(50),
    is_home BOOLEAN,
    minutes_played FLOAT,
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    field_goals_made INTEGER,
    field_goals_attempted INTEGER,
    field_goal_pct FLOAT,
    three_pointers_made INTEGER,
    three_pointers_attempted INTEGER,
    three_point_pct FLOAT,
    free_throws_made INTEGER,
    free_throws_attempted INTEGER,
    free_throw_pct FLOAT,
    offensive_rebounds INTEGER,
    defensive_rebounds INTEGER,
    personal_fouls INTEGER,
    plus_minus INTEGER,
    game_score FLOAT,
    UNIQUE(player_id, game_id)
);

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    team_id VARCHAR(50) PRIMARY KEY,
    team_name VARCHAR(255) NOT NULL,
    team_abbreviation VARCHAR(10) NOT NULL,
    team_city VARCHAR(100),
    team_state VARCHAR(50),
    conference VARCHAR(20),
    division VARCHAR(50),
    year_founded INTEGER,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Player features table
CREATE TABLE IF NOT EXISTS player_features (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    calculation_date DATE NOT NULL,
    pts_last_5 FLOAT,
    pts_last_10 FLOAT,
    pts_last_20 FLOAT,
    pts_season FLOAT,
    reb_last_5 FLOAT,
    reb_last_10 FLOAT,
    reb_last_20 FLOAT,
    reb_season FLOAT,
    ast_last_5 FLOAT,
    ast_last_10 FLOAT,
    ast_last_20 FLOAT,
    ast_season FLOAT,
    fg_pct_last_10 FLOAT,
    ft_pct_last_10 FLOAT,
    fg3_pct_last_10 FLOAT,
    minutes_last_10 FLOAT,
    games_played_season INTEGER,
    days_since_last_game INTEGER,
    back_to_backs_last_10 INTEGER,
    home_ppg FLOAT,
    away_ppg FLOAT,
    vs_opponent_avg_pts FLOAT,
    vs_opponent_avg_reb FLOAT,
    vs_opponent_avg_ast FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(player_id, calculation_date)
);

-- Schedule table
CREATE TABLE IF NOT EXISTS schedule (
    game_id VARCHAR(50) PRIMARY KEY,
    game_date DATE NOT NULL,
    home_team_id VARCHAR(50) NOT NULL,
    away_team_id VARCHAR(50) NOT NULL,
    home_team_score INTEGER,
    away_team_score INTEGER,
    is_completed BOOLEAN DEFAULT FALSE,
    season VARCHAR(20)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_game_logs_player_date ON game_logs(player_id, game_date DESC);
CREATE INDEX IF NOT EXISTS idx_game_logs_game_date ON game_logs(game_date);
CREATE INDEX IF NOT EXISTS idx_game_logs_season ON game_logs(season);
CREATE INDEX IF NOT EXISTS idx_player_features_date ON player_features(calculation_date);
CREATE INDEX IF NOT EXISTS idx_schedule_date ON schedule(game_date);
CREATE INDEX IF NOT EXISTS idx_schedule_teams ON schedule(home_team_id, away_team_id);

-- Create update trigger for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_players_updated_at BEFORE UPDATE ON players
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_teams_updated_at BEFORE UPDATE ON teams
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_player_features_updated_at BEFORE UPDATE ON player_features
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();