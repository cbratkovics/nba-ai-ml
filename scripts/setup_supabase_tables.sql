-- Run this in Supabase SQL editor
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    player_id VARCHAR(50),
    player_name VARCHAR(255),
    prediction_date TIMESTAMPTZ,
    game_date TIMESTAMPTZ,
    opponent_team VARCHAR(10),
    points_predicted FLOAT,
    rebounds_predicted FLOAT,
    assists_predicted FLOAT,
    points_actual FLOAT,
    rebounds_actual FLOAT,
    assists_actual FLOAT,
    confidence FLOAT,
    model_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    key_hash VARCHAR(255) UNIQUE,
    user_id VARCHAR(255),
    clerk_user_id VARCHAR(255),
    tier VARCHAR(50) DEFAULT 'free',
    monthly_limit INTEGER DEFAULT 1000,
    requests_this_month INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    target VARCHAR(20),
    accuracy FLOAT,
    mae FLOAT,
    rmse FLOAT,
    predictions_count INTEGER,
    evaluated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_predictions_user_id ON predictions(user_id);
CREATE INDEX idx_predictions_player_id ON predictions(player_id);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);
CREATE INDEX idx_api_keys_clerk_user_id ON api_keys(clerk_user_id);