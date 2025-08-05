-- Monitoring Views and Indexes for NBA ML Platform
-- This creates optimized views for real-time monitoring dashboards

-- Create materialized views for performance
CREATE MATERIALIZED VIEW IF NOT EXISTS prediction_metrics_hourly AS
SELECT 
    date_trunc('hour', created_at) as hour,
    model_version,
    COUNT(*) as prediction_count,
    AVG(confidence) as avg_confidence,
    AVG(points_predicted - points_actual) as avg_points_error,
    AVG(ABS(points_predicted - points_actual)) as avg_points_mae,
    STDDEV(points_predicted - points_actual) as points_error_std,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY confidence) as median_confidence,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY confidence) as p95_confidence,
    COUNT(CASE WHEN points_actual IS NOT NULL THEN 1 END) as evaluated_count
FROM predictions
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY date_trunc('hour', created_at), model_version;

-- Create index for fast refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_metrics_hourly 
ON prediction_metrics_hourly(hour, model_version);

-- Real-time accuracy view
CREATE OR REPLACE VIEW model_accuracy_realtime AS
SELECT 
    model_version,
    COUNT(*) as total_predictions,
    COUNT(CASE WHEN points_actual IS NOT NULL THEN 1 END) as evaluated_predictions,
    AVG(ABS(points_predicted - points_actual)) as mae_points,
    AVG(ABS(rebounds_predicted - rebounds_actual)) as mae_rebounds,
    AVG(ABS(assists_predicted - assists_actual)) as mae_assists,
    STDDEV(ABS(points_predicted - points_actual)) as mae_points_std,
    CORR(points_predicted, points_actual) as correlation_points,
    CORR(rebounds_predicted, rebounds_actual) as correlation_rebounds,
    CORR(assists_predicted, assists_actual) as correlation_assists,
    AVG(confidence) as avg_confidence,
    MIN(created_at) as first_prediction,
    MAX(created_at) as last_prediction
FROM predictions
WHERE created_at > NOW() - INTERVAL '24 hours'
    AND points_actual IS NOT NULL
GROUP BY model_version;

-- Player performance tracking view
CREATE OR REPLACE VIEW player_prediction_accuracy AS
SELECT 
    player_id,
    player_name,
    COUNT(*) as predictions_made,
    COUNT(CASE WHEN points_actual IS NOT NULL THEN 1 END) as evaluated_count,
    AVG(ABS(points_predicted - points_actual)) as avg_points_error,
    AVG(ABS(rebounds_predicted - rebounds_actual)) as avg_rebounds_error,
    AVG(ABS(assists_predicted - assists_actual)) as avg_assists_error,
    AVG(confidence) as avg_confidence,
    STDDEV(points_predicted) as points_volatility,
    MAX(created_at) as last_prediction,
    COUNT(DISTINCT DATE(created_at)) as days_predicted
FROM predictions
WHERE points_actual IS NOT NULL
GROUP BY player_id, player_name;

-- Team-level accuracy view
CREATE OR REPLACE VIEW team_prediction_accuracy AS
SELECT 
    opponent_team,
    COUNT(*) as games_predicted,
    AVG(ABS(points_predicted - points_actual)) as avg_mae_points,
    STDDEV(ABS(points_predicted - points_actual)) as mae_std,
    AVG(confidence) as avg_confidence,
    COUNT(DISTINCT player_id) as unique_players
FROM predictions
WHERE points_actual IS NOT NULL
    AND created_at > NOW() - INTERVAL '30 days'
GROUP BY opponent_team;

-- Anomaly detection view
CREATE OR REPLACE VIEW prediction_anomalies AS
WITH stats AS (
    SELECT 
        AVG(points_predicted) as mean_points,
        STDDEV(points_predicted) as std_points,
        AVG(rebounds_predicted) as mean_rebounds,
        STDDEV(rebounds_predicted) as std_rebounds,
        AVG(assists_predicted) as mean_assists,
        STDDEV(assists_predicted) as std_assists,
        AVG(confidence) as mean_confidence,
        STDDEV(confidence) as std_confidence
    FROM predictions
    WHERE created_at > NOW() - INTERVAL '7 days'
)
SELECT 
    p.id,
    p.player_id,
    p.player_name,
    p.game_date,
    p.points_predicted,
    p.rebounds_predicted,
    p.assists_predicted,
    p.confidence,
    p.created_at,
    ABS(p.points_predicted - s.mean_points) / NULLIF(s.std_points, 0) as points_z_score,
    ABS(p.rebounds_predicted - s.mean_rebounds) / NULLIF(s.std_rebounds, 0) as rebounds_z_score,
    ABS(p.assists_predicted - s.mean_assists) / NULLIF(s.std_assists, 0) as assists_z_score,
    ABS(p.confidence - s.mean_confidence) / NULLIF(s.std_confidence, 0) as confidence_z_score,
    CASE 
        WHEN p.points_predicted < 0 OR p.points_predicted > 60 THEN 'impossible_value'
        WHEN ABS(p.points_predicted - s.mean_points) > 3 * s.std_points THEN 'statistical_outlier'
        WHEN p.confidence < 0.3 THEN 'low_confidence'
        WHEN p.confidence > 0.99 THEN 'suspiciously_high_confidence'
        ELSE 'normal'
    END as anomaly_type
FROM predictions p, stats s
WHERE p.created_at > NOW() - INTERVAL '24 hours';

-- Feature drift monitoring view
CREATE MATERIALIZED VIEW IF NOT EXISTS feature_drift_daily AS
SELECT 
    DATE(created_at) as date,
    AVG(points_predicted) as avg_points_pred,
    AVG(rebounds_predicted) as avg_rebounds_pred,
    AVG(assists_predicted) as avg_assists_pred,
    AVG(confidence) as avg_confidence,
    STDDEV(points_predicted) as std_points_pred,
    STDDEV(confidence) as std_confidence,
    COUNT(*) as prediction_count,
    COUNT(DISTINCT player_id) as unique_players,
    COUNT(DISTINCT model_version) as model_versions_used
FROM predictions
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at);

-- Create index for drift view
CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_drift_daily ON feature_drift_daily(date);

-- Model comparison view
CREATE OR REPLACE VIEW model_comparison AS
SELECT 
    m1.model_version as model_a,
    m2.model_version as model_b,
    m1.mae_points as model_a_mae,
    m2.mae_points as model_b_mae,
    m1.mae_points - m2.mae_points as mae_difference,
    (m1.mae_points - m2.mae_points) / m1.mae_points * 100 as relative_improvement,
    m1.total_predictions as model_a_predictions,
    m2.total_predictions as model_b_predictions
FROM model_accuracy_realtime m1
CROSS JOIN model_accuracy_realtime m2
WHERE m1.model_version < m2.model_version;

-- Create performance indexes
CREATE INDEX IF NOT EXISTS idx_predictions_created_at_model 
ON predictions(created_at DESC, model_version);

CREATE INDEX IF NOT EXISTS idx_predictions_player_accuracy 
ON predictions(player_id, points_actual, points_predicted)
WHERE points_actual IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_predictions_anomaly_detection
ON predictions(created_at DESC, points_predicted, confidence);

CREATE INDEX IF NOT EXISTS idx_predictions_team_analysis
ON predictions(opponent_team, created_at DESC)
WHERE points_actual IS NOT NULL;

-- Create a function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_monitoring_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY prediction_metrics_hourly;
    REFRESH MATERIALIZED VIEW CONCURRENTLY feature_drift_daily;
END;
$$ LANGUAGE plpgsql;

-- Create a scheduled job to refresh views (requires pg_cron extension)
-- This should be run after enabling pg_cron in Supabase dashboard
-- SELECT cron.schedule('refresh-monitoring-views', '*/10 * * * *', 'SELECT refresh_monitoring_views();');