-- Store SHAP explanations and prediction narratives
CREATE TABLE IF NOT EXISTS prediction_explanations (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
    feature_importances JSONB NOT NULL,
    shap_values JSONB,
    natural_language TEXT,
    explanation_method VARCHAR(50) DEFAULT 'shap',
    model_version VARCHAR(50),
    computation_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Ensure one explanation per prediction
    CONSTRAINT unique_prediction_explanation UNIQUE(prediction_id)
);

-- Index for fast lookups
CREATE INDEX idx_explanations_prediction ON prediction_explanations(prediction_id);
CREATE INDEX idx_explanations_created ON prediction_explanations(created_at DESC);

-- Function to get top feature importances
CREATE OR REPLACE FUNCTION get_top_features(explanation_id INTEGER, top_n INTEGER DEFAULT 5)
RETURNS TABLE(feature_name TEXT, importance NUMERIC, impact TEXT) AS $$
BEGIN
    RETURN QUERY
    WITH feature_data AS (
        SELECT 
            key as feature_name,
            (value::numeric) as importance_value
        FROM prediction_explanations, 
             jsonb_each_text(feature_importances)
        WHERE id = explanation_id
    )
    SELECT 
        feature_name,
        ABS(importance_value) as importance,
        CASE 
            WHEN importance_value > 0 THEN 'positive'
            ELSE 'negative'
        END as impact
    FROM feature_data
    ORDER BY ABS(importance_value) DESC
    LIMIT top_n;
END;
$$ LANGUAGE plpgsql;

-- View for explanation analytics
CREATE OR REPLACE VIEW explanation_analytics AS
SELECT 
    DATE(e.created_at) as date,
    p.model_version,
    COUNT(*) as explanations_generated,
    AVG(e.computation_time_ms) as avg_computation_time_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY e.computation_time_ms) as median_computation_time_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY e.computation_time_ms) as p95_computation_time_ms
FROM prediction_explanations e
JOIN predictions p ON e.prediction_id = p.id
WHERE e.created_at > NOW() - INTERVAL '7 days'
GROUP BY DATE(e.created_at), p.model_version;

-- Feature importance aggregation view
CREATE MATERIALIZED VIEW IF NOT EXISTS feature_importance_summary AS
WITH feature_stats AS (
    SELECT 
        key as feature_name,
        AVG(ABS((value::numeric))) as avg_importance,
        STDDEV(ABS((value::numeric))) as std_importance,
        COUNT(*) as usage_count,
        SUM(CASE WHEN (value::numeric) > 0 THEN 1 ELSE 0 END) as positive_impact_count,
        SUM(CASE WHEN (value::numeric) < 0 THEN 1 ELSE 0 END) as negative_impact_count
    FROM prediction_explanations, 
         jsonb_each_text(feature_importances)
    WHERE created_at > NOW() - INTERVAL '30 days'
    GROUP BY key
)
SELECT 
    feature_name,
    avg_importance,
    std_importance,
    usage_count,
    positive_impact_count,
    negative_impact_count,
    CASE 
        WHEN positive_impact_count > negative_impact_count THEN 'mostly_positive'
        WHEN negative_impact_count > positive_impact_count THEN 'mostly_negative'
        ELSE 'balanced'
    END as typical_impact,
    avg_importance * usage_count as total_influence
FROM feature_stats
ORDER BY avg_importance DESC;

-- Create index for the materialized view
CREATE UNIQUE INDEX idx_feature_importance_summary ON feature_importance_summary(feature_name);

-- Table for storing anomaly alerts
CREATE TABLE IF NOT EXISTS anomaly_alerts (
    id SERIAL PRIMARY KEY,
    anomaly_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    prediction_id INTEGER REFERENCES predictions(id),
    details JSONB NOT NULL,
    alert_sent BOOLEAN DEFAULT FALSE,
    alert_sent_at TIMESTAMPTZ,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for anomaly alerts
CREATE INDEX idx_anomaly_alerts_created ON anomaly_alerts(created_at DESC);
CREATE INDEX idx_anomaly_alerts_unresolved ON anomaly_alerts(resolved, severity) WHERE resolved = FALSE;
CREATE INDEX idx_anomaly_alerts_type ON anomaly_alerts(anomaly_type, created_at DESC);

-- Function to create anomaly alert
CREATE OR REPLACE FUNCTION create_anomaly_alert(
    p_anomaly_type VARCHAR(50),
    p_severity VARCHAR(20),
    p_prediction_id INTEGER,
    p_details JSONB
) RETURNS INTEGER AS $$
DECLARE
    alert_id INTEGER;
BEGIN
    INSERT INTO anomaly_alerts (anomaly_type, severity, prediction_id, details)
    VALUES (p_anomaly_type, p_severity, p_prediction_id, p_details)
    RETURNING id INTO alert_id;
    
    -- Log critical alerts
    IF p_severity = 'critical' THEN
        RAISE NOTICE 'Critical anomaly detected: % (prediction_id: %)', p_anomaly_type, p_prediction_id;
    END IF;
    
    RETURN alert_id;
END;
$$ LANGUAGE plpgsql;