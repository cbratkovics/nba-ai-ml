{#- Column list of the bronze copy of drift/<date>.json, typed, so it renders an identical
    empty relation before the first nightly run writes a report. -#}
{% macro drift_report_columns() %}
    {{ return([
        ['source_file', 'varchar'], ['run_date', 'date'], ['season', 'varchar'], ['position', 'varchar'],
        ['reference_mode', 'varchar'], ['feature_version', 'varchar'], ['model_revision', 'varchar'],
        ['n_rows', 'integer'], ['psi_threshold', 'double'], ['min_features', 'integer'],
        ['calibrated', 'boolean'], ['status', 'varchar'], ['n_flagged', 'integer'],
        ['slate_status', 'varchar'], ['no_schedule_streak', 'integer'], ['no_schedule_warn', 'boolean'],
        ['feature', 'varchar'], ['psi', 'double'], ['flagged', 'boolean'],
    ]) }}
{% endmacro %}
