{#- Column lists of the nightly product files (nba/predict/slate.py OUTPUT_COLUMNS and
    nba/predict/residuals.py RESIDUAL_COLUMNS), typed, so the optional bronze families render
    an identical empty relation before the first in-season run. -#}
{% macro prediction_columns() %}
    {{ return([
        ['source_file', 'varchar'], ['date', 'date'], ['game_id', 'varchar'], ['player_id', 'bigint'],
        ['player_name', 'varchar'], ['team', 'varchar'], ['opponent', 'varchar'], ['home', 'boolean'],
        ['pred_pts', 'double'], ['pred_reb', 'double'], ['pred_ast', 'double'],
        ['pts_mean_last10', 'double'], ['reb_mean_last10', 'double'], ['ast_mean_last10', 'double'],
        ['pts_mean_season', 'double'], ['reb_mean_season', 'double'], ['ast_mean_season', 'double'],
        ['games_played_season', 'integer'], ['model_revision', 'varchar'], ['dataset_revision', 'varchar'],
        ['generated_at', 'varchar'],
        ['minutes_mean_last10', 'double'], ['input_staleness_days', 'double'], ['tip_off_utc', 'timestamp'],
        ['event_id', 'varchar'], ['schedule_source', 'varchar'],
    ]) }}
{% endmacro %}

{% macro residual_columns() %}
    {{ return([
        ['source_file', 'varchar'], ['date', 'date'], ['game_id', 'varchar'], ['player_id', 'bigint'],
        ['player_name', 'varchar'], ['team', 'varchar'], ['opponent', 'varchar'], ['home', 'boolean'],
        ['pred_pts', 'double'], ['pred_reb', 'double'], ['pred_ast', 'double'],
        ['actual_pts', 'double'], ['actual_reb', 'double'], ['actual_ast', 'double'],
        ['resid_pts', 'double'], ['resid_reb', 'double'], ['resid_ast', 'double'], ['minutes', 'double'],
        ['pts_mean_last10', 'double'], ['reb_mean_last10', 'double'], ['ast_mean_last10', 'double'],
        ['has_actual', 'boolean'], ['game_ingested', 'boolean'], ['in_metrics_population', 'boolean'],
        ['model_revision', 'varchar'], ['dataset_revision', 'varchar'],
        ['resolved_game_id', 'varchar'], ['tip_off_utc', 'timestamp'],
    ]) }}
{% endmacro %}

{#- Columns added after a product family's first files were written (ADR-0029, ADR-0030):
    a family whose files all predate them selects typed nulls instead (typed_select absent=). -#}
{% macro residual_columns_added_2026_10() %}
    {{ return(['resolved_game_id', 'tip_off_utc']) }}
{% endmacro %}

{#- select list casting every [name, type] pair from a relation alias, `filename` -> source_file;
    names in `absent` are not in any file of the family and come out as typed nulls. -#}
{% macro typed_select(columns, alias, absent=[]) -%}
{% for name, type in columns -%}
    {% if name in absent -%}
    cast(null as {{ type }}) as {{ name }}{{ "," if not loop.last }}
    {%- else -%}
    cast({{ alias }}.{{ 'filename' if name == 'source_file' else name }} as {{ type }}) as {{ name }}{{ "," if not loop.last }}
    {%- endif %}
{% endfor -%}
{%- endmacro %}

{#- Column list of the private lines files (nba/lines/capture.py LINE_COLUMNS), typed. -#}
{% macro line_columns() %}
    {{ return([
        ['source_file', 'varchar'], ['capture_id', 'varchar'], ['snapshot_at_utc', 'timestamp'], ['source', 'varchar'],
        ['book', 'varchar'], ['event_id', 'varchar'], ['commence_time_utc', 'timestamp'], ['game_date', 'date'],
        ['home_team', 'varchar'], ['away_team', 'varchar'], ['team', 'varchar'], ['player_name_raw', 'varchar'],
        ['player_id', 'bigint'], ['match_method', 'varchar'], ['market', 'varchar'], ['line', 'double'],
        ['over_price', 'integer'], ['under_price', 'integer'], ['line_last_update_utc', 'timestamp'],
    ]) }}
{% endmacro %}
