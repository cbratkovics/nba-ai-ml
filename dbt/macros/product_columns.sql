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
    ]) }}
{% endmacro %}

{#- select list casting every [name, type] pair from a relation alias, `filename` -> source_file. -#}
{% macro typed_select(columns, alias) -%}
{% for name, type in columns -%}
    cast({{ alias }}.{{ 'filename' if name == 'source_file' else name }} as {{ type }}) as {{ name }}{{ "," if not loop.last }}
{% endfor -%}
{%- endmacro %}
