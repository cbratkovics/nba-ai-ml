{#- Column list of the feature module's output file features/asof_v1.parquet
    (nba/warehouse/features.py: asof.ID_COLUMNS + asof.FEATURE_COLUMNS + feature_version), typed.
    Shared by brz_features_asof_v1 and the parity test so the 23 feature names live once in SQL;
    tests/test_feature_parity.py checks the list against the Python module. -#}
{% macro feature_id_columns() %}
    {{ return([
        ['source_file', 'varchar'],
        ['player_id', 'bigint'], ['game_id', 'varchar'], ['game_date', 'date'], ['season', 'varchar'],
        ['team', 'varchar'], ['opponent', 'varchar'],
    ]) }}
{% endmacro %}

{% macro feature_names() %}
    {% set names = [] %}
    {% for stat in ['pts', 'reb', 'ast', 'minutes'] %}
        {% for w in [5, 10, 20] %}
            {% do names.append(stat ~ '_mean_last' ~ w) %}
        {% endfor %}
    {% endfor %}
    {% for stat in ['pts', 'reb', 'ast', 'minutes'] %}
        {% do names.append(stat ~ '_mean_season') %}
    {% endfor %}
    {% do names.extend(['games_played_season', 'days_rest', 'back_to_back', 'home']) %}
    {% for stat in ['pts', 'reb', 'ast'] %}
        {% do names.append(stat ~ '_mean_vs_opp') %}
    {% endfor %}
    {{ return(names) }}
{% endmacro %}

{#- Features the module emits as integers (exact comparison); every other feature is a double. -#}
{% macro integer_feature_names() %}
    {{ return(['games_played_season', 'back_to_back', 'home']) }}
{% endmacro %}

{% macro feature_columns() %}
    {% set cols = feature_id_columns() %}
    {% for name in feature_names() %}
        {% do cols.append([name, 'integer' if name in integer_feature_names() else 'double']) %}
    {% endfor %}
    {% do cols.append(['feature_version', 'varchar']) %}
    {{ return(cols) }}
{% endmacro %}
